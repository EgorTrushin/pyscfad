import numpy
import jax
from jax import vmap, jit
from pyscf import lib as pyscf_lib
from pyscf.lib import logger
from pyscf import df as pyscf_df
from pyscf.gw import sigma as pyscf_sigma
from pyscfad import util
from pyscfad.lib import numpy as np
from pyscfad import gto, scf, dft, df
from pyscf.gw.sigma_utils import get_spline_coeffs

#from numpy import linalg

def kernel(rpa, mo_energy, mo_coeff, Lpq=None, nw=None, verbose=logger.NOTE):
    mf = rpa._scf
    # only support frozen core
    if rpa.frozen is not None:
        assert isinstance(rpa.frozen, int)
        assert rpa.frozen < rpa.nocc

    if Lpq is None:
        Lpq = rpa.ao2mo(mo_coeff)

    # Grids for integration on imaginary axis
    freqs, wts = pyscf_sigma._get_scaled_legendre_roots(nw, 0.5)

    # Compute HF exchange energy (EXX)
    dm = mf.make_rdm1()
    rhf = scf.RHF(rpa.mol)
    e_hf = rhf.energy_elec(dm=dm)[0]
    e_hf += mf.energy_nuc()

    # Compute RPA correlation energy
    e_corr = get_rpa_ecorr(rpa, Lpq, freqs, wts)

    # Compute totol energy
    e_tot = e_hf + e_corr

    logger.debug(rpa, '  RPA total energy = %s', e_tot)
    logger.debug(rpa, '  EXX energy = %s, RPA corr energy = %s', e_hf, e_corr)

    return e_tot, e_hf, e_corr

@jit
def get_rho_response(omega, mo_energy, Lpq):
    """
    Compute density response function in auxiliary basis at freq iw.
    """
    nocc = Lpq.shape[1]
    eia = mo_energy[:nocc, None] - mo_energy[None, nocc:]
    eia = eia / (omega**2 + eia * eia)
    # Response from both spin-up and spin-down density
    Pia = Lpq * (eia * 4.0)
    Pi = np.einsum('Pia, Qia -> PQ', Pia, Lpq)
    return Pi

def get_rpa_ecorr(rpa, Lpq, freqs, wts):
    """
    Compute RPA correlation energy
    """
    mol = rpa.mol
    mf = rpa._scf
    dm = mf.make_rdm1()
    rks = dft.RKS(mol, xc=mf.xc)
    veff = rks.get_veff(mol, dm)
    h1e = rks.get_hcore(mol)
    s1e = rks.get_ovlp(mol)
    fock = rks.get_fock(h1e, s1e, veff, dm)
    mo_energy, _ = rks.eig(fock, s1e)

    #mo_energy = _mo_energy_without_core(rpa, rpa._scf.mo_energy)
    mo_energy = pyscf_sigma._mo_energy_without_core(rpa, mo_energy)
    nocc = rpa.nocc
    naux = Lpq.shape[0]

    if (mo_energy[nocc] - mo_energy[nocc-1]) < 1e-3:
        logger.warn(rpa, 'Current RPA code not well-defined for degeneracy!')

    def body(omega, weight, x, c):
        Pi = get_rho_response(omega, mo_energy, Lpq[:, :nocc, nocc:])
        sigmas, _ = jax.numpy.linalg.eigh(-Pi)
        ec_w_rpa = 0.
        e_corr_i = 0.
        #ec_w_sigma = 0.
        for sigma in sigmas:
            #if sigma > 0.:
            ec_w_rpa += np.log(1.+sigma) - sigma - cspline_integr(c, x, sigma)
                #ec_w_sigma += - cspline_integr(c, x, sigma)
            #else:
            #    assert abs(sigma) < 1.0e-14
        e_corr_i += 1./(2.*np.pi) * ec_w_rpa * weight
#        ec_w  = np.log(np.linalg.det(np.eye(naux) - Pi))
#        ec_w += np.trace(Pi)
#        e_corr_i = 1./(2.*numpy.pi) * ec_w * weight
        return e_corr_i

    x, c = get_spline_coeffs(logger, rpa)
    x, c = jax.numpy.array(x), jax.numpy.array(c)
    #e_corr_i = vmap(body, in_axes=(0, 0, None, None))(freqs, wts, x, c)
    #e_corr = np.sum(e_corr_i)

    e_corr_rpa = 0.
    e_corr_sigma = 0.
    for w in range(len(freqs)):
        Pi = get_rho_response(freqs[w], mo_energy, Lpq[:, :nocc, nocc:])
        sigmas, _ = jax.numpy.linalg.eigh(-Pi)
        ec_w_rpa = 0.
        ec_w_sigma = 0.
        for sigma in sigmas:
            if sigma > 0.:
                ec_w_rpa += np.log(1.+sigma) - sigma
                ec_w_sigma += - cspline_integr(c, x, sigma)
            else:
                assert abs(sigma) < 1.0e-14
        e_corr_rpa += 1./(2.*np.pi) * ec_w_rpa * wts[w]
        e_corr_sigma += 1./(2.*np.pi) * ec_w_sigma * wts[w]
    return e_corr_sigma

@util.pytree_node(['_scf','mol','with_df','mo_energy','mo_coeff'], num_args=1)
class SIGMA(pyscf_sigma.SIGMA):
    def __init__(self, mf, frozen=None, auxbasis=None, param='s2', **kwargs):
        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory
        self.frozen = frozen
        self.with_df = None
        self.param = 's2'

##################################################
# don't modify the following attributes, they are not input options
        self._nocc = None
        self._nmo = None
        self.mo_energy = mf.mo_energy
        self.mo_coeff = mf.mo_coeff
        self.mo_occ = mf.mo_occ
        self.e_corr = None
        self.e_hf = None
        self.e_tot = None

        self.__dict__.update(kwargs)
        if self.with_df is None:
            if getattr(self._scf, 'with_df', None):
                self.with_df = self._scf.with_df
            else:
                if auxbasis is None:
                    auxbasis = pyscf_df.addons.make_auxbasis(self.mol, mp2fit=True)
                auxmol = df.addons.make_auxmol(self.mol, auxbasis)
                self.with_df = df.DF(self.mol, auxmol=auxmol)

    def kernel(self, mo_energy=None, mo_coeff=None, Lpq=None, nw=40):
        """
        Args:
            mo_energy : 1D array (nmo), mean-field mo energy
            mo_coeff : 2D array (nmo, nmo), mean-field mo coefficient
            Lpq : 3D array (naux, nmo, nmo), 3-index ERI
            nw: interger, grid number
        Returns:
            self.e_tot : RPA total eenrgy
            self.e_hf : EXX energy
            self.e_corr : RPA correlation energy
        """
        if mo_coeff is None:
            mo_coeff = pyscf_sigma._mo_without_core(self, self._scf.mo_coeff)
        if mo_energy is None:
            mo_energy = pyscf_sigma._mo_energy_without_core(self, self._scf.mo_energy)

        log = logger.new_logger(self)
        self.dump_flags()
        self.e_tot, self.e_hf, self.e_corr = \
                        kernel(self, mo_energy, mo_coeff, Lpq=Lpq, nw=nw, verbose=self.verbose)

        log.timer('RPA')
        del log
        return self.e_corr

    def ao2mo(self, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        nmo = self.nmo
        naux = self.with_df.get_naoaux()
        mem_incore = (2 * nmo**2*naux) * 8 / 1e6
        mem_now = pyscf_lib.current_memory()[0]

        if (mem_incore + mem_now < 0.99 * self.max_memory) or self.mol.incore_anyway:
            Lpq = np.einsum("lpq,pi,qj->lij", self.with_df._cderi, mo_coeff, mo_coeff)
            return Lpq
        else:
            raise RuntimeError("not enough memory")


def intervalnum(x, s):
    """Determine to which interval s belongs in x.

    Args:
        x: List of non-negative real numbers in increasing order.
        s: Positive real number.

    Returns:
        inum: The number of interval.
    """

    #assert s > 0.  # verify that s is positive
    #assert all(i > -1e-24 for i in x)  # verify that all x-values are positive
    #assert x == sorted(x)  # verify that x-values are in increasing order

    # find interval
    inum = -1
    if s > x[-1]:
        inum = len(x)
    for i in range(0, len(x)-1):
        if (s > x[i] and (s <= x[i+1])):
            inum = i+1

    assert inum != -1  # verify that an interval was determined

    return inum


def cspline_integr(c, x, s):
    """Integrate analytically cubic spline representation of sigma-functional
       'correction' from 0 to s.

    First interval of spline is treated as linear.
    Last interval of spline is treated as a constant.

    Args:
        c: Coefficients of spline
        x: Ordinates of spline. Have to be non-negative and increasingly order
        s: Sigma-value for which one integrate. Has to be positive.

    Returns:
        integral: resulting integral
    """
    m = intervalnum(x, s)  # determine to which interval s belongs

    # evaluate integral
    integral = 0.
    if m == 1:
        integral = 0.5*c[1][0]*s
    if m > 1 and m < len(x):
        h = s-x[m-1]
        integral = 0.5*c[1][0]*x[1]**2/s + (c[0][m-1]*h + c[1][m-1]/2.*h**2 + c[2][m-1]/3.*h**3 + c[3][m-1]/4.*h**4)/s
        for i in range(2, m):
            h = x[i]-x[i-1]
            integral += (c[0][i-1]*h + c[1][i-1]/2.*h**2 + c[2][i-1]/3.*h**3 + c[3][i-1]/4.*h**4)/s
    if m == len(x):
        integral = 0.5*c[1][0]*x[1]**2/s
        for i in range(2, m):
            h = x[i]-x[i-1]
            integral += (c[0][i-1]*h + c[1][i-1]/2.*h**2 + c[2][i-1]/3.*h**3 + c[3][i-1]/4.*h**4)/s
        integral += c[0][-1]*(1.-x[-1]/s)

    return integral*s