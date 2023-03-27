#!/usr/bin/env python3

import jax; jax.config.update('jax_platform_name', 'cpu')
import time
from pyscf import df as pyscf_df
from pyscfad import gto, dft, scf, df
from pyscfad.gw import rpa, sigma
from pyscf.geomopt.berny_solver import optimize, to_berny_geom
from berny import Berny, geomlib
import warnings
warnings.simplefilter("ignore")

def energy_(mol, with_df, method, nw, x0):
    mf = dft.RKS(mol)
    mf.xc = 'pbe'
    mf.kernel(dm0=None)

    if method == 'RPA':
       mymp = rpa.RPA(mf, nw=40, x0=0.5)
    elif method == 'SIGMA':
       mymp = sigma.SIGMA(mf)
    mymp.with_df = with_df
    mymp.kernel()
    return mymp.e_tot

def solver(geom, val_and_grad, basis, method, nw, x0):
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = geom
    mol.basis = basis
    mol.build(trace_exp=False, trace_ctr_coeff=False)

    auxbasis = pyscf_df.addons.make_auxbasis(mol, mp2fit=True)
    auxmol = df.addons.make_auxmol(mol, auxbasis)
    with_df = df.DF(mol, auxmol=auxmol)

    e_tot, jac = val_and_grad(mol, with_df, method, nw, x0)

    return e_tot, jac[0].coords + jac[1].mol.coords + jac[1].auxmol.coords

def get_geom(geom_str):
    mol_ = gto.Mole()
    mol_.build(atom = geom_str, basis = 'sto-3g')
    geom = to_berny_geom(mol_)
    return geom

def optimize_geom(geom_, basis, method, nw, x0, verbose):
    ts = time.time()
    print('starting geometry:', geom_)
    print('basis:', basis)
    print('method:', method)
    print('nw:', nw)
    print('x0:', x0)

    optimizer = Berny(get_geom(geom_))

    val_and_grad = jax.value_and_grad(energy_, (0,1))

    print('\nTime before optimization:', time.time()-ts, flush=True)

    print('')
    for iter_, geom in enumerate(optimizer):
        energy, gradients = solver(list(geom), val_and_grad, basis, method, nw, x0)
        optimizer.send((energy, gradients))
        print(f'iter={iter_+1}   energy={energy:.10f}   elapsed time={time.time()-ts:.2f} seconds', flush=True)
        if verbose:
            print('\nGeometry:')
            print(geom.coords, flush=True)
            print('\nGradients:')
            print(gradients)
            print('')

    print('\nOptimized feometry:')
    print(geom.coords)

optimize_geom(geom_ = '''H 0 0 0; H 0 0 1.''',
              basis = 'ccpvdz',
              method = 'RPA',
              nw=40,
              x0=0.5,
              verbose = False)

