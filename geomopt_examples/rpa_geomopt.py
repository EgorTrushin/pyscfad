#!/usr/bin/env python3

import jax
import time
from pyscf import df as pyscf_df
from pyscfad import gto, dft, scf, df
from pyscfad.gw import rpa
from pyscf.geomopt.berny_solver import optimize, to_berny_geom
from berny import Berny, geomlib
import warnings
warnings.simplefilter("ignore")

def energy(mol, with_df):
    mf = dft.RKS(mol)
    mf.xc = 'pbe'
    mf.kernel(dm0=None)

    mymp = rpa.RPA(mf)
    mymp.with_df = with_df
    mymp.kernel()
    return mymp.e_tot

def solver(geom, val_and_grad, basis):
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = geom
    mol.basis = basis
    mol.build(trace_exp=False, trace_ctr_coeff=False)

    auxbasis = pyscf_df.addons.make_auxbasis(mol, mp2fit=True)
    auxmol = df.addons.make_auxmol(mol, auxbasis)
    with_df = df.DF(mol, auxmol=auxmol)

    e_tot, jac = val_and_grad(mol, with_df)

    return e_tot, jac[0].coords + jac[1].mol.coords + jac[1].auxmol.coords


def get_geom(geom_str):
    mol_ = gto.Mole()
    mol_.build(atom = geom_str, basis = 'sto-3g')
    geom = to_berny_geom(mol_)
    return geom

ts = time.time()

GEOM = '''H 0 0 0; H 0 0 1.'''
BASIS = "augccpvtz"
print("GEOM:", GEOM)
print("BASIS:", BASIS)
optimizer = Berny(get_geom(GEOM))

val_and_grad = jax.value_and_grad(energy, (0,1))

print("Elapsed time:", time.time()-ts, flush=True)

for iter_, geom in enumerate(optimizer):
     print("\n"+30*"*")
     print("OPTIMIZER STEP:", iter_+1)
     print(30*"*")
     print("Geometry:")
     print(geom.coords, flush=True)
     energy, gradients = solver(list(geom), val_and_grad, basis=BASIS)
     print("Energy:", energy)
     print("Gradients:")
     print(gradients)
     optimizer.send((energy, gradients))
     print("Elapsed time:", time.time()-ts, flush=True)
relaxed = geom

print("Relaxed Geometry:")
print(relaxed.coords)
print("Elapsed time:", time.time()-ts)
