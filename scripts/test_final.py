"""FINAL test: OpenFermion uses h[p,q,r,s] = (ps|qr) = chemist.transpose(0,2,3,1)"""
import numpy as np
from pyscf import gto, scf, mcscf, ao2mo
from openfermion import InteractionOperator, get_sparse_operator
from openfermion.chem.molecular_data import spinorb_from_spatial

mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
mf = scf.RHF(mol); mf.kernel()
mc = mcscf.CASCI(mf, 2, 2); mc.kernel()
h1, e_core = mc.get_h1eff()
h2_raw = mc.get_h2eff()
h2_chem = ao2mo.restore(1, h2_raw, 2)

# CORRECT: OpenFermion convention h[p,q,r,s] = (ps|qr) = transpose(0,2,3,1)
h2_of = h2_chem.transpose(0, 2, 3, 1)
h1_s = h1[:2,:2]
h1_so, h2_so = spinorb_from_spatial(h1_s, h2_of)
ham = InteractionOperator(float(e_core), h1_so, 0.5 * h2_so)
sp = get_sparse_operator(ham)
evals = np.sort(np.linalg.eigvalsh(sp.toarray().real))
print(f'CASCI ref: {mc.e_tot:.10f}')
print(f'Qubit GS:  {evals[0]:.10f}')
print(f'Delta:     {(evals[0] - mc.e_tot)*627.509:.6f} kcal/mol')
print(f'Evals[:4]: {evals[:4]}')

# Also test on a slightly bigger system: H2O CAS(4,4)/STO-3G
print('\n--- H2O CAS(4,4)/STO-3G ---')
mol2 = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='sto-3g', verbose=0)
mf2 = scf.RHF(mol2); mf2.kernel()
mc2 = mcscf.CASCI(mf2, 4, 4); mc2.kernel()
h1_2, e_core_2 = mc2.get_h1eff()
h2_raw_2 = mc2.get_h2eff()
h2_chem_2 = ao2mo.restore(1, h2_raw_2, 4)
h2_of_2 = h2_chem_2.transpose(0, 2, 3, 1)
h1_so_2, h2_so_2 = spinorb_from_spatial(h1_2[:4,:4], h2_of_2)
ham2 = InteractionOperator(float(e_core_2), h1_so_2, 0.5 * h2_so_2)
sp2 = get_sparse_operator(ham2)
evals2 = np.sort(np.linalg.eigvalsh(sp2.toarray().real))
print(f'CASCI ref: {mc2.e_tot:.10f}')
print(f'Qubit GS:  {evals2[0]:.10f}')
print(f'Delta:     {(evals2[0] - mc2.e_tot)*627.509:.6f} kcal/mol')
