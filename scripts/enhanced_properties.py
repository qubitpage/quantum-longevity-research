#!/usr/bin/env python3
"""
Enhanced Molecular Properties — HOMO-LUMO, Dipole, Reactivity Descriptors
==========================================================================
Computes additional molecular properties for the longevity compounds to 
complement the VQE correlation energy analysis.

Properties computed:
  - HOMO-LUMO gap (chemical hardness / reactivity)
  - Dipole moment (polarity / solubility indicator)
  - Ionization potential (IP) and electron affinity (EA) via Koopmans
  - Chemical hardness η = (IP - EA) / 2
  - Chemical potential μ = -(IP + EA) / 2
  - Electrophilicity index ω = μ² / (2η)
  - Mulliken charges (charge distribution)
  - Molecular weight and formula
  - Comparison across two basis sets: STO-3G and 6-31G

Uses PySCF DFT (B3LYP) for more accurate orbital energies.
"""

import json
import time
import numpy as np
from pathlib import Path
from pyscf import gto, scf, dft

RESULTS_DIR = Path("/opt/research/results")

# Same compounds as rigorous_gpu_sim.py
COMPOUNDS = [
    {
        "id": "LNG-001", "name": "NMN (Nicotinamide Mononucleotide)",
        "target": "NAMPT", "pathway": "NAD+/Sirtuins",
        "atoms": """
            C  0.000  0.000  0.000;  C  1.395  0.000  0.000;  C  2.093  1.209  0.000
            N  1.395  2.418  0.000;  C  0.000  2.418  0.000;  C -0.698  1.209  0.000
            C  3.573  1.209  0.000;  O  4.271  2.160  0.000;  N  4.100  0.000  0.000
            H  4.959  0.000  0.000;  H  3.573 -0.850  0.000;  H -0.525 -0.951  0.000
            H  1.920 -0.951  0.000;  H -0.525  3.369  0.000;  H -1.788  1.209  0.000
        """,
    },
    {
        "id": "LNG-002", "name": "Resveratrol",
        "target": "SIRT1/SIRT3", "pathway": "NAD+/Sirtuins",
        "atoms": """
            C  0.000  0.000  0.000;  C  1.395  0.000  0.000;  C  2.093  1.209  0.000
            C  1.395  2.418  0.000;  C  0.000  2.418  0.000;  C -0.698  1.209  0.000
            C  3.573  1.209  0.000;  C  4.271  2.418  0.000;  C  5.751  2.418  0.000
            C  6.449  1.209  0.000;  C  5.751  0.000  0.000;  C  4.271  0.000  0.000
            O -2.098  1.209  0.000;  O  6.449  3.627  0.000;  O  7.849  1.209  0.000
        """,
    },
    {
        "id": "LNG-003", "name": "Rapamycin (binding fragment)",
        "target": "mTOR/FKBP12", "pathway": "mTOR/Autophagy",
        "atoms": """
            C  0.000  0.000  0.000;  C  1.500  0.000  0.000;  C  2.250  1.299  0.000
            O  1.750  2.400  0.000;  C  3.750  1.299  0.000;  C  4.500  0.000  0.000
            O  4.000 -1.100  0.000;  C  6.000  0.000  0.000;  C  6.750  1.299  0.000
            O  8.150  1.299  0.000;  H  6.750 -0.951  0.000;  H -0.525  0.951  0.000
        """,
    },
    {
        "id": "LNG-004", "name": "Metformin",
        "target": "AMPK/Complex I", "pathway": "AMPK/Metabolism",
        "atoms": """
            N  0.000  0.000  0.000;  C  1.340  0.000  0.000;  N  2.010  1.160  0.000
            N  2.010 -1.160  0.000;  C  3.460 -1.160  0.000;  N  4.130  0.000  0.000
            N  4.130 -2.320  0.000;  H  0.000  0.940  0.000;  H  0.000 -0.940  0.000
            H  1.510  1.990  0.000;  H  4.670  0.830  0.000;  H  3.630 -3.150  0.000
            H  5.120 -2.320  0.000;  H  2.510 -2.050  0.000
        """,
    },
    {
        "id": "LNG-005", "name": "Quercetin",
        "target": "BCL-2/PI3K", "pathway": "Senolytic/Apoptosis",
        "atoms": """
            C  0.000  0.000  0.000;  C  1.395  0.000  0.000;  C  2.093  1.209  0.000
            C  1.395  2.418  0.000;  C  0.000  2.418  0.000;  C -0.698  1.209  0.000
            O  2.093  3.627  0.000;  C  3.493  3.627  0.000;  C  4.191  2.418  0.000
            C  3.493  1.209  0.000;  O  5.591  2.418  0.000;  O  4.191  4.836  0.000
            O  4.191  0.000  0.000;  O -0.698  3.627  0.000;  O -2.098  1.209  0.000
        """,
    },
    {
        "id": "LNG-006", "name": "Fisetin",
        "target": "BCL-2/BCL-XL", "pathway": "Senolytic/Apoptosis",
        "atoms": """
            C  0.000  0.000  0.000;  C  1.395  0.000  0.000;  C  2.093  1.209  0.000
            C  1.395  2.418  0.000;  C  0.000  2.418  0.000;  C -0.698  1.209  0.000
            O  2.093  3.627  0.000;  C  3.493  3.627  0.000;  C  4.191  2.418  0.000
            C  3.493  1.209  0.000;  O  5.591  2.418  0.000;  O  4.191  4.836  0.000
            O -0.698  3.627  0.000;  O  2.093 -1.209  0.000;  H -1.788  1.209  0.000
        """,
    },
    {
        "id": "LNG-007", "name": "Dasatinib (pyrimidine core)",
        "target": "Src/BCL-2", "pathway": "Senolytic/Tyrosine kinase",
        "atoms": """
            C  0.000  0.000  0.000;  N  1.340  0.000  0.000;  C  2.010  1.160  0.000
            C  1.340  2.320  0.000;  N  0.000  2.320  0.000;  C -0.670  1.160  0.000
            N  2.010  3.480  0.000;  C  3.460  1.160  0.000;  S  4.160  2.500  0.000
            Cl  5.900  2.500  0.000;  H -1.760  1.160  0.000;  H -0.525 -0.951  0.000
        """,
    },
    {
        "id": "LNG-008", "name": "Spermidine",
        "target": "EP300/Autophagy", "pathway": "Autophagy/Epigenetics",
        "atoms": """
            N  0.000  0.000  0.000;  C  1.470  0.000  0.000;  C  2.205  1.350  0.000
            C  3.675  1.350  0.000;  N  4.410  0.000  0.000;  C  5.880  0.000  0.000
            C  6.615  1.350  0.000;  C  8.085  1.350  0.000;  C  8.820  0.000  0.000
            N 10.290  0.000  0.000;  H  0.000  0.940  0.000;  H  0.000 -0.940  0.000
            H  4.410  0.940  0.000;  H 10.290  0.940  0.000;  H 10.290 -0.940  0.000
        """,
    },
    {
        "id": "LNG-009", "name": "Urolithin A",
        "target": "Mitophagy/PINK1", "pathway": "Mitophagy/Mitochondria",
        "atoms": """
            C  0.000  0.000  0.000;  C  1.395  0.000  0.000;  C  2.093  1.209  0.000
            C  1.395  2.418  0.000;  C  0.000  2.418  0.000;  C -0.698  1.209  0.000
            O  2.790  2.418  0.000;  C  4.190  2.418  0.000;  C  4.888  1.209  0.000
            C  4.190  0.000  0.000;  C  2.790  0.000  0.000;  C  6.288  1.209  0.000
            O  6.986  2.418  0.000;  O  6.986  0.000  0.000;  C  4.888  3.627  0.000
        """,
    },
    {
        "id": "LNG-010", "name": "Alpha-Ketoglutarate",
        "target": "TET enzymes", "pathway": "TCA cycle/Epigenetics",
        "atoms": """
            C  0.000  0.000  0.000;  C  1.500  0.000  0.000;  C  2.250  1.299  0.000
            C  3.750  1.299  0.000;  C  5.250  1.299  0.000;  O  0.000  1.200  0.000
            O -1.100 -0.600  0.000;  O  1.400  2.310  0.000;  C  5.600  0.000  0.000
            O  6.300  1.050  0.000;  O  6.300 -1.050  0.000;  H  1.400 -2.310  0.000
        """,
    },
]


def compute_properties(compound, basis="sto-3g"):
    """Compute DFT-level molecular properties."""
    atoms_str = compound["atoms"].replace(";", "\n").strip()
    
    # Count total electrons to detect odd-electron systems
    atom_Z = {"H": 1, "He": 2, "C": 6, "N": 7, "O": 8, "F": 9, "S": 16, "Cl": 17, "P": 15, "Br": 35}
    total_e = 0
    for line in atoms_str.split("\n"):
        parts = line.strip().split()
        if len(parts) >= 4:
            total_e += atom_Z.get(parts[0], 0)
    
    spin = total_e % 2  # 1 if odd electrons
    
    mol = gto.Mole()
    mol.atom = atoms_str
    mol.basis = basis
    mol.charge = 0
    mol.spin = spin
    mol.verbose = 0
    mol.build()
    
    props = {
        "compound_id": compound["id"],
        "name": compound["name"],
        "target": compound["target"],
        "pathway": compound["pathway"],
        "basis": basis,
        "n_atoms": mol.natm,
        "n_electrons": mol.nelectron,
        "formula": {},
        "molecular_weight": 0.0,
    }
    
    # Count formula
    atom_masses = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "S": 32.06, "Cl": 35.45, "P": 30.97, "Br": 79.90}
    for i in range(mol.natm):
        sym = mol.atom_symbol(i)
        props["formula"][sym] = props["formula"].get(sym, 0) + 1
        props["molecular_weight"] += atom_masses.get(sym, 0)
    
    formula_str = ""
    for el in ["C", "H", "N", "O", "S", "Cl", "P", "Br"]:
        if el in props["formula"]:
            formula_str += f"{el}{props['formula'][el]}" if props['formula'][el] > 1 else el
    props["formula_str"] = formula_str
    
    # DFT calculation (B3LYP for better orbital energies)
    if spin == 0:
        mf = dft.RKS(mol)
    else:
        mf = dft.ROKS(mol)
    mf.xc = "b3lyp"
    mf.conv_tol = 1e-10
    mf.max_cycle = 200
    mf.kernel()
    
    props["dft_energy"] = float(mf.e_tot)
    props["converged"] = bool(mf.converged)
    
    # Also do HF for comparison
    if spin == 0:
        mf_hf = scf.RHF(mol)
    else:
        mf_hf = scf.ROHF(mol)
    mf_hf.conv_tol = 1e-10
    mf_hf.max_cycle = 200
    mf_hf.kernel()
    props["hf_energy"] = float(mf_hf.e_tot)
    
    # Orbital energies
    mo_energies = mf.mo_energy
    mo_occ = mf.mo_occ
    
    occ_idx = np.where(mo_occ > 0)[0]
    virt_idx = np.where(mo_occ == 0)[0]
    
    if len(occ_idx) > 0 and len(virt_idx) > 0:
        homo_idx = occ_idx[-1]
        lumo_idx = virt_idx[0]
        homo = float(mo_energies[homo_idx])
        lumo = float(mo_energies[lumo_idx])
        gap = lumo - homo
        
        props["homo_energy_ha"] = homo
        props["lumo_energy_ha"] = lumo
        props["homo_energy_ev"] = homo * 27.2114
        props["lumo_energy_ev"] = lumo * 27.2114
        props["homo_lumo_gap_ha"] = gap
        props["homo_lumo_gap_ev"] = gap * 27.2114
        props["homo_lumo_gap_kcal"] = gap * 627.509
        
        # Koopmans' theorem approximations
        IP = -homo  # ionization potential
        EA = -lumo  # electron affinity
        props["ionization_potential_ev"] = IP * 27.2114
        props["electron_affinity_ev"] = EA * 27.2114
        
        # Chemical reactivity descriptors
        eta = (IP - EA) / 2  # chemical hardness
        mu = -(IP + EA) / 2  # chemical potential
        omega = mu**2 / (2 * eta) if eta > 1e-10 else 0  # electrophilicity
        
        props["chemical_hardness_ha"] = float(eta)
        props["chemical_hardness_ev"] = float(eta * 27.2114)
        props["chemical_potential_ha"] = float(mu)
        props["chemical_potential_ev"] = float(mu * 27.2114)
        props["electrophilicity_ha"] = float(omega)
        props["electrophilicity_ev"] = float(omega * 27.2114)
    
    # Dipole moment
    dm = mf.make_rdm1()
    dip = mf.dip_moment(mol, dm, verbose=0)
    dip_total = np.linalg.norm(dip)
    props["dipole_x"] = float(dip[0])
    props["dipole_y"] = float(dip[1])
    props["dipole_z"] = float(dip[2])
    props["dipole_total_debye"] = float(dip_total)
    
    # Mulliken charges
    mulliken = mf.mulliken_pop(verbose=0)
    # mulliken returns (pop, chg) — charges
    if isinstance(mulliken, tuple) and len(mulliken) >= 2:
        charges = mulliken[1]
        props["mulliken_charges"] = [float(c) for c in charges]
        props["max_positive_charge"] = float(max(charges))
        props["max_negative_charge"] = float(min(charges))
        props["charge_span"] = float(max(charges) - min(charges))
    
    # Number of basis functions
    props["n_basis_functions"] = mol.nao_nr()
    props["n_molecular_orbitals"] = len(mo_energies)
    
    return props


def main():
    print("=" * 90)
    print("ENHANCED MOLECULAR PROPERTIES — LONGEVITY COMPOUNDS")
    print("Method: DFT B3LYP + Koopmans' theorem reactivity descriptors")
    print("=" * 90)
    
    all_results = []
    
    for basis in ["sto-3g", "6-31g"]:
        print(f"\n{'━' * 90}")
        print(f"  BASIS SET: {basis}")
        print(f"{'━' * 90}")
        
        for i, compound in enumerate(COMPOUNDS, 1):
            t0 = time.time()
            print(f"\n  [{i}/{len(COMPOUNDS)}] {compound['name']} ({basis})")
            
            try:
                props = compute_properties(compound, basis)
                dt = time.time() - t0
                props["compute_time_s"] = dt
                
                print(f"    Formula: {props['formula_str']} (MW={props['molecular_weight']:.1f})")
                print(f"    DFT(B3LYP):  {props['dft_energy']:.6f} Ha")
                print(f"    HOMO:        {props.get('homo_energy_ev', 0):.3f} eV")
                print(f"    LUMO:        {props.get('lumo_energy_ev', 0):.3f} eV")
                print(f"    HOMO-LUMO:   {props.get('homo_lumo_gap_ev', 0):.3f} eV ({props.get('homo_lumo_gap_kcal', 0):.1f} kcal/mol)")
                print(f"    Dipole:      {props['dipole_total_debye']:.3f} Debye")
                print(f"    Hardness:    {props.get('chemical_hardness_ev', 0):.3f} eV")
                print(f"    Electrophil: {props.get('electrophilicity_ev', 0):.3f} eV")
                print(f"    IP (Koop):   {props.get('ionization_potential_ev', 0):.3f} eV")
                print(f"    EA (Koop):   {props.get('electron_affinity_ev', 0):.3f} eV")
                print(f"    Converged:   {props['converged']} ({dt:.1f}s)")
                
                all_results.append(props)
                
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()
    
    # Summary comparison table
    print(f"\n\n{'═' * 120}")
    print("  COMPARATIVE TABLE — 6-31G BASIS")
    print(f"{'═' * 120}")
    
    results_631g = [r for r in all_results if r["basis"] == "6-31g"]
    results_631g.sort(key=lambda r: r.get("homo_lumo_gap_ev", 999))
    
    print(f"\n{'Rank':<5} {'Compound':<32} {'Formula':<12} {'MW':<8} "
          f"{'Gap(eV)':<9} {'Dipole(D)':<10} {'η(eV)':<8} {'ω(eV)':<8} {'IP(eV)':<8}")
    print("─" * 120)
    
    for rank, r in enumerate(results_631g, 1):
        print(f"{rank:<5} {r['name'][:31]:<32} {r['formula_str']:<12} "
              f"{r['molecular_weight']:<8.1f} "
              f"{r.get('homo_lumo_gap_ev', 0):>7.3f}  "
              f"{r['dipole_total_debye']:>8.3f}  "
              f"{r.get('chemical_hardness_ev', 0):>6.3f}  "
              f"{r.get('electrophilicity_ev', 0):>6.3f}  "
              f"{r.get('ionization_potential_ev', 0):>6.3f}")
    
    # Drug-likeness assessment (simplified Lipinski)
    print(f"\n\n{'═' * 90}")
    print("  DRUG-LIKENESS ASSESSMENT (Simplified)")
    print(f"{'═' * 90}")
    
    for r in all_results:
        if r["basis"] != "6-31g":
            continue
        f = r["formula"]
        mw = r["molecular_weight"]
        gap = r.get("homo_lumo_gap_ev", 0)
        dipole = r["dipole_total_debye"]
        
        score = 0
        notes = []
        # MW < 500
        if mw < 500:
            score += 1
            notes.append("MW<500 ✓")
        else:
            notes.append(f"MW={mw:.0f} ✗")
        
        # Reasonable HOMO-LUMO gap (not too reactive, not too stable)
        if 3.0 < gap < 12.0:
            score += 1
            notes.append(f"Gap={gap:.1f}eV ✓")
        else:
            notes.append(f"Gap={gap:.1f}eV ✗")
        
        # Dipole > 1 (soluble) but < 15 (too polar)
        if 1.0 < dipole < 15.0:
            score += 1
            notes.append(f"Dip={dipole:.1f}D ✓")
        else:
            notes.append(f"Dip={dipole:.1f}D ✗")
        
        # Hardness > 2 eV (stable)
        if r.get("chemical_hardness_ev", 0) > 2.0:
            score += 1
            notes.append(f"η={r.get('chemical_hardness_ev',0):.1f} ✓")
        else:
            notes.append(f"η={r.get('chemical_hardness_ev',0):.1f} ✗")
        
        print(f"  {r['name'][:35]:<36} Score: {score}/4  [{', '.join(notes)}]")
    
    # Save all results
    out_path = RESULTS_DIR / "enhanced_properties.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved: {out_path}")
    print(f"{'═' * 90}")


if __name__ == "__main__":
    main()
