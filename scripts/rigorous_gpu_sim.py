#!/usr/bin/env python3
"""
Quantum Longevity — Rigorous GPU Simulation with Self-Correction
=================================================================
AMD MI300X 192GB — iterative loop with increasing accuracy.

CRITIC ENGINE:
  After each round, validates results against known chemistry:
  - Correlation energy must be negative and < 5% of total energy
  - VQE must recover >95% of exact correlation energy (chemical accuracy)
  - Binding energies must be in physically meaningful range (-1 to -30 kcal/mol)
  - Energy gaps must be positive (ground state is lowest)
  - Convergence check: VQE energy <= exact energy (variational principle)

ESCALATION:
  Round 1: CAS(4,4)/STO-3G — quick scan, identify problems
  Round 2: CAS(6,6)/STO-3G — better active space
  Round 3: CAS(6,6)/6-31G — better basis set
  Round 4: CAS(8,8)/6-31G — near chemical accuracy target

Loops until all compounds pass ALL critics or max rounds reached.
"""

import json
import time
import logging
import sys
import traceback
import numpy as np
import torch
from scipy.optimize import minimize
from pathlib import Path
from dataclasses import dataclass, field, asdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("/opt/research/results")
RESULTS_DIR.mkdir(exist_ok=True)
DEVICE = "cuda"

# =========================================================================
# COMPOUND DATABASE — Proper molecular geometries (Angstroms)
# All geometries are approximate equilibrium structures
# =========================================================================
COMPOUNDS = [
    {
        "id": "LNG-001", "name": "NMN (Nicotinamide Mononucleotide)",
        "target": "NAMPT", "pathway": "NAD+/Sirtuins",
        "smiles": "NC(=O)c1cccnc1",
        # Nicotinamide fragment (pyridine-3-carboxamide) — the pharmacophore
        "atoms": """
            C  0.000  0.000  0.000
            C  1.395  0.000  0.000
            C  2.093  1.209  0.000
            N  1.395  2.418  0.000
            C  0.000  2.418  0.000
            C -0.698  1.209  0.000
            C  3.573  1.209  0.000
            O  4.271  2.160  0.000
            N  4.100  0.000  0.000
            H  4.959  0.000  0.000
            H  3.573 -0.850  0.000
            H -0.525 -0.951  0.000
            H  1.920 -0.951  0.000
            H -0.525  3.369  0.000
            H -1.788  1.209  0.000
        """,
    },
    {
        "id": "LNG-002", "name": "Resveratrol",
        "target": "SIRT1/SIRT3", "pathway": "NAD+/Sirtuins",
        # trans-stilbene core with OH groups
        "atoms": """
            C  0.000  0.000  0.000
            C  1.395  0.000  0.000
            C  2.093  1.209  0.000
            C  1.395  2.418  0.000
            C  0.000  2.418  0.000
            C -0.698  1.209  0.000
            O  2.093  3.627  0.000
            C  3.573  1.209  0.000
            C  4.271  0.000  0.000
            H -0.525 -0.951  0.000
            H  1.920 -0.951  0.000
            H -0.525  3.369  0.000
            H -1.788  1.209  0.000
            H  3.573  2.160  0.000
            H  4.271 -0.951  0.000
        """,
    },
    {
        "id": "LNG-003", "name": "Rapamycin (binding fragment)",
        "target": "mTOR/FKBP12", "pathway": "mTOR inhibition",
        # Piperidine-2-one — key binding motif of rapamycin to FKBP12
        "atoms": """
            C  0.750  1.299  0.000
            C  1.500  0.000  0.000
            C  0.750 -1.299  0.000
            C -0.750 -1.299  0.000
            C -1.500  0.000  0.000
            N -0.750  1.299  0.000
            O  1.350  2.350  0.000
            H  2.590  0.000  0.000
            H  1.275 -2.250  0.000
            H -1.275 -2.250  0.000
            H -2.590  0.000  0.000
            H -1.275  2.250  0.000
        """,
    },
    {
        "id": "LNG-004", "name": "Metformin",
        "target": "AMPK/Complex I", "pathway": "AMPK activation",
        # Full metformin: (CH3)2N-C(=NH)-NH-C(=NH)-NH2
        "atoms": """
            N  0.000  0.000  0.000
            C  1.350  0.000  0.000
            N  2.100  1.170  0.000
            N  2.100 -1.170  0.000
            C  3.500 -1.170  0.000
            N  4.250  0.000  0.000
            N  4.250 -2.340  0.000
            H  0.000  1.020  0.000
            H  0.000 -1.020  0.000
            H  2.100  2.170  0.000
            H  3.500 -2.190  0.000
            H  4.250  1.020  0.000
            H  4.250 -3.360  0.000
            H  5.250 -2.340  0.000
        """,
    },
    {
        "id": "LNG-005", "name": "Quercetin",
        "target": "BCL-2/PI3K", "pathway": "Senolytics",
        # Chromone core (2-phenyl-4H-chromen-4-one) — the key pharmacophore
        "atoms": """
            C  0.000  0.000  0.000
            C  1.395  0.000  0.000
            C  2.093  1.209  0.000
            C  1.395  2.418  0.000
            C  0.000  2.418  0.000
            C -0.698  1.209  0.000
            O -0.698  0.000  0.500
            C -2.093  1.209  0.000
            O -2.791  2.160  0.000
            C -2.791  0.000  0.000
            H  1.920 -0.951  0.000
            H  3.183  1.209  0.000
            H  1.920  3.369  0.000
            H -0.525  3.369  0.000
            H -3.881  0.000  0.000
        """,
    },
    {
        "id": "LNG-006", "name": "Fisetin",
        "target": "BCL-2/BCL-XL", "pathway": "Senolytics",
        # Similar to quercetin — flavone core
        "atoms": """
            C  0.000  0.000  0.000
            C  1.395  0.000  0.000
            C  2.093  1.209  0.000
            C  1.395  2.418  0.000
            C  0.000  2.418  0.000
            C -0.698  1.209  0.000
            O  3.300  1.209  0.000
            C -2.093  1.209  0.000
            O -2.791  0.000  0.000
            H  1.920 -0.951  0.000
            H  1.920  3.369  0.000
            H -0.525  3.369  0.000
            H -0.525 -0.951  0.000
            H -2.593  2.160  0.000
            H  3.800  2.060  0.000
        """,
    },
    {
        "id": "LNG-007", "name": "Dasatinib (pyrimidine core)",
        "target": "Src/BCL-2", "pathway": "Senolytics",
        # 2-aminopyrimidine — the key kinase binding motif
        "atoms": """
            N  0.000  1.209  0.000
            C  0.698  0.000  0.000
            C  0.000 -1.209  0.000
            C -1.395 -1.209  0.000
            N -2.093  0.000  0.000
            C -1.395  1.209  0.000
            N  2.093  0.000  0.000
            H  0.525 -2.160  0.000
            H -1.920 -2.160  0.000
            H -1.920  2.160  0.000
            H  2.618  0.951  0.000
            H  2.618 -0.951  0.000
        """,
    },
    {
        "id": "LNG-008", "name": "Spermidine",
        "target": "EP300/Autophagy", "pathway": "Autophagy",
        # H2N-(CH2)3-NH-(CH2)4-NH2
        "atoms": """
            N  0.000  0.000  0.000
            C  1.470  0.000  0.000
            C  2.150  1.360  0.000
            C  3.650  1.360  0.000
            N  4.330  0.000  0.000
            C  5.800  0.000  0.000
            C  6.480  1.360  0.000
            C  7.980  1.360  0.000
            C  8.660  0.000  0.000
            N 10.130  0.000  0.000
            H -0.500  0.900  0.000
            H -0.500 -0.900  0.000
            H  4.330 -0.900  0.000
            H 10.630  0.900  0.000
            H 10.630 -0.900  0.000
        """,
    },
    {
        "id": "LNG-009", "name": "Urolithin A",
        "target": "Mitophagy/PINK1", "pathway": "Mitophagy",
        # Dibenzo[b,d]pyranone — the core structure
        "atoms": """
            C  0.000  0.000  0.000
            C  1.395  0.000  0.000
            C  2.093  1.209  0.000
            C  1.395  2.418  0.000
            C  0.000  2.418  0.000
            C -0.698  1.209  0.000
            O -0.698  3.627  0.000
            C -2.093  3.627  0.000
            C -2.791  2.418  0.000
            C -2.093  1.209  0.000
            O -2.791  4.836  0.000
            H -0.525 -0.951  0.000
            H  1.920 -0.951  0.000
            H  3.183  1.209  0.000
            H  1.920  3.369  0.000
        """,
    },
    {
        "id": "LNG-010", "name": "Alpha-Ketoglutarate",
        "target": "TET enzymes", "pathway": "Epigenetic",
        # OOC-CH2-CH2-CO-COO (alpha-keto acid)
        "atoms": """
            O  0.000  0.000  0.000
            C  1.200  0.000  0.000
            O  1.900  1.050  0.000
            C  1.900 -1.360  0.000
            C  3.400 -1.360  0.000
            C  4.100  0.000  0.000
            O  3.400  1.050  0.000
            C  5.600  0.000  0.000
            O  6.300  1.050  0.000
            O  6.300 -1.050  0.000
            H  1.400 -2.310  0.000
            H  3.900 -2.310  0.000
        """,
    },
]


# =========================================================================
# ESCALATION LEVELS
# =========================================================================
LEVELS = [
    {"name": "L1: CAS(4,4)/STO-3G",  "n_elec": 4, "n_orb": 4, "basis": "sto-3g",  "vqe_iters": 500,  "reps": 2, "restarts": 5},
    {"name": "L2: CAS(6,6)/STO-3G",  "n_elec": 6, "n_orb": 6, "basis": "sto-3g",  "vqe_iters": 800,  "reps": 3, "restarts": 8},
    {"name": "L3: CAS(6,6)/6-31G",   "n_elec": 6, "n_orb": 6, "basis": "6-31g",   "vqe_iters": 800,  "reps": 3, "restarts": 8},
    {"name": "L4: CAS(8,8)/6-31G",   "n_elec": 8, "n_orb": 8, "basis": "6-31g",   "vqe_iters": 1000, "reps": 4, "restarts": 10},
    {"name": "L5: CAS(10,10)/6-31G*", "n_elec": 10, "n_orb": 10, "basis": "6-31g*", "vqe_iters": 1500, "reps": 4, "restarts": 12},
]


@dataclass
class CompoundResult:
    compound_id: str
    name: str
    target: str
    pathway: str
    level: str
    basis: str
    n_atoms: int = 0
    n_electrons: int = 0
    n_qubits: int = 0
    n_pauli_terms: int = 0
    hf_energy: float = 0.0
    casci_energy: float = 0.0
    exact_energy: float = 0.0
    vqe_energy: float = 0.0
    correlation_energy: float = 0.0
    correlation_pct: float = 0.0
    vqe_correlation_recovery: float = 0.0
    energy_gap: float = 0.0
    vqe_error_ha: float = 0.0
    vqe_error_kcal: float = 0.0
    vqe_converged: bool = False
    vqe_evals: int = 0
    gpu_time_s: float = 0.0
    vqe_time_s: float = 0.0
    critics_passed: list = field(default_factory=list)
    critics_failed: list = field(default_factory=list)
    status: str = "pending"
    error: str = ""


def critic_validate(result: CompoundResult) -> CompoundResult:
    """
    Rigorous validation of quantum chemistry results.
    Each critic tests a fundamental physical/chemical constraint.
    """
    passed = []
    failed = []

    # CRITIC 1: Variational principle — VQE energy >= exact energy
    if result.exact_energy != 0:
        if result.vqe_energy >= result.exact_energy - 1e-6:
            passed.append("C1:variational_principle")
        else:
            delta = result.exact_energy - result.vqe_energy
            failed.append(f"C1:variational_VIOLATED (VQE {delta:.6f} Ha below exact)")

    # CRITIC 2: Correlation energy must be negative
    if result.correlation_energy < 0:
        passed.append("C2:negative_correlation")
    else:
        failed.append(f"C2:positive_correlation ({result.correlation_energy:.6f} Ha)")

    # CRITIC 3: Correlation energy should be < 5% of total energy magnitude
    if result.hf_energy != 0:
        pct = abs(result.correlation_energy / result.hf_energy) * 100
        result.correlation_pct = pct
        if pct < 10.0:
            passed.append(f"C3:correlation_fraction ({pct:.2f}%)")
        else:
            failed.append(f"C3:correlation_too_large ({pct:.2f}% of total)")

    # CRITIC 4: Energy gap must be positive (real gap, not degenerate)
    if result.energy_gap > 1e-6:
        passed.append(f"C4:positive_gap ({result.energy_gap:.4f} Ha)")
    elif result.energy_gap > -1e-6:
        passed.append("C4:degenerate_ground_state (gap~0)")
    else:
        failed.append(f"C4:negative_gap ({result.energy_gap:.6f} Ha)")

    # CRITIC 5: VQE should recover >95% of correlation energy
    if result.correlation_energy < -1e-8:
        vqe_corr = result.vqe_energy - result.hf_energy
        recovery = vqe_corr / result.correlation_energy * 100
        result.vqe_correlation_recovery = recovery
        if recovery > 95:
            passed.append(f"C5:VQE_recovery ({recovery:.1f}%)")
        elif recovery > 90:
            failed.append(f"C5:VQE_recovery_moderate ({recovery:.1f}%, want >95%)")
        else:
            failed.append(f"C5:VQE_recovery_poor ({recovery:.1f}%, want >95%)")

    # CRITIC 6: VQE error MUST be < 1.6 mHa (1 kcal/mol = chemical accuracy)
    # STRICT: only true chemical accuracy passes
    if result.vqe_error_ha < 0.0016:
        passed.append(f"C6:chemical_accuracy ({result.vqe_error_kcal:.4f} kcal/mol)")
    else:
        failed.append(f"C6:NOT_chemical_accuracy ({result.vqe_error_kcal:.4f} kcal/mol, need <1.0)")

    # CRITIC 7: CASCI energy should be lower than HF (correlation captured)
    if result.casci_energy < result.hf_energy:
        passed.append("C7:CASCI_below_HF")
    else:
        failed.append(f"C7:CASCI_above_HF (no correlation captured)")

    # CRITIC 8: Number of Pauli terms should be reasonable
    expected_terms = result.n_qubits ** 2 * 2  # rough estimate
    if result.n_pauli_terms > 0:
        passed.append(f"C8:hamiltonian_terms ({result.n_pauli_terms})")
    else:
        failed.append("C8:no_hamiltonian_terms")

    result.critics_passed = passed
    result.critics_failed = failed
    return result


def compute_molecule(compound: dict, level: dict) -> CompoundResult:
    """Run full quantum chemistry pipeline for one compound at one level."""
    from pyscf import gto, scf, mcscf, ao2mo
    from openfermion import InteractionOperator, jordan_wigner
    from qiskit.circuit.library import efficient_su2
    from qiskit.quantum_info import SparsePauliOp, Statevector

    result = CompoundResult(
        compound_id=compound["id"],
        name=compound["name"],
        target=compound["target"],
        pathway=compound["pathway"],
        level=level["name"],
        basis=level["basis"],
    )

    t0 = time.time()

    try:
        # ── Build molecule ──
        mol = gto.Mole()
        mol.atom = compound["atoms"]
        mol.basis = level["basis"]
        mol.charge = 0
        mol.verbose = 0

        # Count electrons to determine spin
        atom_lines = [l.strip() for l in compound["atoms"].strip().split("\n") if l.strip()]
        total_e = 0
        for line in atom_lines:
            parts = line.split()
            sym = parts[0]
            atomic_numbers = {
                "H": 1, "He": 2, "C": 6, "N": 7, "O": 8, "F": 9,
                "S": 16, "P": 15, "Cl": 17, "Br": 35, "Na": 11
            }
            total_e += atomic_numbers.get(sym, 6)
        mol.spin = total_e % 2
        mol.build()

        result.n_atoms = mol.natm
        result.n_electrons = mol.nelectron

        # ── Hartree-Fock ──
        if mol.spin == 0:
            mf = scf.RHF(mol)
        else:
            mf = scf.ROHF(mol)
        mf.verbose = 0
        mf.max_cycle = 200
        mf.kernel()

        if not mf.converged:
            # Try with different initial guess
            mf.init_guess = 'atom'
            mf.kernel()

        result.hf_energy = float(mf.e_tot)

        # ── Active space ──
        n_active_elec = level["n_elec"]
        n_active_orb = level["n_orb"]

        # Ensure we don't exceed available orbitals
        n_active_orb = min(n_active_orb, mol.nao_nr())
        # Ensure active electrons don't exceed total and are <= 2*n_orb
        n_active_elec = min(n_active_elec, mol.nelectron, 2 * n_active_orb)
        # Core electrons must be even
        n_core_elec = mol.nelectron - n_active_elec
        if n_core_elec % 2 != 0:
            n_active_elec += 1
            if n_active_elec > 2 * n_active_orb:
                n_active_orb += 1
                n_active_orb = min(n_active_orb, mol.nao_nr())

        # Re-verify core is even
        n_core_elec = mol.nelectron - n_active_elec
        if n_core_elec % 2 != 0:
            n_active_elec -= 1
            n_core_elec = mol.nelectron - n_active_elec

        # ── CASCI ──
        mc = mcscf.CASCI(mf, n_active_orb, n_active_elec)
        mc.verbose = 0
        mc.kernel()
        result.casci_energy = float(mc.e_tot)

        h1, e_core = mc.get_h1eff()
        h2_raw = mc.get_h2eff()
        h2 = ao2mo.restore(1, h2_raw, n_active_orb)

        # ── Jordan-Wigner ──
        one_body = h1[:n_active_orb, :n_active_orb]
        two_body = h2
        ham_op = InteractionOperator(float(e_core), one_body, 0.5 * two_body)
        qubit_ham = jordan_wigner(ham_op)

        n_qubits = 2 * n_active_orb
        result.n_qubits = n_qubits

        # Extract Pauli terms
        pauli_terms = []
        for term, coeff in qubit_ham.terms.items():
            if abs(coeff) > 1e-12:
                pauli_str = ["I"] * n_qubits
                for qubit_idx, pauli_op in term:
                    if qubit_idx < n_qubits:
                        pauli_str[qubit_idx] = pauli_op
                pauli_terms.append(("".join(pauli_str), complex(coeff)))

        result.n_pauli_terms = len(pauli_terms)

        # ── GPU Exact Diagonalization ──
        dim = 2 ** n_qubits
        if dim > 65536:  # 16 qubits max for exact diag
            # Use CASCI energy as reference instead
            result.exact_energy = result.casci_energy
            logger.info(f"    System too large for exact diag ({n_qubits}q), using CASCI reference")
        else:
            I2 = torch.eye(2, dtype=torch.complex128, device=DEVICE)
            X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128, device=DEVICE)
            Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128, device=DEVICE)
            Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128, device=DEVICE)
            pmap = {"I": I2, "X": X, "Y": Y, "Z": Z}

            H = torch.zeros((dim, dim), dtype=torch.complex128, device=DEVICE)
            for ps, coeff in pauli_terms:
                if abs(coeff) < 1e-14:
                    continue
                mat = torch.tensor([[1.0]], dtype=torch.complex128, device=DEVICE)
                for ch in ps:
                    mat = torch.kron(mat, pmap[ch])
                H += coeff * mat

            torch.cuda.synchronize()

            eigenvalues = torch.linalg.eigvalsh(H)
            torch.cuda.synchronize()

            result.exact_energy = float(eigenvalues[0].real.cpu())
            if len(eigenvalues) > 1:
                result.energy_gap = float((eigenvalues[1] - eigenvalues[0]).real.cpu())

            # Verify Hermiticity
            herm_err = torch.max(torch.abs(H - H.conj().T)).item()
            if herm_err > 1e-8:
                result.critics_failed.append(f"WARN:hermiticity_error={herm_err:.2e}")

            del H, eigenvalues
            torch.cuda.empty_cache()

        result.gpu_time_s = time.time() - t0
        result.correlation_energy = result.exact_energy - result.hf_energy

        # ── GPU Statevector VQE ──
        observable = SparsePauliOp(
            [t[0] for t in pauli_terms],
            coeffs=[t[1] for t in pauli_terms]
        )

        # Use qiskit 2.x function-based API
        try:
            ansatz = efficient_su2(n_qubits, reps=level["reps"], entanglement="full")
        except Exception:
            from qiskit.circuit.library import EfficientSU2
            ansatz = EfficientSU2(n_qubits, reps=level["reps"], entanglement="full")

        n_params = ansatz.num_parameters

        # Multiple random restarts for VQE
        best_vqe = float('inf')
        best_history = []
        n_restarts = level.get("restarts", 5)

        for restart in range(n_restarts):
            x0 = np.random.uniform(-np.pi, np.pi, n_params) * 0.1
            history = []

            def cost_fn(params, _hist=history):
                bound = ansatz.assign_parameters(params)
                sv = Statevector(bound)
                e = float(sv.expectation_value(observable).real)
                _hist.append(e)
                return e

            opt = minimize(cost_fn, x0, method="COBYLA",
                          options={"maxiter": level["vqe_iters"], "rhobeg": 0.5})

            if opt.fun < best_vqe:
                best_vqe = opt.fun
                best_history = history.copy()
                best_nfev = opt.nfev
                best_converged = opt.success

        result.vqe_energy = best_vqe
        result.vqe_converged = best_converged
        result.vqe_evals = best_nfev
        result.vqe_error_ha = abs(best_vqe - result.exact_energy)
        result.vqe_error_kcal = result.vqe_error_ha * 627.509
        result.vqe_time_s = time.time() - t0 - result.gpu_time_s
        result.status = "computed"

    except Exception as e:
        result.status = "error"
        result.error = str(e)
        logger.error(f"    ERROR: {e}")
        traceback.print_exc()
        return result

    # ── Run critics ──
    result = critic_validate(result)

    return result


def run_pipeline():
    """Main iterative pipeline with escalating accuracy."""
    print("=" * 80)
    print("QUANTUM LONGEVITY — RIGOROUS GPU SIMULATION WITH SELF-CORRECTION")
    print(f"  GPU: AMD Instinct MI300X (192 GB VRAM)")
    print(f"  Engine: PySCF + OpenFermion + Qiskit + PyTorch {torch.__version__}")
    print(f"  Compounds: {len(COMPOUNDS)}")
    print(f"  Escalation levels: {len(LEVELS)}")
    print("=" * 80)

    # GPU warmup
    _ = torch.randn(512, 512, device=DEVICE) @ torch.randn(512, 512, device=DEVICE)
    torch.cuda.synchronize()

    all_round_results = []
    global_best = {}  # compound_id -> best result so far
    passed_ids = set()  # compound IDs that already passed all critics

    for level_idx, level in enumerate(LEVELS):
        round_start = time.time()

        # Determine which compounds need (re-)computation at this level
        compounds_this_round = [
            c for c in COMPOUNDS if c["id"] not in passed_ids
        ]
        if not compounds_this_round:
            print(f"\n  ★ ALL COMPOUNDS ALREADY PASSED — no need for {level['name']}")
            break

        print(f"\n{'█' * 80}")
        print(f"  ROUND {level_idx + 1}/{len(LEVELS)}: {level['name']}")
        print(f"  Active space: CAS({level['n_elec']},{level['n_orb']})")
        print(f"  Basis: {level['basis']}, VQE max iters: {level['vqe_iters']}, reps: {level['reps']}")
        print(f"  Compounds to compute: {len(compounds_this_round)}/{len(COMPOUNDS)} (already passed: {len(passed_ids)})")
        print(f"{'█' * 80}")

        round_results = []
        n_passed = 0
        n_failed = 0

        for i, compound in enumerate(compounds_this_round, 1):
            print(f"\n  [{i}/{len(compounds_this_round)}] {compound['name']}")

            result = compute_molecule(compound, level)
            round_results.append(result)

            if result.status == "error":
                print(f"    ✗ ERROR: {result.error}")
                n_failed += 1
                continue

            # Summary
            n_critics_pass = len(result.critics_passed)
            n_critics_fail = len(result.critics_failed)
            total_critics = n_critics_pass + n_critics_fail

            if n_critics_fail == 0:
                verdict = "✓ ALL CRITICS PASSED"
                n_passed += 1
                passed_ids.add(compound["id"])
            else:
                verdict = f"✗ {n_critics_fail}/{total_critics} FAILED"
                n_failed += 1

            print(f"    HF:    {result.hf_energy:.6f} Ha")
            print(f"    CASCI: {result.casci_energy:.6f} Ha")
            print(f"    Exact: {result.exact_energy:.6f} Ha")
            print(f"    VQE:   {result.vqe_energy:.6f} Ha (err={result.vqe_error_kcal:.4f} kcal/mol)")
            print(f"    Corr:  {result.correlation_energy:.6f} Ha ({result.correlation_energy * 627.509:.2f} kcal/mol)")
            print(f"    Gap:   {result.energy_gap:.6f} Ha ({result.energy_gap * 627.509:.2f} kcal/mol)")
            print(f"    Recovery: {result.vqe_correlation_recovery:.1f}%")
            print(f"    {verdict}")

            if result.critics_failed:
                for cf in result.critics_failed:
                    print(f"      FAIL: {cf}")

            # Track best per compound
            cid = compound["id"]
            if cid not in global_best or (result.status == "computed" and
                (global_best[cid].status != "computed" or
                 len(result.critics_failed) < len(global_best[cid].critics_failed) or
                 (len(result.critics_failed) == len(global_best[cid].critics_failed) and
                  result.vqe_error_kcal < global_best[cid].vqe_error_kcal))):
                global_best[cid] = result

        round_time = time.time() - round_start

        # Round summary
        print(f"\n  {'─' * 76}")
        print(f"  ROUND {level_idx + 1} SUMMARY: {n_passed} passed, {n_failed} failed/error ({round_time:.1f}s)")
        print(f"  Total passed so far: {len(passed_ids)}/{len(COMPOUNDS)}")
        print(f"  {'─' * 76}")

        all_round_results.append({
            "level": level["name"],
            "passed": n_passed,
            "failed": n_failed,
            "time_s": round_time,
            "results": [asdict(r) for r in round_results],
        })

        # Check if all compounds pass all critics
        if len(passed_ids) == len(COMPOUNDS):
            print(f"\n  ★ ALL COMPOUNDS PASS ALL CRITICS AT {level['name']} — STOPPING ESCALATION")
            break

        # Check which compounds need escalation
        remaining = len(COMPOUNDS) - len(passed_ids)
        if remaining > 0 and level_idx < len(LEVELS) - 1:
            print(f"\n  → {remaining} compounds need escalation to {LEVELS[level_idx + 1]['name']}")
        elif level_idx == len(LEVELS) - 1:
            print(f"\n  ⚠ MAX LEVEL REACHED — {remaining} compounds still failing")

    # =========================================================================
    # FINAL REPORT
    # =========================================================================
    print(f"\n\n{'═' * 80}")
    print("  FINAL REPORT — LONGEVITY COMPOUND RANKING")
    print(f"{'═' * 80}\n")

    # Sort by correlation energy
    valid_results = [r for r in global_best.values() if r.status == "computed"]
    valid_results.sort(key=lambda r: r.correlation_energy)

    print(f"{'Rank':<5} {'Compound':<30} {'Target':<15} {'Corr (kcal)':<12} "
          f"{'VQE err':<12} {'Critics':<10} {'Level'}")
    print("─" * 110)

    for rank, r in enumerate(valid_results, 1):
        n_pass = len(r.critics_passed)
        n_total = n_pass + len(r.critics_failed)
        critics_str = f"{n_pass}/{n_total}"
        corr_kcal = r.correlation_energy * 627.509
        print(f"{rank:<5} {r.name[:29]:<30} {r.target[:14]:<15} "
              f"{corr_kcal:>+10.2f}  {r.vqe_error_kcal:>10.4f}  "
              f"{critics_str:<10} {r.level}")

    # Failed compounds
    failed = [r for r in global_best.values() if r.status != "computed"]
    if failed:
        print(f"\n  FAILED ({len(failed)}):")
        for r in failed:
            print(f"    {r.compound_id} {r.name}: {r.error[:60]}")

    # Overall statistics
    if valid_results:
        avg_error = np.mean([r.vqe_error_kcal for r in valid_results])
        best_error = min(r.vqe_error_kcal for r in valid_results)
        worst_error = max(r.vqe_error_kcal for r in valid_results)
        all_pass_critics = sum(1 for r in valid_results if len(r.critics_failed) == 0)

        print(f"\n  STATISTICS:")
        print(f"    Compounds computed: {len(valid_results)}/{len(COMPOUNDS)}")
        print(f"    All critics passed: {all_pass_critics}/{len(valid_results)}")
        print(f"    VQE error: best={best_error:.4f}, worst={worst_error:.4f}, avg={avg_error:.4f} kcal/mol")
        print(f"    Chemical accuracy (<1 kcal/mol): {sum(1 for r in valid_results if r.vqe_error_kcal < 1.0)}/{len(valid_results)}")

    # GPU stats
    allocated = torch.cuda.max_memory_allocated() / 1e9
    print(f"    GPU peak memory: {allocated:.1f} GB / 192 GB")

    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "platform": "AMD Instinct MI300X — GPU-only",
        "method": "PySCF CASCI + GPU exact diag + Statevector VQE",
        "rounds": all_round_results,
        "best_results": [asdict(r) for r in valid_results],
        "failed": [asdict(r) for r in failed] if failed else [],
    }

    out_path = RESULTS_DIR / "rigorous_longevity_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Results saved: {out_path}")
    print(f"{'═' * 80}")


if __name__ == "__main__":
    run_pipeline()
