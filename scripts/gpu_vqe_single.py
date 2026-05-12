#!/usr/bin/env python3
"""
GPU-Accelerated Quantum VQE — Single Compound Test
====================================================
Runs on AMD MI300X: GPU builds Hamiltonian, IBM Quantum executes VQE.
Controlled test with 1 compound, 1 backend, limited iterations.
"""

import json
import time
import logging
import numpy as np
import torch
from scipy.optimize import minimize

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Config
IBM_TOKEN = "EPhwbJpVQ2V_XVyZ3__GkWz8yy6p4jokLpkhZCeBNI3Z"
TARGET_BACKEND = "ibm_marrakesh"  # best result last time (-75.011 Ha)
MAX_VQE_ITERS = 15  # limit to save quantum time
SHOTS = 4096


def gpu_build_hamiltonian():
    """Build water molecule Hamiltonian using PySCF on CPU + GPU matrix ops."""
    from pyscf import gto, scf, mcscf
    from openfermion import InteractionOperator, jordan_wigner
    
    logger.info("PHASE 1: Building Hamiltonian (PySCF + GPU)")
    
    # Water molecule — same as the VQE that ran today
    mol = gto.Mole()
    mol.atom = "O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0"
    mol.basis = "sto-3g"
    mol.build()
    
    # Hartree-Fock
    mf = scf.RHF(mol)
    mf.kernel()
    hf_energy = mf.e_tot
    logger.info(f"  HF energy: {hf_energy:.6f} Ha")
    
    # Active space: all orbitals (small molecule)
    n_orb = mol.nao_nr()  # 7 for water/sto-3g
    n_elec = mol.nelectron  # 10
    
    mc = mcscf.CASCI(mf, n_orb, n_elec)
    h1, e_core = mc.get_h1eff()
    h2_raw = mc.get_h2eff()
    
    # h2 from CASCI is in compressed (n_pair, n_pair) format — restore to 4D
    from pyscf import ao2mo
    h2 = ao2mo.restore(1, h2_raw, n_orb)  # restore to full 4-index (n,n,n,n)
    
    # Jordan-Wigner transform
    one_body = h1[:n_orb, :n_orb]
    two_body = h2
    hamiltonian_op = InteractionOperator(e_core, one_body, 0.5 * two_body)
    qubit_ham = jordan_wigner(hamiltonian_op)
    
    n_qubits = 2 * n_orb  # 14 qubits
    
    # Extract Pauli terms
    pauli_terms = []
    for term, coeff in qubit_ham.terms.items():
        if abs(coeff) > 1e-10:
            pauli_str = ["I"] * n_qubits
            for qubit_idx, pauli_op in term:
                if qubit_idx < n_qubits:
                    pauli_str[qubit_idx] = pauli_op
            pauli_terms.append(("".join(pauli_str), complex(coeff)))
    
    logger.info(f"  Qubits: {n_qubits}, Pauli terms: {len(pauli_terms)}")
    
    # GPU: exact diagonalization for ground truth
    logger.info("  GPU exact diagonalization...")
    device = "cuda"
    dim = 2 ** n_qubits
    
    I2 = torch.eye(2, dtype=torch.complex64, device=device)
    X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)
    pauli_map = {"I": I2, "X": X, "Y": Y, "Z": Z}
    
    t0 = time.time()
    H_gpu = torch.zeros((dim, dim), dtype=torch.complex64, device=device)
    
    for pauli_str, coeff in pauli_terms:
        if abs(coeff) < 1e-12:
            continue
        mat = torch.tensor([[1.0]], dtype=torch.complex64, device=device)
        for char in pauli_str:
            mat = torch.kron(mat, pauli_map[char])
        H_gpu += coeff * mat
    
    torch.cuda.synchronize()
    build_time = time.time() - t0
    logger.info(f"  GPU Hamiltonian matrix ({dim}x{dim}): {build_time:.2f}s")
    
    # Eigensolve on GPU
    t0 = time.time()
    eigenvalues = torch.linalg.eigvalsh(H_gpu)
    torch.cuda.synchronize()
    eig_time = time.time() - t0
    
    exact_energy = float(eigenvalues[0].real.cpu())
    logger.info(f"  GPU exact ground state: {exact_energy:.8f} Ha ({eig_time:.2f}s)")
    logger.info(f"  Correlation energy: {(exact_energy - hf_energy)*627.509:.2f} kcal/mol")
    
    # GPU: Pre-optimize initial parameters using statevector simulation
    logger.info("  GPU: Pre-optimizing VQE parameters (statevector)...")
    
    from qiskit.circuit.library import EfficientSU2
    from qiskit.quantum_info import SparsePauliOp, Statevector
    
    # Reduce to 10 qubits for hardware (fewer terms, faster VQE)
    n_hw_qubits = 10
    reduced_terms = [(p[:n_hw_qubits], c) for p, c in pauli_terms 
                     if all(p[i] == 'I' for i in range(n_hw_qubits, n_qubits))]
    reduced_terms = [(p, c) for p, c in reduced_terms if abs(c) > 1e-8]
    
    # Sort by magnitude, keep top 200
    reduced_terms.sort(key=lambda x: abs(x[1]), reverse=True)
    reduced_terms = reduced_terms[:200]
    
    observable = SparsePauliOp(
        [t[0] for t in reduced_terms],
        coeffs=[t[1] for t in reduced_terms]
    )
    
    ansatz = EfficientSU2(n_hw_qubits, reps=2, entanglement="linear")
    n_params = ansatz.num_parameters
    
    # Classical pre-optimization (fast, on GPU via statevector)
    best_params = np.random.uniform(-0.1, 0.1, n_params)
    best_energy = float('inf')
    
    def classical_cost(params):
        bound = ansatz.assign_parameters(params)
        sv = Statevector(bound)
        return float(sv.expectation_value(observable).real)
    
    t0 = time.time()
    pre_opt = minimize(classical_cost, best_params, method="COBYLA",
                      options={"maxiter": 100, "rhobeg": 0.5})
    pre_time = time.time() - t0
    
    logger.info(f"  Pre-optimized energy: {pre_opt.fun:.6f} Ha ({pre_time:.1f}s, {pre_opt.nfev} evals)")
    logger.info(f"  Starting params for hardware VQE ready")
    
    return {
        "n_qubits": n_hw_qubits,
        "observable": observable,
        "ansatz": ansatz,
        "initial_params": pre_opt.x,
        "exact_energy": exact_energy,
        "hf_energy": hf_energy,
        "pre_opt_energy": pre_opt.fun,
        "n_terms": len(reduced_terms),
    }


def run_ibm_vqe(hamiltonian_data: dict):
    """Submit VQE to real IBM Quantum hardware with GPU-optimized initial params."""
    from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    
    logger.info(f"\nPHASE 2: IBM Quantum VQE ({TARGET_BACKEND})")
    logger.info(f"  Max iterations: {MAX_VQE_ITERS}, Shots: {SHOTS}")
    
    svc = QiskitRuntimeService(channel="ibm_quantum_platform", token=IBM_TOKEN)
    backend = svc.backend(TARGET_BACKEND)
    
    logger.info(f"  Backend: {backend.name} ({backend.num_qubits} qubits)")
    
    ansatz = hamiltonian_data["ansatz"]
    observable = hamiltonian_data["observable"]
    x0 = hamiltonian_data["initial_params"]
    
    # Transpile for hardware
    pm = generate_preset_pass_manager(optimization_level=2, backend=backend)
    
    estimator = EstimatorV2(mode=backend)
    estimator.options.default_shots = SHOTS
    estimator.options.resilience_level = 1
    
    job_ids = []
    energy_history = []
    n_evals = [0]
    
    def cost_fn(params):
        bound = ansatz.assign_parameters(params)
        isa_circuit = pm.run(bound)
        isa_obs = observable.apply_layout(isa_circuit.layout)
        job = estimator.run([(isa_circuit, isa_obs)])
        job_ids.append(job.job_id())
        result = job.result()
        energy = float(result[0].data.evs)
        n_evals[0] += 1
        energy_history.append(energy)
        logger.info(f"    Iter {n_evals[0]}/{MAX_VQE_ITERS}: E = {energy:.6f} Ha (job: {job.job_id()[:8]}...)")
        return energy
    
    logger.info("  Starting COBYLA on real hardware...")
    t0 = time.time()
    
    opt_result = minimize(cost_fn, x0, method="COBYLA",
                         options={"maxiter": MAX_VQE_ITERS, "rhobeg": 0.3})
    
    wall_time = time.time() - t0
    
    return {
        "final_energy": float(opt_result.fun),
        "converged": opt_result.success,
        "n_iters": n_evals[0],
        "job_ids": job_ids,
        "energy_history": energy_history,
        "wall_time": wall_time,
        "backend": TARGET_BACKEND,
    }


def main():
    print("=" * 60)
    print("QUANTUM LONGEVITY — GPU+IBM HYBRID PIPELINE")
    print(f"  GPU: AMD Instinct MI300X (192 GB VRAM)")
    print(f"  QPU: {TARGET_BACKEND} (156 qubits)")
    print(f"  VQE: {MAX_VQE_ITERS} iterations, {SHOTS} shots")
    print("=" * 60)
    print()
    
    # Phase 1: GPU
    t_total = time.time()
    ham_data = gpu_build_hamiltonian()
    
    # Phase 2: IBM Quantum  
    vqe_result = run_ibm_vqe(ham_data)
    
    total_time = time.time() - t_total
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  HF energy:           {ham_data['hf_energy']:.6f} Ha")
    print(f"  GPU exact energy:    {ham_data['exact_energy']:.8f} Ha")
    print(f"  GPU pre-opt energy:  {ham_data['pre_opt_energy']:.6f} Ha")
    print(f"  IBM VQE energy:      {vqe_result['final_energy']:.6f} Ha")
    print(f"  VQE error vs exact:  {abs(vqe_result['final_energy'] - ham_data['exact_energy']):.6f} Ha")
    print(f"    = {abs(vqe_result['final_energy'] - ham_data['exact_energy']) * 627.509:.2f} kcal/mol")
    print(f"  Backend:             {vqe_result['backend']}")
    print(f"  IBM jobs submitted:  {vqe_result['n_iters']}")
    print(f"  Job IDs:             {vqe_result['job_ids'][:3]}...")
    print(f"  Converged:           {vqe_result['converged']}")
    print(f"  Wall time:           {total_time:.1f}s")
    print(f"  Quantum time (est):  ~{vqe_result['n_iters'] * 30}s")
    print()
    
    # Save
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "gpu": "AMD Instinct MI300X",
        "qpu": TARGET_BACKEND,
        "molecule": "H2O (water)",
        "basis": "sto-3g",
        "hf_energy": ham_data["hf_energy"],
        "exact_energy": ham_data["exact_energy"],
        "pre_opt_energy": ham_data["pre_opt_energy"],
        "vqe_energy": vqe_result["final_energy"],
        "error_ha": abs(vqe_result["final_energy"] - ham_data["exact_energy"]),
        "error_kcal": abs(vqe_result["final_energy"] - ham_data["exact_energy"]) * 627.509,
        "n_iters": vqe_result["n_iters"],
        "job_ids": vqe_result["job_ids"],
        "energy_history": vqe_result["energy_history"],
        "wall_time_s": total_time,
        "converged": vqe_result["converged"],
    }
    
    with open("/opt/research/results/gpu_quantum_result.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("Result saved to /opt/research/results/gpu_quantum_result.json")


if __name__ == "__main__":
    import os
    os.makedirs("/opt/research/results", exist_ok=True)
    main()
