import json, sys
d = json.load(open("/opt/research/results/rigorous_longevity_results.json"))
for r in d["best_results"]:
    print(f"ID={r['compound_id']} | {r['name'][:30]} | HF={r['hf_energy']:.6f} | CASCI={r['casci_energy']:.6f} | Exact={r['exact_energy']:.6f} | VQE={r['vqe_energy']:.6f} | Corr={r['correlation_energy']:.6f} | Gap={r['energy_gap']:.6f} | ErrKcal={r['vqe_error_kcal']:.4f} | Qubits={r['n_qubits']} | Pauli={r['n_pauli_terms']} | Atoms={r['n_atoms']} | Elec={r['n_electrons']} | Recovery={r['vqe_correlation_recovery']:.1f} | Basis={r['basis']} | Level={r['level']}")
print(f"\nTimestamp: {d['timestamp']}")
print(f"Platform: {d['platform']}")
print(f"Method: {d['method']}")
print(f"Rounds: {len(d['rounds'])}")
