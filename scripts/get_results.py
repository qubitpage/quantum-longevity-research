#!/usr/bin/env python3
"""Retrieve actual results from today's VQE jobs on IBM Quantum."""
import os
from qiskit_ibm_runtime import QiskitRuntimeService

token = os.environ["IBM_QUANTUM_TOKEN"]
svc = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)

# Today's jobs (first 5 are from today's estimator runs)
jobs = svc.jobs(limit=5)

print("=" * 60)
print("TODAY'S VQE RESULTS (Real IBM Quantum Hardware)")
print("=" * 60)

for j in jobs:
    print(f"\nJob: {j.job_id()}")
    print(f"  Backend: {j.backend().name}")
    print(f"  Status: {j.status()}")
    print(f"  Created: {j.creation_date}")
    metrics = j.metrics()
    usage = metrics.get("usage", {})
    print(f"  Quantum time: {usage.get('quantum_seconds', 0)}s")
    
    if str(j.status()) == "DONE":
        try:
            result = j.result()
            # EstimatorV2 result
            for idx, pub_result in enumerate(result):
                evs = pub_result.data.evs
                if hasattr(pub_result.data, 'stds'):
                    stds = pub_result.data.stds
                    print(f"  Result[{idx}]: energy = {float(evs):.6f} +/- {float(stds):.6f} Ha")
                else:
                    print(f"  Result[{idx}]: energy = {float(evs):.6f} Ha")
        except Exception as e:
            print(f"  Error reading result: {e}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
total_secs = sum(j.metrics().get("usage", {}).get("quantum_seconds", 0) for j in jobs)
print(f"Total quantum seconds today: {total_secs}s ({total_secs/60:.1f} min)")
print(f"Backends used: {set(j.backend().name for j in jobs)}")
