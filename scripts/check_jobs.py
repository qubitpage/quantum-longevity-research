#!/usr/bin/env python3
"""Check IBM Quantum job history - what consumed 10 minutes?"""
from qiskit_ibm_runtime import QiskitRuntimeService

token = "EPhwbJpVQ2V_XVyZ3__GkWz8yy6p4jokLpkhZCeBNI3Z"
svc = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)

print("=" * 60)
print("IBM QUANTUM JOB HISTORY")
print("=" * 60)

jobs = svc.jobs(limit=20)
print(f"Total jobs found: {len(jobs)}")
print()

for j in jobs:
    print(f"Job: {j.job_id()}")
    print(f"  Backend: {j.backend().name}")
    print(f"  Status: {j.status()}")
    print(f"  Created: {j.creation_date}")
    try:
        metrics = j.metrics()
        if metrics:
            print(f"  Metrics: {metrics}")
    except:
        pass
    try:
        usage = j.usage()
        if usage:
            print(f"  Usage: {usage}")
    except:
        pass
    print()
