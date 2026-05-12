#!/usr/bin/env python3
"""Test IBM Quantum connection and list available backends."""
import os
import sys
sys.path.insert(0, "/opt/longevity-quantum/src")

# Load env
from pathlib import Path
env_file = Path("/opt/longevity-quantum/.env")
for line in env_file.read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        key, val = line.split("=", 1)
        os.environ[key.strip()] = val.strip()

from qiskit_ibm_runtime import QiskitRuntimeService

token = os.environ.get("IBM_QUANTUM_TOKEN", "")
print(f"Token found: {'yes' if token else 'NO'} (len={len(token)})")

try:
    svc = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    backends = svc.backends()
    print(f"Connected! {len(backends)} backends available:")
    for b in backends:
        status = b.status()
        print(f"  {b.name} ({b.num_qubits}q) - {'online' if status.operational else 'offline'} - pending: {status.pending_jobs}")
except Exception as e:
    print(f"ERROR: {e}")
