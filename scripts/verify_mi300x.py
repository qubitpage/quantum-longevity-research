#!/usr/bin/env python3
"""Quick GPU + IBM Quantum verification on MI300X."""
import sys
import torch

print("=" * 50)
print("AMD MI300X GPU CHECK")
print("=" * 50)
print(f"PyTorch: {torch.__version__}")
print(f"HIP/ROCm: {torch.version.hip}")
print(f"GPU available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {p.name} ({p.total_memory/1e9:.1f} GB)")
    
    # Quick tensor op
    a = torch.randn(4096, 4096, device="cuda")
    b = torch.randn(4096, 4096, device="cuda")
    torch.cuda.synchronize()
    import time
    t0 = time.time()
    c = torch.mm(a, b)
    torch.cuda.synchronize()
    dt = time.time() - t0
    tflops = 2 * 4096**3 / dt / 1e12
    print(f"MatMul 4096x4096: {dt*1000:.1f}ms ({tflops:.1f} TFLOPS)")
else:
    print("NO GPU!")
    sys.exit(1)

print()
print("=" * 50)
print("IBM QUANTUM CHECK")
print("=" * 50)
from qiskit_ibm_runtime import QiskitRuntimeService
token = "EPhwbJpVQ2V_XVyZ3__GkWz8yy6p4jokLpkhZCeBNI3Z"
svc = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
backends = svc.backends()
print(f"Connected! {len(backends)} backends:")
for b in backends:
    s = b.status()
    print(f"  {b.name} ({b.num_qubits}q) - {'ONLINE' if s.operational else 'offline'}")
print("\nBOTH GPU + QUANTUM: OK")
