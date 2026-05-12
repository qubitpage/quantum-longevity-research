"""Test all API endpoints on the platform."""
import requests
import json
import sys

BASE = "http://localhost:5050/api/longevity"

def test_status():
    r = requests.get(f"{BASE}/status")
    print(f"[STATUS] {r.status_code}: {r.json()}")
    return r.status_code == 200

def test_simulate():
    payload = {"compound_id": "LNG-001", "target_name": "NAMPT"}
    r = requests.post(f"{BASE}/simulate", json=payload, timeout=180)
    print(f"[SIMULATE] {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"  Energy: {data.get('vqe_energy')}")
        print(f"  Binding: {data.get('binding_affinity')}")
        print(f"  Compound: {data.get('compound_name')}")
    else:
        print(f"  Error: {r.text[:500]}")
    return r.status_code == 200

def test_simulate_all():
    r = requests.post(f"{BASE}/simulate/all", json={}, timeout=300)
    print(f"[SIMULATE ALL] {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"  Results: {len(data.get('results', []))}")
        for res in data.get("results", [])[:3]:
            print(f"    {res.get('compound_id')}: E={res.get('vqe_energy')}")
    else:
        print(f"  Error: {r.text[:300]}")
    return r.status_code == 200

def test_orchestrate_report():
    r = requests.post(f"{BASE}/orchestrate/report", json={}, timeout=120)
    print(f"[ORCHESTRATE REPORT] {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"  Report length: {len(json.dumps(data))} chars")
        if "report" in data:
            print(f"  Report preview: {str(data['report'])[:200]}")
    else:
        print(f"  Error: {r.text[:300]}")
    return r.status_code == 200

def test_interactions():
    payload = {"compound_ids": ["LNG-001", "LNG-003", "LNG-005"]}
    r = requests.post(f"{BASE}/orchestrate/interactions", json=payload, timeout=120)
    print(f"[INTERACTIONS] {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"  Response: {json.dumps(data)[:300]}")
    else:
        print(f"  Error: {r.text[:300]}")
    return r.status_code == 200

def test_dosing():
    payload = {"compound_ids": ["LNG-001", "LNG-005", "LNG-012"]}
    r = requests.post(f"{BASE}/orchestrate/dosing", json=payload, timeout=120)
    print(f"[DOSING] {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"  Response: {json.dumps(data)[:300]}")
    else:
        print(f"  Error: {r.text[:300]}")
    return r.status_code == 200

if __name__ == "__main__":
    print("=" * 60)
    print("QUANTUM LONGEVITY PLATFORM — API TEST SUITE")
    print("=" * 60)
    
    results = {}
    
    # Basic status
    results["status"] = test_status()
    print()
    
    # Simulation (single compound)
    results["simulate"] = test_simulate()
    print()
    
    # Orchestration report
    results["report"] = test_orchestrate_report()
    print()
    
    # Interactions
    results["interactions"] = test_interactions()
    print()
    
    # Dosing
    results["dosing"] = test_dosing()
    print()
    
    # Summary
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'} — {name}")
    print(f"\n  {passed}/{total} passed")
    
    sys.exit(0 if passed == total else 1)
