#!/usr/bin/env python3
"""Test the simulate endpoint locally."""
import urllib.request
import json

data = json.dumps({"compound_id": "LNG-001"}).encode()
req = urllib.request.Request(
    "http://127.0.0.1:5050/api/longevity/simulate",
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    resp = urllib.request.urlopen(req, timeout=120)
    print(f"Status: {resp.status}")
    print(resp.read().decode()[:2000])
except Exception as e:
    print(f"Error: {e}")
    if hasattr(e, "read"):
        print(e.read().decode()[:500])
