import requests, json
r = requests.post("http://localhost:5050/api/longevity/simulate", json={"compound_id":"LNG-001","target_name":"NAMPT"}, timeout=300)
print(json.dumps(r.json(), indent=2)[:3000])
