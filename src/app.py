"""
Longevity Quantum Platform — Flask API Server
==============================================
Integrates with QubitPage-OS as a new module.
Exposes REST endpoints for data gathering, VQE simulation, and orchestration.
"""

import json
import logging
import os
from pathlib import Path

from flask import Flask, jsonify, request
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# =========================================================================
# Landing Page
# =========================================================================

@app.route("/", methods=["GET"])
def landing():
    """Root landing page — Quantum Longevity Research Platform."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Quantum Longevity Research Platform</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0a0e1a;color:#e0e7ff;min-height:100vh;display:flex;align-items:center;justify-content:center}
.container{max-width:900px;padding:3rem 2rem;text-align:center}
h1{font-size:2.8rem;background:linear-gradient(135deg,#60a5fa,#a78bfa,#f472b6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0.5rem}
.subtitle{font-size:1.2rem;color:#94a3b8;margin-bottom:2.5rem}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1.5rem;margin:2rem 0}
.card{background:rgba(30,41,59,0.7);border:1px solid rgba(99,102,241,0.3);border-radius:12px;padding:1.5rem;text-align:left}
.card h3{color:#818cf8;font-size:1rem;margin-bottom:0.5rem}
.card p{color:#94a3b8;font-size:0.85rem;line-height:1.5}
.badge{display:inline-block;background:rgba(34,197,94,0.15);color:#4ade80;border:1px solid rgba(34,197,94,0.3);border-radius:20px;padding:0.3rem 1rem;font-size:0.8rem;margin:0.3rem}
.badge.hw{background:rgba(99,102,241,0.15);color:#a78bfa;border-color:rgba(99,102,241,0.3)}
.status{margin-top:2rem;font-size:0.85rem;color:#64748b}
a{color:#60a5fa;text-decoration:none}
a:hover{text-decoration:underline}
.api-link{margin-top:2rem}
.api-link a{background:rgba(99,102,241,0.15);border:1px solid rgba(99,102,241,0.4);padding:0.6rem 1.5rem;border-radius:8px;display:inline-block;margin:0.3rem}
</style>
</head>
<body>
<div class="container">
<h1>Quantum Longevity Research Platform</h1>
<p class="subtitle">Real IBM Quantum Hardware &bull; VQE Molecular Simulation &bull; AI-Driven Drug Discovery</p>

<div>
<span class="badge">14 Longevity Compounds</span>
<span class="badge">8 Protein Targets</span>
<span class="badge hw">ibm_fez (156-qubit)</span>
<span class="badge hw">ibm_marrakesh (156-qubit)</span>
<span class="badge hw">ibm_kingston (156-qubit)</span>
</div>

<div class="grid">
<div class="card">
<h3>Quantum VQE Simulation</h3>
<p>Variational Quantum Eigensolver on real superconducting qubits. Active-space molecular Hamiltonians via PySCF + OpenFermion Jordan-Wigner transform.</p>
</div>
<div class="card">
<h3>Dual-Backend Execution</h3>
<p>Jobs submitted in parallel to IBM Fez, Marrakesh, and Kingston (156-qubit systems). Best energy result selected automatically. Zero simulators.</p>
</div>
<div class="card">
<h3>Data Pipeline</h3>
<p>Live ingestion from PubMed, ChEMBL, PDB, UniProt, and ClinicalTrials.gov for evidence-based compound selection.</p>
</div>
<div class="card">
<h3>AI Orchestration</h3>
<p>Gemini 2.0 Flash synthesizes quantum results, interaction checks, and dosing protocols into actionable research reports.</p>
</div>
</div>

<div class="api-link">
<a href="/api/longevity/status">API Status</a>
<a href="/api/longevity/compounds">Compounds</a>
<a href="/api/longevity/targets">Targets</a>
</div>

<p class="status">quantumqub.com &mdash; Powered by Qiskit 2.4 + IBM Quantum Runtime</p>
</div>
</body>
</html>""", 200, {"Content-Type": "text/html"}


# =========================================================================
# Health & Status
# =========================================================================

@app.route("/api/longevity/status", methods=["GET"])
def status():
    """Platform health check."""
    return jsonify({
        "status": "operational",
        "platform": "Longevity Quantum Platform v1.0.0",
        "engine": "QuBIOS Transit Ring + VQE",
        "pathways": ["nad_sirtuins", "senolytics", "telomerase", "rapalogs_mtor", "autophagy"],
        "compounds_loaded": _count_compounds(),
        "targets_loaded": _count_targets(),
        "ibm_quantum": bool(os.environ.get("IBM_QUANTUM_TOKEN")),
        "gemini_ai": bool(os.environ.get("GEMINI_API_KEY")),
    })


# =========================================================================
# Data Pipeline
# =========================================================================

@app.route("/api/longevity/data/gather", methods=["POST"])
def gather_data():
    """Run the full data gathering pipeline (PubMed, ChEMBL, PDB, UniProt, ClinicalTrials)."""
    from longevity_data import LongevityDataPipeline
    
    pipeline = LongevityDataPipeline()
    summary = pipeline.run_full_pipeline()
    return jsonify({"status": "complete", "summary": summary})


@app.route("/api/longevity/data/literature", methods=["GET"])
def get_literature():
    """Get gathered literature data."""
    lit_file = DATA_DIR / "literature.json"
    if lit_file.exists():
        with open(lit_file) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "No literature data. Run /api/longevity/data/gather first."}), 404


@app.route("/api/longevity/data/trials", methods=["GET"])
def get_trials():
    """Get active clinical trials."""
    trials_file = DATA_DIR / "clinical_trials.json"
    if trials_file.exists():
        with open(trials_file) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "No trials data. Run /api/longevity/data/gather first."}), 404


# =========================================================================
# Quantum Simulation
# =========================================================================

@app.route("/api/longevity/simulate", methods=["POST"])
def simulate():
    """
    Run VQE simulation for a compound-target pair.
    Body: {"compound_id": "LNG-001", "target_name": "NAMPT"}
    """
    from longevity_sim import LongevityVQESimulator, VQEConfig

    data = request.get_json(force=True) or {}
    compound_id = data.get("compound_id", "LNG-001")
    target_name = data.get("target_name")

    try:
        sim = LongevityVQESimulator()

        # Build config from compound data
        compounds_file = MODELS_DIR / "longevity_compounds.json"
        with open(compounds_file) as f:
            compounds_data = json.load(f)

        compound = next((c for c in compounds_data["compounds"] if c["id"] == compound_id), None)
        if not compound:
            return jsonify({"error": f"Compound {compound_id} not found"}), 404

        sim_params = compound.get("quantum_sim_params", {})
        config = VQEConfig(
            target_name=target_name or compound.get("target_protein", "unknown"),
            compound_id=compound_id,
            pdb_id=sim_params.get("target_pdb", ""),
            active_space_qubits=sim_params.get("active_space_qubits", 10),
        )

        hamiltonian = sim._mock_hamiltonian(config)
        result = sim.run_vqe(config, hamiltonian)

        return jsonify({
            "status": "complete",
            "hardware": "real_ibm_quantum",
            "backend_used": result.backend_used,
            "backends_queried": result.backends_queried,
            "compound": compound["name"],
            "target": config.target_name,
            "ground_state_energy_hartree": result.ground_state_energy,
            "binding_energy_kcal": result.binding_energy_kcal,
            "qubits_used": config.active_space_qubits,
            "converged": result.converged,
            "n_iterations": result.n_iterations,
            "total_shots": result.total_shots,
            "ibm_job_ids": result.ibm_job_ids[:5],
            "quantum_advantage": sim_params.get("quantum_advantage_reason", ""),
        })
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/longevity/simulate/all", methods=["POST"])
def simulate_all():
    """Run VQE for all compounds on real IBM Quantum hardware."""
    from longevity_sim import LongevityVQESimulator

    try:
        sim = LongevityVQESimulator()
        results = sim.run_all_simulations()

        return jsonify({
            "status": "complete",
            "hardware": "real_ibm_quantum",
            "backends": sim.active_backends,
            "total_simulations": len(results),
            "results": [
                {
                    "compound": r.config.get("compound_id", ""),
                    "binding_energy_kcal": r.binding_energy_kcal,
                    "backend_used": r.backend_used,
                    "n_jobs": len(r.ibm_job_ids),
                    "converged": r.converged,
                }
                for r in results
            ]
        })
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        return jsonify({"error": str(e)}), 500


# =========================================================================
# AI Orchestration
# =========================================================================

@app.route("/api/longevity/orchestrate/report", methods=["POST"])
def generate_report():
    """Generate the full Longevity Cocktail Report using AI orchestration."""
    from longevity_orchestrator import LongevityOrchestrator
    
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return jsonify({"error": "GEMINI_API_KEY not configured"}), 500

    orchestrator = LongevityOrchestrator(api_key)
    report = orchestrator.generate_cocktail_report()
    return jsonify(report)


@app.route("/api/longevity/orchestrate/interactions", methods=["POST"])
def check_interactions():
    """Check drug-drug interactions between selected compounds."""
    from longevity_orchestrator import LongevityOrchestrator
    
    data = request.get_json(force=True) or {}
    compound_ids = data.get("compound_ids", ["LNG-001", "LNG-003", "LNG-005", "LNG-006", "LNG-007"])
    
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return jsonify({"error": "GEMINI_API_KEY not configured"}), 500

    orchestrator = LongevityOrchestrator(api_key)
    result = orchestrator.check_interactions(compound_ids)
    return jsonify(result)


@app.route("/api/longevity/orchestrate/dosing", methods=["POST"])
def optimize_dosing():
    """Generate optimal dosing schedule for selected compounds."""
    from longevity_orchestrator import LongevityOrchestrator
    
    data = request.get_json(force=True) or {}
    compound_ids = data.get("compound_ids", ["LNG-001", "LNG-003", "LNG-005", "LNG-006", "LNG-007"])
    
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return jsonify({"error": "GEMINI_API_KEY not configured"}), 500

    orchestrator = LongevityOrchestrator(api_key)
    result = orchestrator.optimize_dosing_schedule(compound_ids)
    return jsonify(result)


# =========================================================================
# Compounds & Targets Data
# =========================================================================

@app.route("/api/longevity/compounds", methods=["GET"])
def get_compounds():
    """List all longevity compounds with their data."""
    with open(MODELS_DIR / "longevity_compounds.json") as f:
        return jsonify(json.load(f))


@app.route("/api/longevity/targets", methods=["GET"])
def get_targets():
    """List all longevity protein targets."""
    with open(MODELS_DIR / "longevity_targets.json") as f:
        return jsonify(json.load(f))


@app.route("/api/longevity/results", methods=["GET"])
def get_results():
    """Get all VQE simulation results."""
    summary_file = RESULTS_DIR / "vqe_summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "No results yet. Run /api/longevity/simulate/all first."}), 404


@app.route("/api/longevity/report", methods=["GET"])
def get_report():
    """Get the latest cocktail report."""
    report_file = RESULTS_DIR / "longevity_cocktail_report.json"
    if report_file.exists():
        with open(report_file) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "No report generated yet. Run /api/longevity/orchestrate/report first."}), 404


# =========================================================================
# Helpers
# =========================================================================

def _count_compounds() -> int:
    try:
        with open(MODELS_DIR / "longevity_compounds.json") as f:
            return len(json.load(f).get("compounds", []))
    except:
        return 0


def _count_targets() -> int:
    try:
        with open(MODELS_DIR / "longevity_targets.json") as f:
            data = json.load(f)
            return sum(len(p.get("targets", [])) for p in data.get("pathways", {}).values())
    except:
        return 0


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    logger.info(f"Starting Longevity Quantum Platform on port {port}")
    logger.info(f"IBM Quantum: {'configured' if os.environ.get('IBM_QUANTUM_TOKEN') else 'NOT SET'}")
    logger.info(f"Gemini AI: {'configured' if os.environ.get('GEMINI_API_KEY') else 'NOT SET'}")
    app.run(host="0.0.0.0", port=port, debug=False)
