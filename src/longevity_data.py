"""
Longevity Quantum Platform — Data Gathering Pipeline
=====================================================
Fetches validated scientific data from:
- PubMed (NCBI E-utilities) — peer-reviewed papers
- ChEMBL (REST API) — bioactivity data (IC50, Ki)
- RCSB PDB (REST API) — 3D crystal structures
- UniProt (REST API) — protein sequences/annotations
- ClinicalTrials.gov (API v2) — active aging trials

All sources are open-access. No API keys required for PubMed/ChEMBL/PDB/UniProt.
"""

import json
import os
import time
import logging
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Rate limit: NCBI requests max 3/sec without API key, 10/sec with
NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
PDB_BASE = "https://data.rcsb.org/rest/v1/core"
UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb"
CLINICALTRIALS_BASE = "https://clinicaltrials.gov/api/v2"


class LongevityDataPipeline:
    """Gathers and structures longevity research data from validated scientific sources."""

    def __init__(self, ncbi_api_key: Optional[str] = None):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "QubitPage-LongevityPlatform/1.0 (contact@qubitpage.com)"})
        self.ncbi_api_key = ncbi_api_key or os.environ.get("NCBI_API_KEY", "")
        self.rate_delay = 0.34 if not self.ncbi_api_key else 0.1  # 3/sec vs 10/sec

    def _get(self, url: str, params: Optional[dict] = None) -> Optional[dict]:
        """Safe HTTP GET with retry and rate limiting."""
        time.sleep(self.rate_delay)
        for attempt in range(3):
            try:
                resp = self.session.get(url, params=params, timeout=30)
                if resp.status_code == 200:
                    return resp.json() if "json" in resp.headers.get("content-type", "") else {"text": resp.text}
                elif resp.status_code == 429:
                    wait = 2 ** (attempt + 1)
                    logger.warning(f"Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    logger.error(f"HTTP {resp.status_code} for {url}")
                    return None
            except requests.RequestException as e:
                logger.error(f"Request failed (attempt {attempt+1}): {e}")
                time.sleep(2)
        return None

    # =========================================================================
    # PubMed — Papers on longevity compounds/pathways
    # =========================================================================

    def search_pubmed(self, query: str, max_results: int = 20) -> list[dict]:
        """Search PubMed for peer-reviewed papers. Returns PMIDs + titles."""
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "sort": "relevance",
            "retmode": "json",
        }
        if self.ncbi_api_key:
            params["api_key"] = self.ncbi_api_key

        data = self._get(f"{NCBI_BASE}/esearch.fcgi", params)
        if not data or "esearchresult" not in data:
            return []

        pmids = data["esearchresult"].get("idlist", [])
        if not pmids:
            return []

        # Fetch summaries for these PMIDs
        summary_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "json",
        }
        if self.ncbi_api_key:
            summary_params["api_key"] = self.ncbi_api_key

        summaries = self._get(f"{NCBI_BASE}/esummary.fcgi", summary_params)
        if not summaries or "result" not in summaries:
            return []

        papers = []
        for pmid in pmids:
            info = summaries["result"].get(pmid, {})
            if isinstance(info, dict) and "title" in info:
                papers.append({
                    "pmid": pmid,
                    "title": info.get("title", ""),
                    "authors": [a.get("name", "") for a in info.get("authors", [])[:3]],
                    "journal": info.get("fulljournalname", ""),
                    "year": info.get("pubdate", "")[:4],
                    "doi": next((eid["value"] for eid in info.get("articleids", []) if eid.get("idtype") == "doi"), ""),
                })
        return papers

    def gather_pathway_literature(self) -> dict:
        """Fetch top papers for each longevity pathway."""
        queries = {
            "nad_sirtuins": '("NAD+" OR "nicotinamide mononucleotide" OR "sirtuin") AND ("aging" OR "longevity") AND ("clinical trial" OR "human")',
            "senolytics": '("senolytic" OR "senescent cells" OR "dasatinib quercetin") AND ("aging" OR "longevity") AND ("clinical trial" OR "human")',
            "telomerase": '("telomerase activator" OR "telomere lengthening" OR "hTERT") AND ("aging" OR "longevity") AND ("clinical" OR "human")',
            "rapalogs_mtor": '("rapamycin" OR "mTOR inhibitor" OR "everolimus") AND ("aging" OR "longevity" OR "healthspan") AND ("clinical trial" OR "human")',
            "autophagy": '("spermidine" OR "autophagy inducer") AND ("aging" OR "longevity") AND ("clinical trial" OR "human")',
        }

        results = {}
        for pathway, query in queries.items():
            logger.info(f"Searching PubMed for pathway: {pathway}")
            papers = self.search_pubmed(query, max_results=15)
            results[pathway] = papers
            logger.info(f"  Found {len(papers)} papers")

        return results

    # =========================================================================
    # ChEMBL — Bioactivity data (IC50, Ki values)
    # =========================================================================

    def get_chembl_bioactivity(self, target_chembl_id: str, max_results: int = 50) -> list[dict]:
        """Fetch bioactivity data (IC50, Ki) for a target from ChEMBL."""
        params = {
            "target_chembl_id": target_chembl_id,
            "type__in": "IC50,Ki,Kd",
            "limit": max_results,
            "format": "json",
        }
        data = self._get(f"{CHEMBL_BASE}/activity.json", params)
        if not data or "activities" not in data:
            return []

        activities = []
        for act in data["activities"]:
            activities.append({
                "molecule_name": act.get("molecule_pref_name", ""),
                "molecule_chembl_id": act.get("molecule_chembl_id", ""),
                "type": act.get("type", ""),
                "value": act.get("value", ""),
                "units": act.get("units", ""),
                "assay_description": act.get("assay_description", ""),
            })
        return activities

    def search_chembl_target(self, uniprot_id: str) -> Optional[str]:
        """Find ChEMBL target ID from UniProt accession."""
        data = self._get(f"{CHEMBL_BASE}/target.json", {"target_components__accession": uniprot_id, "format": "json"})
        if data and "targets" in data and data["targets"]:
            return data["targets"][0].get("target_chembl_id")
        return None

    def gather_bioactivity_data(self) -> dict:
        """Fetch bioactivity for all longevity targets."""
        # Key targets with their UniProt IDs
        targets = {
            "NAMPT": "P43490",
            "SIRT1": "Q96EB6",
            "BCL-xL": "Q07817",
            "mTOR": "P42345",
            "AMPK_alpha": "Q13131",
            "hTERT": "O14746",
        }

        results = {}
        for name, uniprot in targets.items():
            logger.info(f"Looking up ChEMBL target for {name} ({uniprot})...")
            chembl_id = self.search_chembl_target(uniprot)
            if chembl_id:
                logger.info(f"  Found {chembl_id}, fetching bioactivity...")
                activities = self.get_chembl_bioactivity(chembl_id)
                results[name] = {"chembl_id": chembl_id, "activities": activities}
                logger.info(f"  Got {len(activities)} bioactivity records")
            else:
                logger.warning(f"  No ChEMBL target found for {name}")
                results[name] = {"chembl_id": None, "activities": []}

        return results

    # =========================================================================
    # RCSB PDB — 3D Crystal Structures
    # =========================================================================

    def get_pdb_info(self, pdb_id: str) -> Optional[dict]:
        """Fetch PDB entry metadata (resolution, method, ligands)."""
        data = self._get(f"{PDB_BASE}/entry/{pdb_id}")
        if not data:
            return None

        return {
            "pdb_id": pdb_id,
            "title": data.get("struct", {}).get("title", ""),
            "method": data.get("exptl", [{}])[0].get("method", "") if data.get("exptl") else "",
            "resolution": data.get("rcsb_entry_info", {}).get("resolution_combined", [None])[0],
            "polymer_count": data.get("rcsb_entry_info", {}).get("polymer_entity_count", 0),
            "release_date": data.get("rcsb_accession_info", {}).get("initial_release_date", ""),
        }

    def download_pdb_structure(self, pdb_id: str) -> Optional[Path]:
        """Download PDB coordinate file for molecular simulation."""
        outpath = DATA_DIR / f"{pdb_id}.pdb"
        if outpath.exists():
            logger.info(f"  PDB {pdb_id} already downloaded")
            return outpath

        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        time.sleep(self.rate_delay)
        try:
            resp = self.session.get(url, timeout=30)
            if resp.status_code == 200:
                outpath.write_text(resp.text)
                logger.info(f"  Downloaded {pdb_id}.pdb ({len(resp.text)} bytes)")
                return outpath
            else:
                logger.error(f"  Failed to download PDB {pdb_id}: HTTP {resp.status_code}")
                return None
        except requests.RequestException as e:
            logger.error(f"  PDB download error: {e}")
            return None

    def gather_structures(self) -> dict:
        """Download all target protein structures."""
        # Load targets from JSON
        targets_file = MODELS_DIR / "longevity_targets.json"
        with open(targets_file) as f:
            targets_data = json.load(f)

        results = {}
        for pathway_key, pathway in targets_data["pathways"].items():
            for target in pathway["targets"]:
                pdb_id = target["pdb_id"]
                logger.info(f"Fetching PDB {pdb_id} ({target['name']})...")
                info = self.get_pdb_info(pdb_id)
                filepath = self.download_pdb_structure(pdb_id)
                results[pdb_id] = {
                    "target_name": target["name"],
                    "pathway": pathway_key,
                    "info": info,
                    "local_file": str(filepath) if filepath else None,
                }

        return results

    # =========================================================================
    # UniProt — Protein annotations
    # =========================================================================

    def get_uniprot_entry(self, accession: str) -> Optional[dict]:
        """Fetch UniProt protein entry (sequence, function, GO terms)."""
        data = self._get(f"{UNIPROT_BASE}/{accession}.json")
        if not data:
            return None

        # Extract key fields
        function_text = ""
        for comment in data.get("comments", []):
            if comment.get("commentType") == "FUNCTION":
                texts = comment.get("texts", [])
                if texts:
                    function_text = texts[0].get("value", "")
                    break

        return {
            "accession": accession,
            "protein_name": data.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", ""),
            "gene_name": data.get("genes", [{}])[0].get("geneName", {}).get("value", "") if data.get("genes") else "",
            "organism": data.get("organism", {}).get("scientificName", ""),
            "function": function_text,
            "sequence_length": data.get("sequence", {}).get("length", 0),
            "go_terms": [
                next((p.get("value", "") for p in ref.get("properties", []) if p.get("key") == "GoTerm"), ref.get("id", ""))
                for ref in data.get("uniProtKBCrossReferences", [])
                if ref.get("database") == "GO"
            ][:10],
        }

    def gather_protein_annotations(self) -> dict:
        """Fetch UniProt annotations for all target proteins."""
        targets_file = MODELS_DIR / "longevity_targets.json"
        with open(targets_file) as f:
            targets_data = json.load(f)

        results = {}
        for pathway_key, pathway in targets_data["pathways"].items():
            for target in pathway["targets"]:
                uniprot_id = target["uniprot_id"]
                logger.info(f"Fetching UniProt {uniprot_id} ({target['name']})...")
                entry = self.get_uniprot_entry(uniprot_id)
                if entry:
                    results[uniprot_id] = entry
                    logger.info(f"  {entry['protein_name']} — {entry['sequence_length']} aa")

        return results

    # =========================================================================
    # ClinicalTrials.gov — Active longevity trials
    # =========================================================================

    def search_clinical_trials(self, query: str, max_results: int = 20) -> list[dict]:
        """Search ClinicalTrials.gov for active aging/longevity trials."""
        params = {
            "query.term": query,
            "filter.overallStatus": "RECRUITING,ACTIVE_NOT_RECRUITING,ENROLLING_BY_INVITATION",
            "pageSize": max_results,
            "format": "json",
        }
        data = self._get(f"{CLINICALTRIALS_BASE}/studies", params)
        if not data or "studies" not in data:
            return []

        trials = []
        for study in data["studies"]:
            protocol = study.get("protocolSection", {})
            id_module = protocol.get("identificationModule", {})
            status_module = protocol.get("statusModule", {})
            design_module = protocol.get("designModule", {})

            trials.append({
                "nct_id": id_module.get("nctId", ""),
                "title": id_module.get("briefTitle", ""),
                "status": status_module.get("overallStatus", ""),
                "phase": design_module.get("phases", ["Unknown"]),
                "start_date": status_module.get("startDateStruct", {}).get("date", ""),
            })
        return trials

    def gather_longevity_trials(self) -> dict:
        """Find all active clinical trials related to longevity compounds."""
        queries = {
            "nad_boosters": "nicotinamide mononucleotide OR nicotinamide riboside aging",
            "senolytics": "senolytic OR dasatinib quercetin aging",
            "telomerase": "telomerase activator OR telomere aging",
            "rapamycin_aging": "rapamycin aging OR sirolimus longevity",
            "metformin_aging": "metformin aging TAME",
            "spermidine": "spermidine aging OR spermidine longevity",
        }

        results = {}
        for category, query in queries.items():
            logger.info(f"Searching clinical trials: {category}...")
            trials = self.search_clinical_trials(query)
            results[category] = trials
            logger.info(f"  Found {len(trials)} active trials")

        return results

    # =========================================================================
    # Master pipeline: gather all data
    # =========================================================================

    def run_full_pipeline(self) -> dict:
        """Execute complete data gathering pipeline. Returns summary."""
        logger.info("=" * 60)
        logger.info("LONGEVITY DATA PIPELINE — Starting full data gathering")
        logger.info("=" * 60)

        summary = {}

        # 1. Literature
        logger.info("\n[1/5] Gathering PubMed literature...")
        literature = self.gather_pathway_literature()
        self._save_json("literature.json", literature)
        summary["literature"] = {k: len(v) for k, v in literature.items()}

        # 2. Bioactivity
        logger.info("\n[2/5] Gathering ChEMBL bioactivity data...")
        bioactivity = self.gather_bioactivity_data()
        self._save_json("bioactivity.json", bioactivity)
        summary["bioactivity"] = {k: len(v.get("activities", [])) for k, v in bioactivity.items()}

        # 3. Crystal structures
        logger.info("\n[3/5] Downloading PDB crystal structures...")
        structures = self.gather_structures()
        self._save_json("structures.json", structures)
        summary["structures"] = {k: bool(v.get("local_file")) for k, v in structures.items()}

        # 4. Protein annotations
        logger.info("\n[4/5] Fetching UniProt protein annotations...")
        proteins = self.gather_protein_annotations()
        self._save_json("protein_annotations.json", proteins)
        summary["proteins"] = len(proteins)

        # 5. Clinical trials
        logger.info("\n[5/5] Searching active clinical trials...")
        trials = self.gather_longevity_trials()
        self._save_json("clinical_trials.json", trials)
        summary["trials"] = {k: len(v) for k, v in trials.items()}

        # Save summary
        self._save_json("pipeline_summary.json", summary)

        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info(f"Data saved to: {DATA_DIR}")
        logger.info("=" * 60)

        return summary

    def _save_json(self, filename: str, data: dict):
        """Save data to JSON file."""
        filepath = DATA_DIR / filename
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"  Saved: {filepath}")


if __name__ == "__main__":
    pipeline = LongevityDataPipeline()
    summary = pipeline.run_full_pipeline()
    print("\n=== Pipeline Summary ===")
    print(json.dumps(summary, indent=2))
