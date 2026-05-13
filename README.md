<p align="center">
  <img src="docs/assets/molecule-header.svg" alt="Quantum Longevity" width="120"/>
</p>

<h1 align="center">Quantum-Chemical Characterization of Ten Longevity Compounds</h1>

<p align="center">
  <em>A GPU-Accelerated Multi-Reference Study on AMD MI300X — With Honest Self-Assessment</em>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Method-CASCI%20%2B%20Exact%20Diag-blue?style=for-the-badge" alt="CASCI"/></a>
  <a href="#"><img src="https://img.shields.io/badge/GPU-AMD%20MI300X%20206GB-red?style=for-the-badge&logo=amd" alt="AMD MI300X"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Compounds-10%20Studied-informational?style=for-the-badge" alt="10 Compounds"/></a>
  <a href="#"><img src="https://img.shields.io/badge/CASCI%20%E2%89%A1%20Exact-Validated-brightgreen?style=for-the-badge" alt="CASCI Validated"/></a>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/PySCF-2.13.0-blue?style=flat-square" alt="PySCF"/></a>
  <a href="#"><img src="https://img.shields.io/badge/OpenFermion-1.7.1-orange?style=flat-square" alt="OpenFermion"/></a>
  <a href="#"><img src="https://img.shields.io/badge/PyTorch-2.5.1%2BROCm6.2-ee4c2c?style=flat-square&logo=pytorch" alt="PyTorch"/></a>
  <a href="#"><img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License"/></a>
</p>

---

**Authors:** Qubit OS Research Laboratory  
**Date:** May 13, 2026  
**Platform:** AMD Instinct MI300X (206 GB HBM3) — ROCm 6.2  
**Software:** PySCF 2.13.0, OpenFermion 1.7.1, PyTorch 2.5.1+ROCm 6.2  
**Pipeline runtime:** 19,983 seconds (~5.5 hours) for full 10-compound study

---

## Table of Contents

- [What This Project Is](#what-this-project-is)
- [What We Actually Computed](#what-we-actually-computed)
- [Results](#results)
- [The Bug We Found and Fixed](#the-bug-we-found-and-fixed)
- [What This Does NOT Tell You](#what-this-does-not-tell-you)
- [Honest Self-Criticism](#honest-self-criticism)
- [Methods](#methods)
- [The Ten Compounds](#the-ten-compounds)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [References](#references)
- [License](#license)

---

## What This Project Is

This repository contains the code and results of a computational quantum chemistry study of ten compounds with published evidence for lifespan extension in model organisms. We computed their electronic ground states using **Complete Active Space Configuration Interaction (CASCI)** with GPU-accelerated exact diagonalization on an AMD MI300X accelerator.

**What we set out to do:**
- Compute multi-reference electronic structure for 10 longevity compounds
- Validate the Jordan-Wigner qubit Hamiltonian against PySCF's CASCI (they must agree exactly)
- Test whether a variational quantum eigensolver (VQE) ansatz can recover the correlation energy
- Compute solvation shifts using ddCOSMO implicit solvent model
- Report everything honestly, including failures

**What we did NOT set out to do:**
- Discover new drugs or predict biological activity
- Replace or replicate clinical trials
- Claim quantum advantage

---

## What We Actually Computed

For each compound, we ran a multi-stage pipeline:

1. **Geometry optimization** at HF/6-31G* level (PySCF)
2. **CASCI** with escalating active spaces: CAS(4,4) → CAS(6,6) → CAS(7,6) at 6-31G* and cc-pVDZ basis sets
3. **Jordan-Wigner mapping** to qubit Hamiltonian (OpenFermion)
4. **GPU exact diagonalization** of the full qubit Hamiltonian matrix (PyTorch on MI300X)
5. **VQE simulation** using EfficientSU2 ansatz with LBFGS optimizer (10 restarts)
6. **Solvation** at the best gas-phase level using ddCOSMO (water, ε=78.39)
7. **11-critic validation engine** checking physical/chemical consistency

The central validation: **CASCI energy from PySCF must exactly equal the ground state from our GPU exact diagonalization of the Jordan-Wigner Hamiltonian.** If they disagree, something is wrong with our integral mapping. This is not a discovery — it's a correctness check.

---

## Results

### Core Result: CASCI ≡ Exact Diagonalization (Δ = 0.0000 kcal/mol)

For all 10 compounds, the PySCF CASCI energy matches our GPU exact diagonalization of the Jordan-Wigner qubit Hamiltonian to machine precision:

| ID | Compound | Level | E(CASCI) [Ha] | E(Exact Diag) [Ha] | Δ [kcal/mol] |
|---|---|---|---:|---:|---:|
| LNG-001 | NMN (nicotinamide ring) | CAS(6,6)/cc-pVDZ | -414.537629 | -414.537629 | **0.0000** |
| LNG-002 | Resveratrol | CAS(6,6)/cc-pVDZ | -382.501778 | -382.501778 | **0.0000** |
| LNG-003 | Rapamycin (piperidone) | CAS(5,4)/6-31G* | -322.113890 | -322.113890 | **0.0000** |
| LNG-004 | Metformin | CAS(6,6)/cc-pVDZ | -430.129723 | -430.129723 | **0.0000** |
| LNG-005 | Quercetin (chromone) | CAS(7,6)/cc-pVDZ | -455.476673 | -455.476673 | **0.0000** |
| LNG-006 | Fisetin (flavone) | CAS(6,6)/cc-pVDZ | -418.352261 | -418.352261 | **0.0000** |
| LNG-007 | Dasatinib (aminopyrimidine) | CAS(6,6)/cc-pVDZ | -317.790530 | -317.790530 | **0.0000** |
| LNG-008 | Spermidine | CAS(7,6)/cc-pVDZ | -438.790252 | -438.790252 | **0.0000** |
| LNG-009 | Urolithin A | CAS(7,6)/cc-pVDZ | -645.143551 | -645.143551 | **0.0000** |
| LNG-010 | Alpha-Ketoglutarate | CAS(6,6)/cc-pVDZ | -567.241522 | -567.241522 | **0.0000** |

**This validates that our PySCF → OpenFermion integral mapping is correct** (chemist notation → `h[p,q,r,s] = (ps|qr)` convention with proper spin-orbital expansion via `spinorb_from_spatial`).

### Correlation Energies and VQE Performance

| Rank | Compound | Corr Energy | VQE Error | VQE Recovery | Critics |
|---:|---|---:|---:|---:|---|
| 1 | NMN | -18.93 kcal/mol | 15.75 kcal/mol | 16.8% | 7/9 |
| 2 | Resveratrol | -16.47 kcal/mol | 14.18 kcal/mol | 13.9% | 7/9 |
| 3 | Fisetin | -14.40 kcal/mol | 10.70 kcal/mol | 25.7% | 7/9 |
| 4 | Dasatinib | -14.34 kcal/mol | 12.36 kcal/mol | 13.8% | 7/9 |
| 5 | Urolithin A | -13.57 kcal/mol | 8.48 kcal/mol | 37.5% | 7/9 |
| 6 | Quercetin | -4.99 kcal/mol | 2.07 kcal/mol | 58.6% | 7/9 |
| 7 | **Rapamycin** | -4.23 kcal/mol | **0.07 kcal/mol** | **98.4%** | **9/9** ✓ |
| 8 | AKG | -3.17 kcal/mol | 0.99 kcal/mol | 68.8% | 8/9 |
| 9 | Metformin | -0.89 kcal/mol | 0.77 kcal/mol | 13.7% | 8/9 |
| 10 | Spermidine | -0.73 kcal/mol | 0.65 kcal/mol | 11.3% | 8/9 |

**Only 1/10 passed all critics.** 4/10 achieved <1 kcal/mol VQE accuracy. The VQE struggles on compounds with strong electron correlation (>10 kcal/mol). This is an honest result — the EfficientSU2 ansatz with 2-3 layers cannot capture the ground state of 12-qubit Hamiltonians with significant correlation.

### Solvation Effects (ddCOSMO, water)

| Compound | ΔE_solv (kcal/mol) | Interpretation |
|---|---:|---|
| Resveratrol | **-13.42** | Strongly stabilized in water — consistent with phenolic hydroxyl groups |
| Fisetin | **-4.01** | Moderate aqueous stabilization |
| NMN | +2.87 | Slight destabilization (zwitterionic) |
| Metformin | +2.14 | Slight destabilization |
| Dasatinib | +1.45 | Near neutral |
| Quercetin | +1.33 | Near neutral |
| Spermidine | +1.15 | Near neutral |
| AKG | +1.10 | Near neutral |
| Rapamycin | +1.03 | Near neutral |
| Urolithin A | +0.39 | Near neutral |

### Basis Set Convergence

The pipeline ran CAS calculations at multiple levels. Energies lower with larger basis sets, as expected:

```
NMN:           CAS(4,4)/6-31G*  = -414.496553  →  CAS(6,6)/cc-pVDZ = -414.537629  (Δ = -25.8 kcal/mol)
Resveratrol:   CAS(4,4)/6-31G*  = -382.459374  →  CAS(6,6)/cc-pVDZ = -382.501778  (Δ = -26.6 kcal/mol)
Metformin:     CAS(4,4)/6-31G*  = -430.084813  →  CAS(6,6)/cc-pVDZ = -430.129723  (Δ = -28.2 kcal/mol)
Fisetin:       CAS(4,4)/6-31G*  = -418.312366  →  CAS(6,6)/cc-pVDZ = -418.352261  (Δ = -25.0 kcal/mol)
Dasatinib:     CAS(4,4)/6-31G*  = -317.760381  →  CAS(6,6)/cc-pVDZ = -317.790530  (Δ = -18.9 kcal/mol)
Spermidine:    CAS(5,4)/6-31G*  = -438.741255  →  CAS(7,6)/cc-pVDZ = -438.790252  (Δ = -30.7 kcal/mol)
Urolithin A:   CAS(5,4)/6-31G*  = -645.080885  →  CAS(7,6)/cc-pVDZ = -645.143551  (Δ = -39.3 kcal/mol)
AKG:           CAS(4,4)/6-31G*  = -567.177136  →  CAS(6,6)/cc-pVDZ = -567.241522  (Δ = -40.4 kcal/mol)
```

These 18-40 kcal/mol basis set effects dwarf the correlation energies (0.7-19 kcal/mol), which means the absolute energies are far from the complete basis set limit.

### Pipeline Statistics

| Metric | Value |
|---|---|
| Compounds studied | 10 |
| All critics passed | **1/10** (rapamycin only) |
| VQE < 1 kcal/mol accuracy | **4/10** |
| Average VQE error | 6.60 kcal/mol |
| Best VQE error | 0.07 kcal/mol (rapamycin) |
| Worst VQE error | 15.75 kcal/mol (NMN) |
| GPU peak memory | 8.5 GB / 206 GB |
| Total pipeline time | 19,983s (5h 33min) |
| Fragment compounds | 7/10 |

---

## The Bug We Found and Fixed

During development, we discovered and fixed **two critical bugs** in the PySCF → OpenFermion integral mapping. This is documented here because it is an easy mistake to make and could affect other researchers.

### Bug 1: Wrong Two-Electron Integral Transpose

PySCF stores two-electron integrals in **chemist notation**: `(pq|rs)`.  
OpenFermion's `InteractionOperator` expects: `h[p,q,r,s] = (ps|qr)`.  
This is **neither** standard chemist `(pq|rs)` **nor** standard physicist `<pq|rs>`.

The correct mapping from PySCF chemist → OpenFermion is:

```python
h2_of = h2.transpose(0, 2, 3, 1)  # (pq|rs) → h[p,q,r,s] = (ps|qr)
```

Source: [`openfermionpyscf/_run_pyscf.py`](https://github.com/quantumlib/OpenFermion-PySCF/blob/master/openfermionpyscf/_run_pyscf.py) line ~95.

### Bug 2: Missing Spin-Orbital Expansion

`InteractionOperator` expects **spin-orbital** tensors (2n × 2n), not spatial-orbital tensors (n × n). You must expand using:

```python
from openfermion.chem.molecular_data import spinorb_from_spatial
h1_so, h2_so = spinorb_from_spatial(h1_spatial, h2_of)
ham_op = InteractionOperator(e_core, h1_so, 0.5 * h2_so)
```

### How We Verified the Fix

We tested on molecules with known exact solutions:

| Molecule | Active Space | Our Exact Diag | PySCF CASCI | Δ |
|---|---|---:|---:|---:|
| H₂ | CAS(2,2)/STO-3G | -1.101151 Ha | -1.101151 Ha | 0.000000 kcal/mol |
| H₂O | CAS(4,4)/STO-3G | -75.013376 Ha | -75.013376 Ha | 0.000000 kcal/mol |

With the wrong convention, these disagreed by 0.05-2.5 Ha (30-1500 kcal/mol). After the fix: exact agreement.

### Why It Mattered

The previous version of this repository reported artificially good VQE results ("10/10 passed all critics, 0.249 kcal/mol average error") because the wrong Hamiltonian had an easier energy landscape. The VQE was finding the ground state of the *wrong* Hamiltonian easily. With the correct Hamiltonian, VQE convergence is much harder — which is the honest reality of variational quantum simulation.

---

## What This Does NOT Tell You

This section is arguably the most important part of this README.

### This is NOT drug discovery

We computed electronic structure — the quantum-mechanical properties of isolated molecular fragments in vacuum and implicit solvent. This tells you **nothing directly** about:

- Whether these compounds will extend human lifespan
- How they interact with proteins, DNA, or cell membranes
- Their bioavailability, metabolism, or toxicity
- Optimal dosing for any therapeutic purpose

The biological evidence for these compounds comes from published animal and human studies (see [References](#references)), not from our calculations.

### The correlation energies are not "rankings"

Larger |E(corr)| means the compound has more strongly correlated electrons in the chosen active space. This does NOT mean it is "more effective" or "more potent" as a drug. Correlation energy is a property of the electronic wavefunction, not a pharmacological metric.

### Fragment approximation limits everything

7 of 10 compounds were computed as fragments (active binding moiety only), not full molecules. Rapamycin (MW=914) was computed as a 12-atom piperidone fragment. This makes the absolute energies incomparable across compounds of different sizes.

### Basis set incompleteness dominates

The 18-40 kcal/mol basis set effects between 6-31G* and cc-pVDZ are far larger than the correlation energies we're studying. Our absolute energies are probably 50-100 kcal/mol from the complete basis set (CBS) limit.

### VQE mostly failed

Only 4/10 compounds achieved chemical accuracy (<1 kcal/mol VQE error). The EfficientSU2 ansatz with 2-3 layers is insufficiently expressive for 12-qubit Hamiltonians with strong correlation. This is consistent with the known limitations of hardware-efficient ansätze. A more sophisticated ansatz (UCCSD, ADAPT-VQE) or more VQE layers would likely improve results but at much higher computational cost.

---

## Honest Self-Criticism

### What we did right

1. **Found and fixed a real bug.** The two-electron integral convention mismatch between PySCF and OpenFermion is a genuine pitfall. We documented it thoroughly so others don't fall into the same trap.

2. **Validated rigorously.** The CASCI ≡ Exact Diag check at 0.0000 kcal/mol for all 10 compounds proves the integral mapping is now correct. We tested on known molecules (H₂, H₂O) first.

3. **Reported failures honestly.** The previous README claimed "10/10 passed all critics, avg 0.249 kcal/mol." The corrected results show 1/10 passed and 6.60 kcal/mol average error. We didn't hide this.

4. **Built infrastructure that works.** The 11-critic engine, multi-round escalation, solvation integration, and GPU pipeline are genuinely useful tools for quantum chemistry research.

### What we got wrong

1. **Initially shipped wrong results.** The previous version of this repository contained results from incorrect integral mapping. The "too good to be true" VQE numbers should have been a red flag earlier. The VQE was solving the wrong Hamiltonian.

2. **Overclaimed significance.** The old README framed this as having "three anti-aging clusters" and correlation energy "rankings." In reality, computing the electronic structure of known molecules at small active space levels is a standard quantum chemistry exercise, not a discovery.

3. **The compound selection is not scientifically rigorous for QC.** We picked 10 compounds that already have published evidence for lifespan extension. Computing their electronic structure doesn't add new biological evidence — it's a computational characterization, not a validation of their efficacy.

4. **Fragment approximation undermines comparisons.** Computing a 12-atom fragment of rapamycin (MW=914) and comparing its correlation energy to a 15-atom fragment of NMN (MW=334) is not physically meaningful. Rankings based on this were misleading.

### What this project actually contributes

1. **A documented PySCF → OpenFermion integral mapping pitfall** with a clear fix and verification procedure. This is genuinely useful for the quantum chemistry community.

2. **A working GPU-accelerated CASCI + VQE pipeline** that runs on AMD MI300X with ROCm. The infrastructure (critic engine, escalation, solvation, checkpointing) could be repurposed for other quantum chemistry studies.

3. **Honest VQE benchmarking data.** Showing that EfficientSU2(2-3 layers) fails on most 12-qubit molecular Hamiltonians with moderate correlation is a useful data point for the variational quantum simulation community.

4. **A case study in computational honesty.** Finding that your initial results are wrong, fixing the bug, and publishing the much-less-impressive corrected results is how science should work.

### What would make this study meaningful

If we wanted to do this properly, we would need:

| Improvement | Why It Matters |
|---|---|
| Full molecules, not fragments | Physically meaningful energies |
| Larger active spaces (CAS(12,12)+) | Capture more correlation |
| CBS extrapolation (cc-pVDZ/TZ/QZ) | Remove basis set error |
| Explicit solvent (QM/MM) | Realistic biological environment |
| Protein-ligand binding (QM/MM) | Actual drug-relevant property |
| ADAPT-VQE or UCCSD ansatz | Better VQE convergence |
| Comparison to experiment | Validation against known data |

---

## Methods

### Pipeline Architecture

```
  ┌──────────┐    ┌──────────────┐    ┌───────────┐    ┌────────────────┐    ┌──────────┐    ┌──────────┐
  │ Phase 1   │───▶│ Phase 2       │───▶│ Jordan-   │───▶│ GPU Exact      │───▶│ GPU VQE   │───▶│ Phase 3   │
  │ Geometry  │    │ CASCI         │    │ Wigner    │    │ Diag           │    │ 10 restarts│   │ Solvation │
  │ Opt (HF)  │    │ Escalation    │    │ + spinorb │    │ (eigvalsh)     │    │ LBFGS     │    │ ddCOSMO   │
  └──────────┘    └──────────────┘    └───────────┘    └────────────────┘    └──────────┘    └──────────┘
                   5 rounds:
                   R1: CAS(4,4)/6-31G*
                   R2: CAS(6,6)/6-31G*
                   R3: CAS(6,6)/cc-pVDZ
                   R4: CAS(8,8)/cc-pVDZ
                   R5: CAS(8,8)/cc-pVTZ
```

### Integral Convention (The Hard Part)

```python
# PySCF gives chemist notation: (pq|rs)
h2_pyscf = mc.get_h2eff()  # shape (n, n, n, n)

# OpenFermion InteractionOperator expects: h[p,q,r,s] = (ps|qr)
h2_of = h2_pyscf.transpose(0, 2, 3, 1)

# InteractionOperator expects SPIN orbitals (2n × 2n), not spatial (n × n)
from openfermion.chem.molecular_data import spinorb_from_spatial
h1_so, h2_so = spinorb_from_spatial(h1_spatial, h2_of)

# The 0.5 factor: InteractionOperator uses Σ h2[p,q,r,s] a†p a†q ar as
# but the physical operator has 1/2 Σ, so pass 0.5 * h2_so
ham = InteractionOperator(e_core, h1_so, 0.5 * h2_so)
qubit_ham = jordan_wigner(ham)
```

### 11-Critic Validation Engine

| Critic | Test | Threshold |
|---|---|---|
| C1 | Variational principle: E(VQE) ≥ E(exact) | within 1e-6 Ha |
| C2 | Negative correlation energy | E(corr) < 0 |
| C3 | Correlation fraction | \|E(corr)/E(HF)\| < 10% |
| C4 | Positive energy gap | ΔE ≥ 0 |
| C5 | VQE recovery | > 95% of correlation energy |
| C6 | Algorithmic accuracy | \|E(VQE) - E(exact)\| < 1.6 mHa |
| C7 | CASCI below HF | E(CASCI) < E(HF) |
| C8 | Hamiltonian terms | N(Pauli) > 0 |
| C9 | Basis set convergence | Checked if multi-basis data available |
| C10 | Active space convergence | Checked if multi-CAS data available |
| C11 | HF convergence | HF energy converged |

### Computational Resources

| Resource | Specification |
|---|---|
| GPU | AMD Instinct MI300X, 206 GB HBM3, ROCm 6.2 |
| GPU memory used | 8.5 GB peak (4% of available) |
| Total pipeline time | 19,983 seconds (5h 33min) |
| Largest matrix | 4096 × 4096 (CAS(6,6) = 12 qubits) |
| VQE restarts per compound | 10 |
| Solvation model | ddCOSMO (ε=78.39, water) |

---

## The Ten Compounds

These compounds were selected based on published evidence for lifespan extension in model organisms:

| ID | Compound | Key Reference | Model | Effect |
|---|---|---|---|---|
| LNG-001 | NMN | Mills et al., 2016 | Mouse | NAD⁺ restoration |
| LNG-002 | Resveratrol | Baur et al., 2006 | Mouse | SIRT1 activation |
| LNG-003 | Rapamycin | Harrison et al., 2009 | Mouse | mTOR inhibition |
| LNG-004 | Metformin | Bannister et al., 2014 | Human (T2D) | AMPK activation |
| LNG-005 | Quercetin | Zhu et al., 2015 | Mouse | Senolytic (with D) |
| LNG-006 | Fisetin | Yousefzadeh et al., 2018 | Mouse | Senolytic |
| LNG-007 | Dasatinib | Zhu et al., 2015 | Mouse | Senolytic (with Q) |
| LNG-008 | Spermidine | Eisenberg et al., 2009 | Yeast/Mouse | Autophagy |
| LNG-009 | Urolithin A | Ryu et al., 2016 | C. elegans | Mitophagy |
| LNG-010 | AKG | Asadi Shahmirzadi, 2020 | Mouse | TCA/Epigenetics |

**Important:** We computed the electronic structure of these molecules. We did not evaluate their biological effects. The biological evidence cited above comes from the original published studies.

---

## Repository Structure

```
quantum-longevity-research/
├── README.md                              # This file
├── RESEARCH_ARTICLE.md                    # Extended research article
├── LICENSE                                # MIT License
├── requirements.txt                       # Python dependencies
│
├── scripts/
│   ├── publication_gpu_pipeline.py        # ★ Main pipeline (corrected integral convention)
│   ├── test_final.py                      # ★ H2/H2O verification of integral fix
│   ├── enhanced_properties.py             # DFT B3LYP property calculations
│   ├── gpu_quantum_pipeline.py            # GPU quantum pipeline utilities
│   └── ...                                # Other utility scripts
│
├── results/
│   ├── publication_results_v2.json        # ★ Corrected pipeline results (10 compounds)
│   └── enhanced_properties.json           # DFT B3LYP properties (HOMO-LUMO, dipole, etc.)
│
├── src/                                   # Web platform (Flask)
│   ├── app.py                             # Flask API server
│   ├── longevity_data.py                  # Compound data module
│   └── longevity_sim.py                   # Simulation engine
│
├── models/                                # Compound databases
│   ├── longevity_compounds.json           # 10 compound definitions
│   └── longevity_targets.json             # Biological target mappings
│
└── docs/assets/                           # Graphics
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- AMD MI300X GPU with ROCm 6.2 (for GPU acceleration)
- Or any machine with CPU (PySCF supports CPU-only mode)

### Installation

```bash
git clone https://github.com/qubitpage/quantum-longevity-research.git
cd quantum-longevity-research
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the Pipeline

```bash
# Set environment variable for IBM Quantum features (optional)
export IBM_QUANTUM_TOKEN="your-token-here"

# Full 10-compound CASCI/VQE pipeline
python scripts/publication_gpu_pipeline.py

# Verify integral convention fix (H2 + H2O)
python scripts/test_final.py
```

### Verify the Fix Yourself

```python
from pyscf import gto, scf, mcscf
from openfermion import InteractionOperator, jordan_wigner
from openfermion.chem.molecular_data import spinorb_from_spatial
import numpy as np

# Build H2
mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
mf = scf.RHF(mol).run()
mc = mcscf.CASCI(mf, 2, 2).run()

# Extract integrals
h1, e_core = mc.get_h1eff()
h2 = mc.get_h2eff()

# Correct mapping: transpose(0,2,3,1) + spinorb_from_spatial
h2_of = np.asarray(h2.transpose(0, 2, 3, 1))
h1_so, h2_so = spinorb_from_spatial(h1, h2_of)
ham = InteractionOperator(float(e_core), h1_so, 0.5 * h2_so)

# Jordan-Wigner and diagonalize
from openfermion import get_sparse_operator
import scipy.sparse.linalg as sla
sparse_ham = get_sparse_operator(ham)
eigvals = sla.eigsh(sparse_ham, k=1, which='SA')[0]

print(f"PySCF CASCI: {mc.e_tot:.10f}")
print(f"Our exact:   {eigvals[0]:.10f}")
print(f"Delta = {abs(mc.e_tot - eigvals[0]) * 627.509:.6f} kcal/mol")  # Should be 0.000000
```

---

## References

1. López-Otín, C. et al. (2023). Hallmarks of aging: An expanding universe. *Cell*, 186(2), 243-278.
2. Mills, K.F. et al. (2016). Long-term NMN administration mitigates age-associated decline. *Cell Metab.*, 24(6), 795-806.
3. Baur, J.A. et al. (2006). Resveratrol improves health and survival on high-calorie diet. *Nature*, 444, 337-342.
4. Harrison, D.E. et al. (2009). Rapamycin fed late in life extends lifespan. *Nature*, 460, 392-395.
5. Bannister, C.A. et al. (2014). Can people with T2D live longer? *Diabetes Obes. Metab.*, 16(11), 1165-1173.
6. Zhu, Y. et al. (2015). From transcriptome to senolytic drugs. *Aging Cell*, 14(4), 644-658.
7. Yousefzadeh, M.J. et al. (2018). Fisetin is a senotherapeutic. *EBioMedicine*, 36, 18-28.
8. Eisenberg, T. et al. (2009). Spermidine promotes longevity. *Nat. Cell Biol.*, 11(11), 1305-1314.
9. Ryu, D. et al. (2016). Urolithin A induces mitophagy. *Nature Med.*, 22(8), 879-888.
10. Asadi Shahmirzadi, A. et al. (2020). AKG extends lifespan. *Cell Metab.*, 32(3), 447-456.
11. Sun, Q. et al. (2020). PySCF: Recent developments. *J. Chem. Phys.*, 153(2), 024109.
12. McClean, J.R. et al. (2020). OpenFermion. *Quantum Sci. Technol.*, 5(3), 034014.
13. Kandala, A. et al. (2017). Hardware-efficient VQE. *Nature*, 549, 242-246.

---

## Disclaimer

This repository contains computational quantum chemistry results. **It is not medical advice.** The biological efficacy of the studied compounds comes from published literature, not from our calculations. Computing the electronic structure of a molecule says nothing about its safety, dosing, or therapeutic value. Consult a healthcare professional before starting any supplementation.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>Qubit OS Research Laboratory</strong><br>
  May 2026<br>
  <sub>GPU: AMD Instinct MI300X (206 GB HBM3) • PySCF 2.13.0 + OpenFermion 1.7.1 + PyTorch 2.5.1+ROCm 6.2</sub>
</p>
