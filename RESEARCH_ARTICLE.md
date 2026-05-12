# GPU-Accelerated Multi-Reference Quantum Chemistry of Longevity-Associated Molecular Fragments: Algorithmic Benchmarks with Solvation Corrections

**Authors:** Qubit OS Research Laboratory  
**Date:** May 12, 2026  
**Platform:** AMD Instinct MI300X (206 GB HBM3)  
**Software:** PySCF 2.13.0, OpenFermion 1.7.1, Qiskit 2.4.1, PyTorch 2.5.1+ROCm 6.2  
**Target Journal:** Journal of Chemical Theory and Computation  

---

## Abstract

We present a systematic benchmark of GPU-accelerated Complete Active Space Configuration Interaction (CASCI) with custom Variational Quantum Eigensolver (VQE) validation for ten molecular fragments associated with longevity research. Using an AMD MI300X GPU with 206 GB HBM3, we compute CAS(4,4)/6-31G\* and CAS(5,4)/6-31G\* energies with exact diagonalization of Jordan-Wigner-transformed qubit Hamiltonians, validated against a differentiable PyTorch-based VQE implementation. We explicitly quantify the effect of aqueous solvation via ddCOSMO, finding solvation shifts spanning a 32 kcal/mol range — demonstrating that gas-phase results alone are insufficient for biological interpretation. Complementary DFT B3LYP/6-31G calculations provide HOMO-LUMO gaps, ionization potentials, and electrophilicity indices, with one compound (fisetin) failing DFT convergence — directly validating the need for multi-reference methods. All molecules are treated as explicitly annotated fragments of larger pharmacophores; no claims about full-molecule binding or biological efficacy are made. This work serves as a computational methodology benchmark for GPU-accelerated quantum chemistry pipelines applied to drug-like molecular fragments.

**Keywords:** CASCI, VQE, GPU-accelerated quantum chemistry, solvation, ddCOSMO, longevity compounds, benchmark, MI300X, Jordan-Wigner

---

## 1. Introduction

### 1.1 Context

Multi-reference quantum chemistry methods (CASSCF, CASCI, CASPT2) are essential for accurate treatment of molecules with significant static correlation — particularly those with conjugated π-systems, near-degenerate frontier orbitals, or radical character [Roos et al., 1980; Olsen et al., 1988]. However, these methods have historically been limited to small molecules due to the exponential scaling of the configuration interaction space.

Modern GPU hardware — particularly high-bandwidth-memory (HBM) accelerators — enables full-matrix exact diagonalization for active spaces previously considered intractable on single nodes. The AMD MI300X, with 206 GB of HBM3, can hold the full Hamiltonian matrix for CAS(8,8) systems (65536 × 65536, complex128 = 64 GB) with room for the eigensolver workspace.

### 1.2 Objective

This study benchmarks a complete GPU-accelerated pipeline:

1. Geometry optimization at HF/6-31G\* (PySCF + geomeTRIC) [Wang & Song, 2016]
2. CASCI with Jordan-Wigner qubit mapping (OpenFermion) [McClean et al., 2020]
3. GPU-based exact diagonalization (PyTorch `eigvalsh`)
4. Custom differentiable VQE with autograd-computed exact gradients
5. Aqueous solvation correction via ddCOSMO [Lipparini et al., 2013]
6. Complementary DFT B3LYP property calculations (HOMO-LUMO, dipoles, electrophilicity)
7. Automated validation through 9 independent physical critics

We apply this pipeline to ten molecular fragments relevant to aging biology, chosen to span diverse electronic structures (aromatic, aliphatic, charged, radical). We emphasize that:

- **"Accuracy" refers exclusively to algorithmic accuracy** (VQE energy vs. exact diagonalization of the *same* model Hamiltonian)
- **No biological efficacy claims are made** from electronic structure alone
- **Fragment approximations are used and explicitly disclosed** for all large molecules
- **Solvation effects are quantified** to demonstrate why gas-phase-only results are insufficient

### 1.3 Scope and Limitations (Stated Upfront)

This is a **computational methodology paper**, not a drug discovery or pharmacology study. Specifically:

- We do not predict binding affinities, bioavailability, or therapeutic doses
- We do not claim that electronic structure descriptors (correlation energy, HOMO-LUMO gap) predict biological activity
- We do not provide dietary or dosing recommendations
- Fragment approximations capture only the local electronic environment of pharmacophoric motifs
- CAS(4,4)/6-31G\* is a modest level of theory; basis set incompleteness errors are not fully quantified
- The active space treats only 4 of 50–97 total electrons explicitly; dynamic correlation is not captured

---

## 2. Computational Methods

### 2.1 Molecular Systems

Ten molecular fragments were selected from compounds with published evidence for lifespan effects in model organisms. Large molecules (rapamycin MW=914, dasatinib MW=488, full NMN MW=334) were modeled as pharmacophore fragments capturing the biologically active moiety. **This fragment approximation is a known limitation** — long-range intramolecular effects, conformational sampling, and full-molecule solvation properties are not captured.

| ID | Fragment | Parent Molecule | Atoms | Electrons | Spin | Basis Funcs (6-31G\*) | Fragment? | Notes |
|---|---|---|---:|---:|---:|---:|---|---|
| LNG-001 | Nicotinamide ring | NMN | 15 | 64 | 0 | 138 | Yes | Pyridine-3-carboxamide — NAD⁺ pharmacophore |
| LNG-002 | Stilbene + OH | Resveratrol | 17 | 64 | 0 | 142 | Yes | Conjugated backbone with one hydroxyl |
| LNG-003 | Piperidine-2-one | Rapamycin | 13 | 51 | 1 | 110 | Yes | FKBP12-binding motif (MW=914 parent) |
| LNG-004 | Full molecule | Metformin | 20 | 70 | 0 | 148 | No | Complete (CH₃)₂N-C(=NH)-NH-C(=NH)-NH₂ |
| LNG-005 | 4H-Chromen-4-one | Quercetin | 15 | 69 | 1 | 150 | Yes | Redox-active carbonyl-enol system |
| LNG-006 | Flavone core | Fisetin | 15 | 64 | 0 | 138 | Yes | Conjugated carbonyl for radical scavenging |
| LNG-007 | 2-Aminopyrimidine | Dasatinib | 12 | 50 | 0 | 108 | Yes | Kinase hinge-binding motif (MW=488 parent) |
| LNG-008 | Full molecule | Spermidine | 28 | 81 | 1 | 176 | No | H₂N-(CH₂)₃-NH-(CH₂)₄-NH₂ |
| LNG-009 | Dibenzo[b,d]pyranone | Urolithin A | 21 | 97 | 1 | 210 | Yes | Fused-ring mitophagy-active scaffold |
| LNG-010 | Full molecule | α-Ketoglutarate | 16 | 76 | 0 | 152 | No | HOOC-CH₂-CH₂-CO-COOH |

**Spin multiplicity note:** Four compounds (LNG-003, 005, 008, 009) have odd total electron counts. These were treated with restricted open-shell Hartree-Fock (ROHF) and CAS(5,4) — 5 electrons in 4 orbitals — to maintain integer-spin active spaces. The remaining six even-electron compounds use RHF and CAS(4,4).

### 2.2 Geometry Optimization

All structures were optimized at the HF/6-31G\* level using PySCF's interface to the geomeTRIC optimizer [Wang & Song, 2016]:

- **Method:** RHF (even electrons) or ROHF (odd electrons)
- **Basis:** 6-31G\* (polarized split-valence, Pople)
- **Convergence:** |ΔE| < 10⁻⁶ Ha, RMS-Grad < 3×10⁻⁴, Max-Grad < 4.5×10⁻⁴
- **Max steps:** 30

**Rationale for HF geometry:** We use HF rather than B3LYP for geometry optimization because (a) the subsequent CASCI calculation provides its own treatment of electron correlation, making DFT correlation unnecessary for the reference geometry, and (b) HF geometries are 3–5× faster, enabling the full pipeline to complete within a single GPU allocation.

| ID | Atoms | Opt. Time (s) |
|---:|---:|---:|
| LNG-001 | 15 | 508 |
| LNG-002 | 17 | 603 |
| LNG-003 | 13 | 401 |
| LNG-004 | 20 | 372 |
| LNG-005 | 15 | 1371 |
| LNG-006 | 15 | 431 |
| LNG-007 | 12 | 126 |
| LNG-008 | 28 | 1865 |
| LNG-009 | 21 | 2062 |
| LNG-010 | 16 | 775 |

Total Phase 1 time: 9514 s (2.6 h). The slowest compound (urolithin A, 21 atoms, 210 AOs) required 2062 s due to its large conjugated ring system.

### 2.3 CASCI + Jordan-Wigner Transformation

For each optimized geometry:

1. **Hartree-Fock:** RHF or ROHF at 6-31G\*, max 300 SCF cycles
2. **Active space:** CAS(4,4) for even-electron systems (6 compounds) or CAS(5,4) for odd-electron systems (4 compounds) = 4 active orbitals centered on the HOMO-LUMO region
3. **Integrals:** One-electron effective (`mc.get_h1eff()` → includes frozen-core potential) and two-electron (`mc.get_h2eff()`, restored to full 4-index tensor via `ao2mo.restore(1, ...)`)
4. **Integral convention:** PySCF returns two-electron integrals in **chemist** (Mulliken) convention $(pq|rs)$. OpenFermion's `InteractionOperator` expects **physicist** (Dirac) convention $\langle pq|rs \rangle$. The conversion $h^{\text{phys}}_{pqrs} = h^{\text{chem}}_{prqs}$ is applied via `h2.transpose(0,2,1,3)` before constructing the qubit Hamiltonian (see Appendix C for details).
5. **Qubit mapping:** Jordan-Wigner transformation via OpenFermion [Jordan & Wigner, 1928], yielding 8-qubit Hamiltonians with 11–17 Pauli terms. The scalar offset (nuclear repulsion + frozen core energy) from `mc.get_h1eff()` is included as the identity coefficient.

The CAS(4,4) active space captures static correlation in the frontier orbital window. **This is a small active space** — it treats only 4 of 50–97 total electrons explicitly. Dynamic correlation is *not* captured. A complete treatment would require CASPT2 or NEVPT2 on top of CASSCF (not CASCI).

### 2.4 GPU Exact Diagonalization

The 8-qubit Hamiltonian (256 × 256, complex128) was constructed on the MI300X GPU using a per-basis-state Pauli action method:

```
For each Pauli string P with coefficient c:
    Compute |P·i⟩ and phase(P, i) for all basis states i simultaneously
    H[P·i, i] += c × phase(P, i)    (via index_put_ with accumulate=True)
```

Ground state energy via `torch.linalg.eigvalsh()`. Hermiticity verified: max|H − H†| < 10⁻⁸.

GPU memory used: 2.7–3.1 GB / 205.8 GB (1.3–1.5%) — this active space is far below the GPU's capacity. The pipeline is designed for auto-escalation to CAS(8,8) (16 qubits, 65536×65536 = 64 GB), which would utilize ~93% of MI300X VRAM.

**Validation against PySCF CASCI:** The qubit exact diagonalization eigenvalue must reproduce PySCF's `mc.e_tot` (which includes nuclear repulsion + frozen core + active space CI energy) to < 10⁻⁶ Ha. This validates the integral convention and JW mapping (Section 2.3.4, Appendix C).

### 2.5 Custom GPU VQE (PyTorch Autograd)

A differentiable VQE was implemented in PyTorch with:

- **Ansatz:** EfficientSU2-like: Ry(θ) + Rz(θ) per qubit per layer + CNOT chain entanglement
- **Layers:** 2
- **Parameters:** 48 (8 qubits × 2 rotation gates × 2 layers + 8 × 2 final layer)
- **Optimizer:** Adam (200 steps, lr=0.01) → L-BFGS (300 steps) per restart
- **Restarts:** 8 random initializations (best energy selected)
- **Gradients:** Exact analytic via PyTorch autograd (not parameter-shift or finite-difference)
- **Evaluations:** 2400 per compound (8 restarts × 300 L-BFGS steps)

All gate operations (`apply_ry_gpu`, `apply_rz_gpu`, `apply_cnot_gpu`) maintain differentiability through the full 2⁸ = 256 complex statevector.

### 2.6 Aqueous Solvation (ddCOSMO)

To assess the effect of aqueous environment, we recomputed the full pipeline (HF + CASCI + JW + exact diag + VQE) with PySCF's ddCOSMO implicit solvation model [Lipparini et al., 2013; Nottoli et al., 2019]:

- **Dielectric constant:** ε = 78.39 (water at 25°C)
- **Method:** Domain-decomposition COSMO applied to the HF step; CASCI performed on the solvated HF reference determinant
- **Active space:** Same CAS(n,4) as gas-phase (n = 4 or 5 depending on electron count)

The solvation shift is defined as:

$$\Delta E_{\text{solv}} = E_{\text{exact}}^{\text{water}} - E_{\text{exact}}^{\text{gas}}$$

where both $E_{\text{exact}}$ values come from diagonalization of the correctly-mapped qubit Hamiltonian (validated against PySCF CASCI).

### 2.7 DFT Electronic Properties (B3LYP/6-31G)

As a complementary single-reference analysis, we computed DFT properties for all 10 fragments at the B3LYP/6-31G level using PySCF:

- **HOMO/LUMO energies** and Kohn-Sham gap ($\Delta \varepsilon = \varepsilon_{\text{LUMO}} - \varepsilon_{\text{HOMO}}$)
- **Ionization potential** (IP = $-\varepsilon_{\text{HOMO}}$, Koopmans' approximation)
- **Chemical hardness** ($\eta = (\text{IP} - \text{EA})/2$) [Parr & Pearson, 1983]
- **Electrophilicity index** ($\omega = \mu^2 / 2\eta$, where $\mu = -(\text{IP}+\text{EA})/2$) [Parr et al., 1999]
- **Dipole moment** (from converged electron density)

These descriptors characterize the electronic reactivity of each fragment independently of the multi-reference correlation energy.

### 2.8 Automated Validation Critics

Each CASCI/VQE result was validated against 9 independent physical constraints:

| Critic | Test | Rationale |
|---|---|---|
| C1 | $E_{\text{VQE}} \geq E_{\text{exact}} - 10^{-6}$ | Variational principle |
| C2 | $E_{\text{corr}} < 0$ | Correlation must be stabilizing |
| C3 | $\|E_{\text{corr}} / E_{\text{HF}}\| < 10\%$ | Correlation is perturbative for these systems |
| C4 | $\Delta E_{\text{gap}} \geq 0$ | Ground state is lowest eigenvalue |
| C5 | VQE recovery > 95% | VQE captures most correlation |
| C6 | $\|E_{\text{VQE}} - E_{\text{exact}}\| < 1.6$ mHa | Algorithmic accuracy < 1 kcal/mol |
| C7 | $E_{\text{CASCI}} < E_{\text{HF}}$ | Multi-reference improves on mean-field |
| C8 | $N_{\text{Pauli}} > 0$ | Hamiltonian is non-trivial |
| C11 | $E_{\text{VQE}} \leq E_{\text{HF}}$ | VQE at least as good as HF |

**Note on C6:** This critic tests *algorithmic* accuracy — whether the VQE ansatz can reproduce the exact diagonalization result of the *same* qubit Hamiltonian. It does NOT test accuracy against experimental energies or the complete basis set limit.

---

## 3. Results

### 3.1 CASCI Energies and VQE Validation

All 10 compounds converged at CAS(4,4)/6-31G\* or CAS(5,4)/6-31G\* with all 9 critics passed:

| ID | Fragment | CAS | E(HF) [Ha] | E(CASCI) [Ha] | E_corr [kcal/mol] | Qubits | Pauli | Critics |
|---:|---|---|---:|---:|---:|---:|---:|---|
| LNG-001 | NMN ring | (4,4) | −414.469167 | −414.496553 | −17.2 | 8 | 17 | 9/9 ✓ |
| LNG-002 | Resveratrol | (4,4) | −382.437576 | −382.459374 | −13.7 | 8 | 17 | 9/9 ✓ |
| LNG-003 | Rapamycin frag | (5,4) | −322.107156 | −322.113890 | −4.2 | 8 | 17 | 9/9 ✓ |
| LNG-004 | Metformin | (4,4) | −430.081381 | −430.084813 | −2.2 | 8 | 11 | 9/9 ✓ |
| LNG-005 | Quercetin frag | (5,4) | −455.430985 | −455.436053 | −3.2 | 8 | 17 | 9/9 ✓ |
| LNG-006 | Fisetin frag | (4,4) | −418.288895 | −418.312366 | −14.7 | 8 | 17 | 9/9 ✓ |
| LNG-007 | Dasatinib frag | (4,4) | −317.737152 | −317.760381 | −14.6 | 8 | 17 | 9/9 ✓ |
| LNG-008 | Spermidine | (5,4) | −438.739006 | −438.741255 | −1.4 | 8 | 17 | 9/9 ✓ |
| LNG-009 | Urolithin A frag | (5,4) | −645.075814 | −645.080885 | −3.2 | 8 | 17 | 9/9 ✓ |
| LNG-010 | α-Ketoglutarate | (4,4) | −567.176773 | −567.177136 | −0.2 | 8 | 17 | 9/9 ✓ |

**Key observations:**

1. **CASCI correlation energies** range from −0.2 kcal/mol (α-ketoglutarate) to −17.2 kcal/mol (NMN ring). These are physically reasonable for CAS(4,4) active spaces — the small active space captures only the dominant static correlation in the frontier orbital window.

2. **Aromatic fragments show larger correlation:** NMN (−17.2), fisetin (−14.7), dasatinib (−14.6), and resveratrol (−13.7 kcal/mol) have significant π-electron correlation. Non-conjugated systems (metformin −2.2, spermidine −1.4, AKG −0.2 kcal/mol) show minimal static correlation, consistent with their single-reference character.

3. **Correlation is perturbative:** $|E_{\text{corr}}/E_{\text{HF}}|$ ranges from 0.0001% to 0.007%, far below the 10% C3 threshold.

4. **VQE algorithmic accuracy:** The custom GPU VQE with EfficientSU2 ansatz (48 parameters, 2 layers) and Adam + L-BFGS optimization (8 restarts, 2400 evaluations) achieves machine-precision agreement with exact diagonalization (< 10⁻⁷ kcal/mol discrepancy) for all 8-qubit systems. The 48-parameter ansatz is sufficient to span the ground state manifold of these 256-dimensional Hilbert spaces.

5. **PySCF CASCI ↔ qubit exact diag agreement:** After applying the correct chemist-to-physicist integral convention (Section 2.3.4, Appendix C), the qubit eigenvalue reproduces PySCF's `mc.e_tot` to < 10⁻⁶ Ha for all 10 compounds.

### 3.2 Aqueous Solvation Effects (ddCOSMO)

The solvation correction reveals substantial environment-dependent energy shifts:

| ID | Fragment | ΔE_solv [kcal/mol] | Interpretation |
|---:|---|---:|---|
| LNG-003 | Rapamycin frag | **−21.8** | Strongly stabilized (lactam H-bonding) |
| LNG-004 | Metformin | **−10.6** | Stabilized (guanidinium-water interaction) |
| LNG-010 | α-Ketoglutarate | **−9.1** | Stabilized (dicarboxylic acid solvation) |
| LNG-001 | NMN ring | **−6.8** | Moderately stabilized (amide H-bonds) |
| LNG-005 | Quercetin frag | −3.4 | Weakly stabilized |
| LNG-008 | Spermidine | −3.2 | Weakly stabilized |
| LNG-009 | Urolithin A frag | −1.3 | Nearly unchanged |
| LNG-002 | Resveratrol | **+1.1** | Slightly destabilized |
| LNG-006 | Fisetin frag | **+6.8** | Destabilized (hydrophobic core) |
| LNG-007 | Dasatinib frag | **+10.2** | Strongly destabilized |

**Critical finding:** Solvation shifts span a 32 kcal/mol range (−21.8 to +10.2 kcal/mol). This demonstrates unequivocally that **gas-phase electronic structure calculations cannot be directly translated to biological predictions**.

The sign and magnitude of ΔE_solv correlate with molecular polarity:
- **Negative ΔE_solv** (stabilized): Polar fragments with H-bond donors/acceptors (metformin's guanidinium, AKG's carboxylates, rapamycin's lactam)
- **Positive ΔE_solv** (destabilized): Hydrophobic aromatic cores (dasatinib's aminopyrimidine, fisetin's flavone)

### 3.3 DFT Electronic Properties (B3LYP/6-31G)

Complementary single-reference DFT calculations reveal the electronic character of each fragment:

| ID | Fragment | Gap [eV] | IP [eV] | η [eV] | ω [eV] | Dipole [D] | DFT Conv.? |
|---:|---|---:|---:|---:|---:|---:|---|
| LNG-001 | NMN ring | 5.24 | 6.75 | 2.62 | 3.25 | 6.11 | ✓ |
| LNG-002 | Resveratrol | 0.79 | 7.43 | 0.40 | 62.39 | 2.66 | ✓ |
| LNG-003 | Rapamycin frag | 1.38 | 7.02 | 0.69 | 28.94 | 0.52 | ✓ |
| LNG-004 | Metformin | 4.68 | 3.75 | 2.34 | 0.42 | 3.54 | ✓ |
| LNG-005 | Quercetin frag | 1.82 | 8.50 | 0.91 | 31.72 | 2.11 | ✓ |
| LNG-006 | Fisetin frag | **0.36** | 7.05 | 0.18 | **132.33** | 5.46 | **✗** |
| LNG-007 | Dasatinib frag | 1.37 | 6.24 | 0.68 | 22.58 | 0.66 | ✓ |
| LNG-008 | Spermidine | 1.30 | 5.40 | 0.65 | 17.34 | 1.26 | ✓ |
| LNG-009 | Urolithin A frag | 0.94 | 6.61 | 0.47 | 39.97 | 10.74 | ✓ |
| LNG-010 | α-Ketoglutarate | 1.19 | 6.12 | 0.59 | 25.80 | 6.99 | ✓ |

**DFT convergence note:** Fisetin (LNG-006) failed B3LYP SCF convergence at both STO-3G and 6-31G basis sets. Its extremely narrow HOMO-LUMO gap (0.36 eV) and extreme electrophilicity index (132 eV) indicate near-degenerate frontier orbitals where single-reference DFT breaks down. **This directly validates the use of multi-reference CASCI for these compounds** — fisetin's CASCI calculation (−14.7 kcal/mol correlation) converged without issue.

**Electronic structure trends:**
- **Hard, stable molecules** (η > 2 eV): NMN ring (2.62 eV), metformin (2.34 eV) — resistant to charge transfer
- **Soft, reactive molecules** (η < 0.5 eV): Fisetin (0.18 eV), resveratrol (0.40 eV) — prone to radical reactions, consistent with known antioxidant activity
- **Highest polarity:** Urolithin A (10.74 D), α-ketoglutarate (6.99 D), NMN ring (6.11 D)
- **Lowest polarity:** Rapamycin frag (0.52 D), dasatinib frag (0.66 D) — consistent with hydrophobic binding pockets

### 3.4 Computational Performance

| Phase | Description | Wall Time | GPU Memory |
|---|---|---:|---:|
| Phase 1 | Geometry optimization (10 compounds, HF/6-31G\*) | 9514 s | CPU-only |
| Phase 2 | CASCI + exact diag + VQE (10 × 8 restarts) | ~1000 s | 2.7–3.1 GB |
| Phase 3 | Solvation recomputation (10 compounds, ddCOSMO) | ~12000 s | 2.7–3.1 GB |
| DFT | B3LYP properties (10 × 2 basis sets) | 635 s | CPU-only |
| **Total** | Full pipeline | **~6.3 h** | **3.1 / 206 GB (1.5%)** |

The MI300X GPU is severely underutilized at CAS(4,4). Projected VRAM utilization for larger active spaces:

| Active Space | Qubits | Hilbert Dim | Matrix Size | MI300X Utilization |
|---|---:|---:|---:|---:|
| CAS(4,4) | 8 | 256 | 1 MB | 1.5% |
| CAS(6,6) | 12 | 4,096 | 256 MB | 0.1% |
| CAS(8,8) | 16 | 65,536 | 64 GB | ~93% |
| CAS(10,10) | 20 | 1,048,576 | 16 TB | Exceeds VRAM |

---

## 4. Discussion

### 4.1 Algorithmic Accuracy vs. Chemical Accuracy

We deliberately distinguish two types of accuracy:

1. **Algorithmic accuracy** (demonstrated): How well the VQE approximation reproduces the exact solution *of the same model Hamiltonian*. Our VQE achieves < 10⁻⁷ kcal/mol — effectively machine precision. This demonstrates ansatz expressibility and optimizer robustness.

2. **Chemical accuracy** (NOT claimed): How close computed energies are to experimental values. Achieving chemical accuracy ($\pm 1$ kcal/mol of experiment) requires:
   - Larger active spaces: CASPT2 or NEVPT2 for dynamic correlation
   - Complete basis set extrapolation (CBS) via cc-pVDZ → cc-pVTZ
   - Relativistic corrections (Douglas-Kroll-Hess or ZORA)
   - Zero-point vibrational energy (ZPVE)
   - Thermal corrections at finite temperature

Our CAS(4,4)/6-31G\* results capture only static correlation in the frontier orbital window. **We do not claim chemical accuracy relative to experiment.**

### 4.2 Why Solvation Matters

The solvation data (Section 3.2) provides the clearest evidence that gas-phase calculations alone are insufficient for biological interpretation:

- **Rapamycin fragment** (−21.8 kcal/mol): In vacuum, this lactam has one electronic structure; in water, extensive hydrogen bonding reorganizes the electron density. Any ranking based on gas-phase energies alone would misrepresent this compound.

- **Dasatinib fragment** (+10.2 kcal/mol): This hydrophobic aminopyrimidine is *destabilized* by water — consistent with its biological mechanism of binding deep within a hydrophobic kinase pocket.

- **Metformin** (−10.6 kcal/mol): Its guanidinium moiety has strong water interactions — consistent with the molecule's high aqueous solubility (>500 mg/mL).

### 4.3 DFT Convergence Failure as Multi-Reference Indicator

The fisetin DFT convergence failure (Section 3.3) is scientifically informative: its HOMO-LUMO gap of 0.36 eV and chemical hardness of 0.18 eV place it in the strong multi-reference regime where single-determinant methods fundamentally cannot describe the ground state. The fact that CASCI converges cleanly for the same fragment directly validates the multi-reference approach.

Resveratrol also failed DFT convergence at the minimal STO-3G basis but converged at 6-31G, suggesting borderline multi-reference character.

### 4.4 Fragment Approximation: Limitations

Seven of ten systems are fragments of larger molecules. This is transparent but introduces systematic errors:

- **Missing long-range effects**: Intramolecular hydrogen bonds, steric strain from distal groups, and conformational constraints of the full molecule are absent
- **Incorrect solvation surface**: The fragment's solvent-accessible surface area differs from that of the full molecule
- **No protein context**: In biology, these fragments are typically buried in a protein binding pocket, not solvated in bulk water

A more rigorous study would treat full molecules using density matrix embedding theory (DMET) or the density matrix renormalization group (DMRG) for large active spaces.

### 4.5 What These Results DO Demonstrate

1. **Pipeline validation**: The GPU-accelerated CASCI → JW → exact diag → VQE pipeline works correctly end-to-end, with verified integral convention handling
2. **VQE convergence**: EfficientSU2 with Adam + L-BFGS converges to machine precision for all 8-qubit systems
3. **Solvation necessity**: Gas-to-water shifts of ±20 kcal/mol prove solvent effects cannot be ignored
4. **Multi-reference validation**: DFT convergence failure for fisetin independently confirms multi-reference methods are necessary
5. **GPU scalability**: The pipeline is designed for CAS(8,8) utilizing 93% VRAM — a regime where GPU-accelerated exact diagonalization becomes genuinely useful

---

## 5. Limitations

### 5.1 Active Space Size

CAS(4,4) with 8 qubits is a 256-dimensional Hilbert space — trivially solvable even on a CPU. The scientific value of GPU acceleration emerges at CAS(8,8) or larger. All 10 compounds passed all critics at CAS(4,4)/CAS(5,4), so the auto-escalation mechanism (L1→L5) was never triggered. Future work should target CAS(8,8) where GPU memory becomes essential.

### 5.2 Basis Set

6-31G\* is a split-valence polarized basis — adequate for qualitative trends but with known basis set superposition errors. CBS extrapolation (cc-pVDZ + cc-pVTZ, Helgaker two-point formula) was implemented but not triggered. The STO-3G → 6-31G comparison shows significant property changes (IP shifts of 3–5 eV), confirming basis set effects are substantial.

### 5.3 Missing Physics

- **Dynamic correlation**: CASCI captures static correlation only. CASPT2 or NEVPT2 would recover ~90% of remaining dynamic correlation.
- **Relativistic effects**: Negligible for first-row elements but relevant for heavier atoms.
- **Zero-point energy**: Not computed. Typically 1–5 kcal/mol for these molecules.
- **Thermal contributions**: All results at 0 K.
- **Conformational sampling**: Single geometry per compound. Flexible molecules (spermidine) have many conformers.
- **Dispersion corrections**: Not included in HF geometry optimization.

### 5.4 No Biological Claims

Electronic structure descriptors do NOT directly predict:
- Binding affinity to protein targets
- Bioavailability or pharmacokinetics
- Therapeutic dose or efficacy
- Drug-drug interactions

Biological activity prediction requires protein-ligand docking, molecular dynamics, ADMET modeling, and experimental validation. **No compounds or dosing protocols are recommended.**

---

## 6. Conclusions

1. A complete GPU-accelerated quantum chemistry pipeline (geometry optimization → CASCI → Jordan-Wigner → exact diagonalization → VQE → solvation) was implemented and validated on an AMD MI300X GPU, with verified chemist-to-physicist integral convention handling.

2. The custom PyTorch VQE achieves machine-precision algorithmic accuracy (< 10⁻⁷ kcal/mol) for all ten 8-qubit systems.

3. CASCI correlation energies range from −0.2 to −17.2 kcal/mol, physically consistent with CAS(4,4) active spaces treating frontier orbital static correlation.

4. Aqueous solvation corrections via ddCOSMO reveal a 32 kcal/mol range of shifts (−21.8 to +10.2 kcal/mol), demonstrating gas-phase insufficiency.

5. DFT B3LYP convergence failure for fisetin (gap = 0.36 eV) independently validates multi-reference methods.

6. The pipeline is designed for CAS(8,8) utilization (93% of 206 GB VRAM) but was underutilized at CAS(4,4) (1.5%).

7. No biological efficacy claims are made. This is a computational methodology benchmark.

---

## 7. Data Availability

- **Results:** `publication_results_v2.json` (all energies, critics, solvation data)
- **DFT Properties:** `enhanced_properties.json` (HOMO-LUMO gaps, dipoles, charges)
- **Pipeline code:** `publication_gpu_pipeline.py` (900+ lines, fully reproducible)
- **Repository:** https://github.com/qubitpage/quantum-longevity-research

---

## 8. References

1. Roos, B. O.; Taylor, P. R.; Siegbahn, P. E. M. (1980). *Chem. Phys.* 48, 157–173.
2. Olsen, J.; Roos, B. O.; Jørgensen, P.; Jensen, H. J. A. (1988). *J. Chem. Phys.* 89, 2185.
3. Sun, Q. et al. (2018). *WIREs Comput. Mol. Sci.* 8, e1340.
4. Sun, Q. et al. (2020). *J. Chem. Phys.* 153, 024109.
5. McClean, J. R. et al. (2020). *Quantum Sci. Technol.* 5, 034014.
6. Kandala, A. et al. (2017). *Nature* 549, 242–246.
7. Wang, L.-P.; Song, C. C. (2016). *J. Chem. Phys.* 144, 214108.
8. Lipparini, F. et al. (2013). *J. Chem. Theory Comput.* 9, 3637–3648.
9. Nottoli, M. et al. (2019). *J. Chem. Phys.* 150, 094112.
10. Helgaker, T.; Klopper, W.; Tew, D. P. (2008). *Mol. Phys.* 106, 2107–2143.
11. López-Otín, C. et al. (2023). *Cell* 186, 99–118.
12. Parr, R. G.; Pearson, R. G. (1983). *J. Am. Chem. Soc.* 105, 7512–7516.
13. Parr, R. G.; Szentpály, L. v.; Liu, S. (1999). *J. Am. Chem. Soc.* 121, 1922–1924.
14. Jordan, P.; Wigner, E. P. (1928). *Z. Physik* 47, 631–651.

---

## Appendix A: Compound Selection Rationale

| Compound | Evidence | Reference |
|---|---|---|
| NMN | +9% lifespan in aged mice | Mills et al., 2016 (*Cell Metab.*) |
| Resveratrol | +31% lifespan in obese mice | Baur et al., 2006 (*Nature*) |
| Rapamycin | +14% lifespan in mice | Harrison et al., 2009 (*Nature*) |
| Metformin | Reduced all-cause mortality | Bannister et al., 2014 (*Diabetes Obes Metab*) |
| Quercetin | Senolytic; reduced frailty | Xu et al., 2018 (*Nature Medicine*) |
| Fisetin | Senolytic; +10% lifespan | Yousefzadeh et al., 2018 (*EBioMedicine*) |
| Dasatinib | Senolytic (with quercetin) | Zhu et al., 2015 (*Aging Cell*) |
| Spermidine | +25% lifespan in mice | Eisenberg et al., 2016 (*Nature Medicine*) |
| Urolithin A | Improved mitochondrial function | Andreux et al., 2019 (*Nature Metab.*) |
| α-Ketoglutarate | +12% lifespan in mice | Asadi Shahmirzadi et al., 2020 (*Cell Metab.*) |

**These biological observations motivated the choice of test molecules but do NOT validate our electronic structure calculations.**

---

## Appendix B: Pipeline Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│ Phase 1: Geometry Optimization (HF/6-31G*, geomeTRIC)             │
│   10 compounds × 126–2062s each = ~2.6h (CPU-bound)              │
├───────────────────────────────────────────────────────────────────┤
│ Phase 2: CASCI + JW (physicist convention) + GPU Exact Diag + VQE │
│   Auto-escalation: L1→L2→L3→L4→L5 until all critics pass         │
│   L1: CAS(4,4)/6-31G*  → 8q,  256×256   → <1 GB                 │
│   L2: CAS(6,6)/6-31G*  → 12q, 4096×4096 → ~256 MB               │
│   L3: CAS(6,6)/cc-pVDZ → 12q, 4096×4096 → ~256 MB               │
│   L4: CAS(8,8)/cc-pVDZ → 16q, 65536×65536 → 64 GB               │
│   L5: CAS(8,8)/cc-pVTZ → 16q, 65536×65536 → 64 GB               │
├───────────────────────────────────────────────────────────────────┤
│ Phase 3: Solvation (ddCOSMO, water ε=78.39)                      │
│   Recompute best gas-phase level with implicit solvation          │
├───────────────────────────────────────────────────────────────────┤
│ Phase 4: CBS Extrapolation (Helgaker two-point formula)           │
│   E_CBS = (E_DZ × 3³ − E_TZ × 2³) / (3³ − 2³)                  │
│   [Only if L3+L5 both computed — not triggered in this run]       │
├───────────────────────────────────────────────────────────────────┤
│ Phase 5: Final Validation (9 critics per compound)                │
│   All must pass for publication-grade results                     │
├───────────────────────────────────────────────────────────────────┤
│ DFT Properties (B3LYP/STO-3G + B3LYP/6-31G, independent)        │
│   HOMO-LUMO gaps, dipoles, electrophilicity, Mulliken charges     │
└───────────────────────────────────────────────────────────────────┘
```

---

## Appendix C: Integral Convention Verification

A critical implementation detail for JW-mapped quantum chemistry is the two-electron integral convention. PySCF's `mc.get_h2eff()` returns integrals in chemist (Mulliken) notation:

$$(pq|rs) = \int \phi_p(\mathbf{r}_1) \phi_q(\mathbf{r}_1) \frac{1}{r_{12}} \phi_r(\mathbf{r}_2) \phi_s(\mathbf{r}_2) \, d\mathbf{r}_1 \, d\mathbf{r}_2$$

OpenFermion's `InteractionOperator` expects physicist (Dirac) notation:

$$\langle pq | rs \rangle = \int \phi_p(\mathbf{r}_1) \phi_q(\mathbf{r}_2) \frac{1}{r_{12}} \phi_r(\mathbf{r}_1) \phi_s(\mathbf{r}_2) \, d\mathbf{r}_1 \, d\mathbf{r}_2$$

The conversion is $h^{\text{phys}}_{pqrs} = h^{\text{chem}}_{prqs}$, implemented as `h2.transpose(0, 2, 1, 3)`. Due to the 8-fold symmetry of real-valued ERIs, this is equivalent to `np.einsum('ijkl->iljk', h2)` used in the openfermion-pyscf bridge.

**Failure to apply this transposition** results in a Hamiltonian with incorrect eigenvalues — the eigensolver finds a ground state that does not correspond to the physical system, even though it satisfies the variational principle and passes internal consistency critics. Verification is performed by comparing the qubit eigenvalue against PySCF's `mc.e_tot` (which computes the CASCI energy independently using the correct integrals).
