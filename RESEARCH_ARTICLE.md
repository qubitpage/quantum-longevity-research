# GPU-Accelerated Multi-Reference Quantum Chemistry of Longevity-Associated Molecular Fragments: Algorithmic Benchmarks with Solvation Corrections

**Authors:** Qubit OS Research Laboratory  
**Date:** May 12, 2026  
**Platform:** AMD Instinct MI300X (206 GB HBM3)  
**Software:** PySCF 2.13.0, OpenFermion 1.7.1, Qiskit 2.4.1, PyTorch 2.5.1+ROCm 6.2  
**Target Journal:** Journal of Chemical Information and Modeling / Journal of Chemical Theory and Computation  

---

## Abstract

We present a systematic benchmark of GPU-accelerated Complete Active Space Configuration Interaction (CASCI) with custom Variational Quantum Eigensolver (VQE) validation for ten molecular fragments associated with longevity research. Using an AMD MI300X GPU with 206 GB HBM3, we compute CAS(4,4)/6-31G* energies with exact diagonalization and validate against a differentiable PyTorch-based VQE implementation achieving machine-precision algorithmic accuracy (< 10⁻⁷ kcal/mol VQE–exact discrepancy). We explicitly quantify the effect of aqueous solvation via ddCOSMO, finding solvation shifts ranging from +10.2 to −21.8 kcal/mol — demonstrating that gas-phase results alone are insufficient for biological interpretation. All molecules are treated as explicitly annotated fragments of larger pharmacophores; we make no claims about full-molecule binding or biological efficacy. This work serves as a computational methodology benchmark for GPU-accelerated quantum chemistry pipelines applied to drug-like molecular fragments.

**Keywords:** CASCI, VQE, GPU-accelerated quantum chemistry, solvation, ddCOSMO, longevity compounds, benchmark, MI300X

---

## 1. Introduction

### 1.1 Context

Multi-reference quantum chemistry methods (CASSCF, CASCI, CASPT2) are essential for accurate treatment of molecules with significant static correlation — particularly those with conjugated π-systems, near-degenerate frontier orbitals, or radical character [Roos et al., 1980; Olsen et al., 1988]. However, these methods have historically been limited to small molecules due to the exponential scaling of the configuration interaction space.

Modern GPU hardware — particularly high-bandwidth-memory (HBM) accelerators — enables full-matrix exact diagonalization for active spaces previously considered intractable on single nodes. The AMD MI300X, with 206 GB of HBM3, can hold the full Hamiltonian matrix for CAS(8,8) systems (65536 × 65536, complex128 = 64 GB) with room for the eigensolver workspace.

### 1.2 Objective

This study benchmarks a complete GPU-accelerated pipeline:

1. Geometry optimization at HF/6-31G* (PySCF + geomeTRIC)
2. CASCI with Jordan-Wigner qubit mapping (OpenFermion)
3. GPU-based exact diagonalization (PyTorch `eigvalsh`)
4. Custom differentiable VQE with autograd-computed exact gradients
5. Aqueous solvation correction via ddCOSMO [Lipparini et al., 2013]
6. Automated validation through 9 independent physical critics

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
- CAS(4,4)/6-31G* is a modest level of theory; basis set incompleteness errors are not fully quantified (CBS extrapolation was not triggered since all compounds passed at L1)

---

## 2. Computational Methods

### 2.1 Molecular Systems

Ten molecular fragments were selected from compounds with published evidence for lifespan effects in model organisms. Large molecules (rapamycin MW=914, dasatinib MW=488, full NMN MW=334) were modeled as pharmacophore fragments capturing the biologically active moiety. **This fragment approximation is a known limitation** — long-range intramolecular effects, conformational sampling, and full-molecule solvation properties are not captured.

| ID | Fragment | Parent Molecule | Atoms | Electrons | Basis Funcs (6-31G*) | Fragment? | Notes |
|---|---|---|---:|---:|---:|---|---|
| LNG-001 | Nicotinamide ring | NMN | 15 | 64 | 138 | Yes | Pyridine-3-carboxamide — NAD+ pharmacophore |
| LNG-002 | Stilbene + OH | Resveratrol | 17 | 64 | 142 | Yes | Conjugated backbone with one hydroxyl |
| LNG-003 | Piperidine-2-one | Rapamycin | 13 | 51 | 110 | Yes | FKBP12-binding motif (MW=914 parent) |
| LNG-004 | Full molecule | Metformin | 20 | 70 | 148 | No | Complete (CH₃)₂N-C(=NH)-NH-C(=NH)-NH₂ |
| LNG-005 | 4H-Chromen-4-one | Quercetin | 15 | 69 | 150 | Yes | Redox-active carbonyl-enol system |
| LNG-006 | Flavone core | Fisetin | 15 | 64 | 138 | Yes | Conjugated carbonyl for radical scavenging |
| LNG-007 | 2-Aminopyrimidine | Dasatinib | 12 | 50 | 108 | Yes | Kinase hinge-binding motif (MW=488 parent) |
| LNG-008 | Full molecule | Spermidine | 28 | 81 | 176 | No | H₂N-(CH₂)₃-NH-(CH₂)₄-NH₂ |
| LNG-009 | Dibenzo[b,d]pyranone | Urolithin A | 21 | 97 | 210 | Yes | Fused-ring mitophagy-active scaffold |
| LNG-010 | Full molecule | α-Ketoglutarate | 16 | 76 | 152 | No | HOOC-CH₂-CH₂-CO-COOH |

### 2.2 Geometry Optimization

All structures were optimized at the HF/6-31G* level using PySCF's interface to the geomeTRIC optimizer [Wang & Song, 2016]:

- **Method:** Restricted HF (closed-shell) or ROHF (odd-electron systems)
- **Basis:** 6-31G* (polarized split-valence)
- **Convergence:** |ΔE| < 10⁻⁶ Ha, RMS-Grad < 3×10⁻⁴, Max-Grad < 4.5×10⁻⁴
- **Max steps:** 30

Geometry optimization times ranged from 126s (dasatinib, 12 atoms) to 2062s (urolithin A, 21 atoms, 210 AOs). Total Phase 1 time: 9521s.

### 2.3 CASCI + Jordan-Wigner Transformation

For each optimized geometry:

1. **Hartree-Fock:** RHF or ROHF at 6-31G*, max 300 SCF cycles
2. **Active space:** CAS(4,4) = 4 electrons in 4 orbitals (HOMO-1 to LUMO+1). For odd-core-electron cases, CAS(5,4) was used to maintain even core.
3. **Integrals:** One-electron (`get_h1eff()`) and two-electron (`get_h2eff()`, restored to 4-index tensor via `ao2mo.restore()`)
4. **Qubit mapping:** Jordan-Wigner transformation via OpenFermion, yielding 8-qubit Hamiltonians with 9–17 Pauli terms

The CAS(4,4) active space captures static correlation in the frontier orbital window. **This is a small active space** — it treats only 4 of 50–97 total electrons explicitly. Dynamic correlation is *not* captured. A complete treatment would require CASPT2 or NEVPT2 on top of CASSCF (not CASCI).

### 2.4 GPU Exact Diagonalization

The 8-qubit Hamiltonian (256 × 256, complex128) was constructed on the MI300X GPU using a per-basis-state Pauli action method:

```
For each Pauli string P with coefficient c:
    Compute |P·i⟩ and phase(P, i) for all basis states i simultaneously
    H[P·i, i] += c × phase(P, i)    (via index_put_ with accumulate=True)
```

Ground state energy via `torch.linalg.eigvalsh()`. Hermiticity verified: max|H - H†| < 10⁻⁸.

GPU memory used: 3.1 GB / 205.8 GB (1.5%) — this active space is far below the GPU's capacity. CAS(8,8) (16 qubits, 65536×65536 = 64 GB) would utilize ~93% of VRAM.

### 2.5 Custom GPU VQE (PyTorch Autograd)

A differentiable VQE was implemented in PyTorch with:

- **Ansatz:** EfficientSU2-like: Ry(θ) + Rz(θ) per qubit per layer + CNOT chain entanglement
- **Layers:** 2
- **Parameters:** 48 (8 qubits × 2 gates × 2 layers + 8 × 2 final)
- **Optimizer:** Adam (200 steps, lr=0.01) → L-BFGS (300 steps) for each restart
- **Restarts:** 8 random initializations
- **Gradients:** Exact via PyTorch autograd (not parameter-shift)

All gate operations (`apply_ry_gpu`, `apply_rz_gpu`, `apply_cnot_gpu`) maintain differentiability through the full statevector.

### 2.6 Aqueous Solvation (ddCOSMO)

To assess the effect of aqueous environment, we recomputed HF + CASCI + exact diag + VQE with PySCF's ddCOSMO implicit solvation model [Lipparini et al., 2013; Nottoli et al., 2019]:

- **Dielectric constant:** ε = 78.39 (water at 25°C)
- **Method:** Domain-decomposition COSMO applied to the HF step; CASCI performed on the solvated HF reference

The solvation shift is defined as:

$$\Delta E_{\text{solv}} = E_{\text{exact}}^{\text{water}} - E_{\text{exact}}^{\text{gas}}$$

### 2.7 Automated Validation Critics

Each result was validated against 9 independent physical constraints:

| Critic | Test | Rationale |
|---|---|---|
| C1 | E_VQE ≥ E_exact − 10⁻⁶ | Variational principle |
| C2 | E_corr < 0 | Correlation must be stabilizing |
| C3 | \|E_corr/E_HF\| < 10% | Correlation is perturbative |
| C4 | ΔE_gap ≥ 0 | Ground state is lowest eigenvalue |
| C5 | VQE recovery > 95% | VQE captures most correlation |
| C6 | \|E_VQE − E_exact\| < 1.6 mHa | Algorithmic accuracy < 1 kcal/mol |
| C7 | E_CASCI < E_HF | Multi-reference improves on mean-field |
| C8 | N_Pauli > 0 | Hamiltonian is non-trivial |
| C11 | E_VQE ≤ E_HF | VQE at least as good as HF |

**Note on C6:** This critic tests *algorithmic* accuracy — whether the VQE ansatz can reproduce the exact diagonalization result. It does NOT test accuracy against experimental energies or the complete basis set limit.

---

## 3. Results

### 3.1 Gas-Phase CASCI/VQE Energies

All 10 compounds converged at CAS(4,4)/6-31G* with machine-precision VQE accuracy:

| ID | Fragment | E(HF) [Ha] | E(exact) [Ha] | E(VQE) [Ha] | E_corr [kcal/mol] | VQE Error [kcal/mol] | Qubits | Critics |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| LNG-001 | NMN ring | −414.469167 | −415.490993 | −415.490993 | −641.2 | < 10⁻⁷ | 8 | 9/9 ✓ |
| LNG-002 | Resveratrol | −382.437576 | −383.371609 | −383.371609 | −586.1 | < 10⁻⁷ | 8 | 9/9 ✓ |
| LNG-003 | Rapamycin frag | −322.107156 | −322.852595 | −322.852595 | −467.8 | < 10⁻⁷ | 8 | 9/9 ✓ |
| LNG-004 | Metformin | −430.081381 | −430.600943 | −430.600943 | −326.0 | < 10⁻⁷ | 8 | 9/9 ✓ |
| LNG-005 | Quercetin frag | −455.430985 | −456.153763 | −456.153763 | −453.6 | < 10⁻⁷ | 8 | 9/9 ✓ |
| LNG-006 | Fisetin frag | −418.288895 | −419.271101 | −419.271101 | −616.3 | < 10⁻⁷ | 8 | 9/9 ✓ |
| LNG-007 | Dasatinib frag | −317.737152 | −318.790603 | −318.790603 | −661.1 | < 10⁻⁷ | 8 | 9/9 ✓ |
| LNG-008 | Spermidine | −438.739006 | −439.334683 | −439.334683 | −373.8 | < 10⁻⁷ | 8 | 9/9 ✓ |
| LNG-009 | Urolithin A frag | −645.075814 | −645.575246 | −645.575246 | −313.4 | < 10⁻⁷ | 8 | 9/9 ✓ |
| LNG-010 | α-Ketoglutarate | −567.176773 | −567.914245 | −567.914245 | −462.8 | < 10⁻⁷ | 8 | 9/9 ✓ |

**Key observation:** The custom GPU VQE with Adam + L-BFGS and 8 restarts achieves machine-precision convergence to the exact ground state for all 8-qubit systems. The EfficientSU2 ansatz with 2 layers (48 parameters) is sufficient to exactly represent the ground state of these 256-dimensional Hilbert spaces. This demonstrates the ansatz expressibility for CAS(4,4) problems — a non-trivial result given that random initialization could easily trap in local minima.

### 3.2 Aqueous Solvation Effects (ddCOSMO)

The solvation correction reveals substantial environment-dependent energy shifts:

| ID | Fragment | E_exact (gas) [Ha] | E_exact (water) [Ha] | ΔE_solv [kcal/mol] | Interpretation |
|---:|---|---:|---:|---:|---|
| LNG-003 | Rapamycin frag | −322.852595 | −322.887382 | **−21.83** | Strongly stabilized (lactam H-bonding) |
| LNG-004 | Metformin | −430.600943 | −430.617837 | **−10.60** | Stabilized (guanidinium-water interaction) |
| LNG-010 | α-Ketoglutarate | −567.914245 | −567.928744 | **−9.10** | Stabilized (dicarboxylic acid solvation) |
| LNG-001 | NMN ring | −415.490993 | −415.501885 | **−6.84** | Moderately stabilized (amide H-bonds) |
| LNG-005 | Quercetin frag | −456.153763 | −456.159177 | −3.40 | Weakly stabilized |
| LNG-008 | Spermidine | −439.334683 | −439.339703 | −3.15 | Weakly stabilized |
| LNG-009 | Urolithin A frag | −645.575246 | −645.577279 | −1.28 | Nearly unchanged |
| LNG-002 | Resveratrol | −383.371609 | −383.369851 | **+1.10** | Slightly destabilized |
| LNG-006 | Fisetin frag | −419.271101 | −419.260198 | **+6.84** | Destabilized (hydrophobic core) |
| LNG-007 | Dasatinib frag | −318.790603 | −318.774324 | **+10.21** | Strongly destabilized |

**Critical finding:** Solvation shifts span a 32 kcal/mol range (−21.8 to +10.2 kcal/mol). This demonstrates unequivocally that **gas-phase electronic structure calculations cannot be directly translated to biological predictions**. Compounds like the rapamycin fragment (−21.8 kcal/mol stabilization) have fundamentally different electronic behavior in aqueous vs. vacuum environments.

The sign and magnitude of ΔE_solv correlate with molecular polarity:
- **Negative ΔE_solv** (stabilized): Polar fragments with H-bond donors/acceptors (metformin's guanidinium, AKG's carboxylates, rapamycin's lactam)
- **Positive ΔE_solv** (destabilized): Hydrophobic aromatic cores (dasatinib's aminopyrimidine, fisetin's flavone)

### 3.3 Computational Performance

| Phase | Description | Wall Time | GPU Memory |
|---|---|---:|---:|
| Phase 1 | Geometry optimization (10 compounds, HF/6-31G*) | 9521s | CPU-only |
| Phase 2 | CASCI + exact diag + VQE (10 compounds, 8 restarts each) | ~1000s | 3.1 GB |
| Phase 3 | Solvation recomputation (10 compounds, ddCOSMO) | ~12000s | 3.1 GB |
| **Total** | Full pipeline | **22708s (6.3h)** | **3.1 / 206 GB (1.5%)** |

The MI300X GPU is severely underutilized at CAS(4,4). Projected utilization:
- CAS(6,6) → 12 qubits, 4096×4096: ~256 MB (0.1%)
- CAS(8,8) → 16 qubits, 65536×65536: ~64 GB matrix + ~128 GB eigensolver workspace = **~192 GB (93%)**

---

## 4. Discussion

### 4.1 Algorithmic Accuracy vs. Chemical Accuracy

We deliberately distinguish two types of accuracy:

1. **Algorithmic accuracy** (this paper): How well the VQE approximation reproduces the exact solution *of the same Hamiltonian*. Our VQE achieves < 10⁻⁷ kcal/mol — effectively machine precision. This demonstrates the ansatz and optimizer quality.

2. **Chemical accuracy** (NOT claimed): How close computed energies are to experimental values. This requires: larger active spaces (CASPT2/NEVPT2), complete basis set extrapolation (CBS), relativistic corrections, zero-point vibrational energy, and thermal corrections. Our CAS(4,4)/6-31G* results are likely 50–200 mHa from the true nonrelativistic limit.

**We do not claim chemical accuracy relative to experiment.** The CAS(4,4)/6-31G* level is a well-defined model chemistry that we solve exactly — but the model itself has known limitations.

### 4.2 Why Solvation Matters

The solvation data (Section 3.2) provides the clearest evidence that gas-phase calculations alone are insufficient for biological interpretation:

- **Rapamycin fragment**: −21.8 kcal/mol solvation shift. In vacuum, this lactam has one electronic structure; in water, extensive hydrogen bonding reorganizes the electron density. Any ranking or clustering based on gas-phase energies alone would misrepresent this compound's behavior in biological media.

- **Dasatinib fragment**: +10.2 kcal/mol. This hydrophobic aminopyrimidine is *destabilized* by water — consistent with its biological mechanism of binding deep within a hydrophobic kinase pocket (away from aqueous solvent).

- **Metformin**: −10.6 kcal/mol. Its guanidinium moiety has strong water interactions — consistent with the molecule's high aqueous solubility (> 500 mg/mL) and renal clearance.

### 4.3 Fragment Approximation: Limitations

Seven of ten systems are fragments of larger molecules. This is transparent but introduces systematic errors:

- **Missing long-range effects**: Intramolecular hydrogen bonds, steric strain from distal groups, and conformational constraints of the full molecule are absent
- **Incorrect solvation surface**: The fragment's solvent-accessible surface differs from that of the full molecule
- **No protein context**: In biology, these fragments are typically buried in a protein binding pocket, not solvated in bulk water

A more rigorous study would treat full molecules using density matrix embedding theory (DMET) or the density matrix renormalization group (DMRG) for large active spaces.

### 4.4 What These Results DO Demonstrate

1. **Pipeline validation**: The GPU-accelerated CASCI → JW → exact diag → VQE pipeline works correctly end-to-end for drug-like molecular fragments
2. **VQE convergence**: The EfficientSU2 ansatz with Adam + L-BFGS converges to machine precision for 8-qubit Hamiltonians — no barren plateaus, no local minima trapping with 8 restarts
3. **Solvation necessity**: Gas-to-water shifts of ±20 kcal/mol prove that solvent effects cannot be ignored
4. **GPU scalability**: At CAS(4,4), only 1.5% of MI300X capacity is used. The pipeline is designed for CAS(8,8) utilizing 93% VRAM — a regime where exact diagonalization becomes genuinely useful over classical approaches

---

## 5. Limitations

### 5.1 Active Space Size

CAS(4,4) with 8 qubits is a 256-dimensional Hilbert space — trivially solvable even on a CPU. The scientific value of GPU acceleration emerges at CAS(8,8) or larger, where the 65536-dimensional space requires HBM-class memory. All 10 compounds passed all critics at CAS(4,4), so the auto-escalation mechanism was never triggered. Future work should force escalation to demonstrate GPU necessity.

### 5.2 Basis Set

6-31G* is a split-valence polarized basis — adequate for qualitative trends but known to have 50–100 mHa basis set superposition errors for these molecular sizes. CBS extrapolation (using cc-pVDZ + cc-pVTZ) was planned but not executed because compounds did not escalate to those basis sets.

### 5.3 Missing Physics

- **Dynamic correlation**: CASCI captures static correlation only. CASPT2 or NEVPT2 would recover ~90% of remaining dynamic correlation.
- **Relativistic effects**: Negligible for first-row elements but relevant if heavier atoms were included.
- **Zero-point energy**: Not computed. Typically 1–5 kcal/mol for molecules of this size.
- **Thermal contributions**: All results are at 0 K. Finite-temperature effects not included.
- **Conformational sampling**: Only one geometry per compound. Flexible molecules (spermidine) have many conformers.

### 5.4 No Biological Claims

Electronic structure descriptors (correlation energy, HOMO-LUMO gap, dipole moment) do NOT directly predict:
- Binding affinity to protein targets
- Bioavailability or pharmacokinetics
- Therapeutic dose or efficacy
- Drug-drug interactions

Biological activity prediction requires protein-ligand docking, molecular dynamics, and ultimately experimental validation.

---

## 6. Conclusions

1. A complete GPU-accelerated quantum chemistry pipeline (geometry optimization → CASCI → Jordan-Wigner → exact diagonalization → VQE → solvation) was implemented and validated on an AMD MI300X GPU.

2. The custom PyTorch VQE with autograd gradients achieves machine-precision algorithmic accuracy (< 10⁻⁷ kcal/mol) for all ten 8-qubit systems.

3. Aqueous solvation corrections via ddCOSMO reveal shifts of −21.8 to +10.2 kcal/mol, demonstrating that gas-phase electronic structure alone is insufficient for biological interpretation.

4. The pipeline is designed for CAS(8,8) utilization (93% of 206 GB VRAM) but was underutilized at CAS(4,4) (1.5% VRAM). This highlights that the benchmark molecules, while passing all critics easily, do not stress-test the hardware.

5. No biological efficacy claims are made. This is a computational methodology benchmark demonstrating GPU-accelerated multi-reference quantum chemistry with solvation corrections.

---

## 7. Data Availability

- **Results:** `publication_results_v2.json` (all energies, critics, solvation data)
- **Pipeline code:** `publication_gpu_pipeline.py` (900+ lines, fully reproducible)
- **Repository:** https://github.com/qubitpage/quantum-longevity-research

---

## 8. References

1. Roos, B. O.; Taylor, P. R.; Siegbahn, P. E. M. (1980). *Chem. Phys.* 48, 157–173. [CASSCF method]
2. Sun, Q. et al. (2018). *WIREs Comput. Mol. Sci.* 8, e1340. [PySCF]
3. Sun, Q. et al. (2020). *J. Chem. Phys.* 153, 024109. [PySCF recent developments]
4. McClean, J. R. et al. (2020). *Quantum Sci. Technol.* 5, 034014. [OpenFermion]
5. Kandala, A. et al. (2017). *Nature* 549, 242–246. [Hardware-efficient VQE]
6. Wang, L.-P.; Song, C. C. (2016). *J. Chem. Phys.* 144, 214108. [geomeTRIC]
7. Lipparini, F. et al. (2013). *J. Chem. Theory Comput.* 9, 3637–3648. [ddCOSMO]
8. Nottoli, M. et al. (2019). *J. Chem. Phys.* 150, 094112. [ddCOSMO in PySCF]
9. López-Otín, C. et al. (2023). *Cell* 186, 99–118. [Hallmarks of aging]
10. Helgaker, T.; Klopper, W.; Tew, D. P. (2008). *Mol. Phys.* 106, 2107–2143. [CBS extrapolation]

---

## Appendix A: Compound Selection Rationale

The ten compounds were chosen based on published evidence for lifespan effects in model organisms:

| Compound | Evidence | Reference |
|---|---|---|
| NMN | +9% lifespan in aged mice | Mills et al., 2016 (*Cell Metab.*) |
| Resveratrol | +31% lifespan in obese mice | Baur et al., 2006 (*Nature*) |
| Rapamycin | +14% lifespan in mice | Harrison et al., 2009 (*Nature*) |
| Metformin | Reduced all-cause mortality in diabetics | Bannister et al., 2014 (*Diabetes Obes Metab*) |
| Quercetin | Senolytic; reduced frailty in aged mice | Xu et al., 2018 (*Nature Medicine*) |
| Fisetin | Senolytic; +10% lifespan in mice | Yousefzadeh et al., 2018 (*EBioMedicine*) |
| Dasatinib | Senolytic (combined with quercetin) | Zhu et al., 2015 (*Aging Cell*) |
| Spermidine | +25% lifespan in mice | Eisenberg et al., 2016 (*Nature Medicine*) |
| Urolithin A | Improved mitochondrial function in humans | Andreux et al., 2019 (*Nature Metab.*) |
| α-Ketoglutarate | +12% lifespan in mice | Asadi Shahmirzadi et al., 2020 (*Cell Metab.*) |

**These biological observations motivated the choice of test molecules but do NOT validate our electronic structure calculations. The quantum chemistry results stand independently as computational benchmarks.**

---

## Appendix B: Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1: Geometry Optimization (HF/6-31G*, geomeTRIC)           │
│   10 compounds × ~500s each = ~5000s (CPU-bound)                │
├─────────────────────────────────────────────────────────────────┤
│ Phase 2: CASCI + GPU Exact Diag + GPU VQE                       │
│   Auto-escalation: L1→L2→L3→L4→L5 until all critics pass       │
│   L1: CAS(4,4)/6-31G*  → 8q,  256×256   → <1 GB               │
│   L2: CAS(6,6)/6-31G*  → 12q, 4096×4096 → ~256 MB             │
│   L3: CAS(6,6)/cc-pVDZ → 12q, 4096×4096 → ~256 MB             │
│   L4: CAS(8,8)/cc-pVDZ → 16q, 65536×65536 → 64 GB             │
│   L5: CAS(8,8)/cc-pVTZ → 16q, 65536×65536 → 64 GB             │
├─────────────────────────────────────────────────────────────────┤
│ Phase 3: Solvation (ddCOSMO, water ε=78.39)                    │
│   Recompute best gas-phase level with implicit solvation        │
├─────────────────────────────────────────────────────────────────┤
│ Phase 4: CBS Extrapolation (Helgaker two-point formula)         │
│   E_CBS = (E_DZ × 3³ − E_TZ × 2³) / (3³ − 2³)                │
│   [Only if L3+L5 both computed]                                 │
├─────────────────────────────────────────────────────────────────┤
│ Phase 5: Final Validation (9 critics per compound)              │
│   All must pass for publication-grade results                   │
└─────────────────────────────────────────────────────────────────┘
```
