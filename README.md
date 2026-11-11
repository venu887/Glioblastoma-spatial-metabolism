# Glioblastoma-spatial-metabolism

Code repository for the 10x Genomics Visium spatial transcriptomics analysis supporting the manuscript:

> **"Integrated multi-omic analysis implicates coupled *de novo* pyrimidine and folate-mediated one-carbon metabolism in glioblastoma"**  
> *Kambati Niharika, Krishnasalini Gunanathan, Venugopala Reddy Mekala, Badi Shyam Sundar, Pemasang Pakhrin, Sai Sanwid Pradhan, Datta Darshan V.M, Utkarsh Bele, Sumit Thakar, Joshy E V, N.K Venkataramana, D. Bhanu Prakash, Subbarao Kanchi, Venketesh Sivaramakrishnan, Anusha Elumalai, Saiswaroop Rajaratnam (2026).*

---

## Repository Contents

This repository contains only the code files and documentation:

* **`gleo.py`**: Analyzes the Primary Cohort (GBM-Space Atlas; 97 sections). Computes raw counts vs. $\log_1p(\text{CP10k})$ normalized co-expression, evaluates sequencing depth confounding, and calculates stoichiometric coupling in dual-positive spots.
* **`Zenodo_gleo.py`**: Analyzes Validation Cohort 1 (Ravi et al.; 24 sections). Evaluates tumor-specific metabolic induction comparing primary glioblastoma to non-tumor human cortex.
* **`analyze_ucsc.py`**: Analyzes Validation Cohort 2 (UCSC Atlas; 32 sections). Maps spot detection frequencies across 10 Ivy GAP histological anatomical structures.
* **`LICENSE`**: MIT open-source license.
* **`README.md`**: Project overview, cohort details, and execution instructions.

---

## Datasets Analyzed

The scripts analyze three publicly available 10x Genomics Visium cohorts comprising **153 sections and 535,386 spatial spots**:

| Cohort | Source / Accession | Slices ($n$) | Spots ($N$) | Focus |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Cohort** | **GBM-Space Atlas**<br>(De Jong et al., *Nature* 2026; `www.gbmspace.org`) | 97 | 343,799 | Within-site co-localization across 49 surgical sites; sequencing depth confounding vs. $\log_1p(\text{CP10k})$ co-expression |
| **Validation 1** | **Ravi et al. Cohort**<br>(*Cancer Cell* 2022; `doi:10.5061/dryad.h70rxwdmj`) | 24 | 75,673 | Tumor-specific induction (16 Glioblastoma vs. 8 Non-tumor cortex) |
| **Validation 2** | **UCSC Multiomic Atlas**<br>(Sonpatki et al., *Nat Commun* 2026; UCSC Cell Browser) | 32 | 115,914 | Histological mapping across 10 Ivy GAP anatomical structures |
| **Total** | — | **153** | **535,386** | — |

*Note: Raw spatial matrix files are hosted in their respective public repositories due to size limitations (>20 GB total) and are not stored in this GitHub repository.*

---

## Key Spatial Findings

* **Resolution of Sequencing Depth Artifact:** Raw UMI counts showed an artificial positive correlation ($r = 0.225$, $p < 1.0 \times 10^{-300}$) driven by spot-to-spot variations in cellularity and sequencing depth. Library-size normalization ($\log_1p(\text{CP10k})$) resolved this technical bias ($r = 0.071$).
* **Stoichiometric Coupling in Dual-Active Spots:** In spatial microenvironments actively transcribing both enzymes ($n = 34,033$ dual-positive spots; 9.9% of cohort), *DHODH* and *SHMT2* demonstrated strong stoichiometric co-regulation (**Pearson $r = 0.581$, $p < 1.0 \times 10^{-300}$**).
* **Spatial Conservation:** Positive co-expression was preserved across **46 of 49 independent biological surgical biopsy sites** (mean within-site $r = 0.156$).
* **Tumor Specificity:** Both enzymes were significantly induced in primary glioblastoma compared to non-tumor human cortex (**3.19-fold for *DHODH***, $p = 0.0015$; **4.28-fold for *SHMT2***, $p = 0.0003$).
* **Ivy GAP Niche Enrichment:** Detection peaked in the perinecrotic zone (PNZ) and hyperplastic blood vessels (HBV), with lowest detection in leading-edge white/grey matter and non-malignant stroma.

---

## Dependencies & Execution

### Required Packages
The scripts run on Python 3.10 and require standard scientific and single-cell libraries:
* `scanpy`
* `pandas`
* `numpy`
* `scipy`
* `matplotlib`
* `seaborn`

### Running the Scripts
Run each analysis script directly from your terminal:
```bash
# 1. Primary cohort analysis (GBM-Space Atlas)
python gleo.py

# 2. Validation cohort 1 (Ravi et al. cortex vs. tumor)
python Zenodo_gleo.py

# 3. Validation cohort 2 (UCSC Ivy GAP niches)
python analyze_ucsc.py




@article{niharika2026integrated,
  title={Integrated multi-omic analysis implicates coupled de novo pyrimidine and folate-mediated one-carbon metabolism in glioblastoma},
  author={Niharika, Kambati and Gunanathan, Krishnasalini and Mekala, Venugopala Reddy and Sundar, Badi Shyam and Pakhrin, Pemasang and Pradhan, Sai Sanwid and Darshan V.M, Datta and Bele, Utkarsh and Thakar, Sumit and E V, Joshy and Venkataramana, N.K and Prakash, D. Bhanu and Kanchi, Subbarao and Sivaramakrishnan, Venketesh and Elumalai, Anusha and Rajaratnam, Saiswaroop},
  journal={bioRxiv / Journal},
  year={2026}
}

