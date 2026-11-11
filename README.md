# Glioblastoma-spatial-metabolism: Spatially Resolved Stoichiometric Coupling of Pyrimidine and One-Carbon Metabolism in Glioblastoma

[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Technology: 10x Visium](https://img.shields.io/badge/Platform-10x_Genomics_Visium-FF5722.svg)](https://www.10xgenomics.com/products/spatial-gene-expression)
[![Scanpy](https://img.shields.io/badge/Scanpy-v1.10-008080.svg)](https://scanpy.readthedocs.io/)
[![AnnData](https://img.shields.io/badge/AnnData-v0.9+-F37626.svg)](https://anndata.readthedocs.io/)

Official spatial transcriptomics repository supporting the manuscript:  
> **"Integrated multi-omic analysis implicates coupled *de novo* pyrimidine and folate-mediated one-carbon metabolism in glioblastoma"**  
> *Kambati Niharika, Krishnasalini Gunanathan, Venugopala Reddy Mekala, Badi Shyam Sundar, Pemasang Pakhrin, Sai Sanwid Pradhan, Datta Darshan V.M, Utkarsh Bele, Sumit Thakar, Joshy E V, N.K Venkataramana, D. Bhanu Prakash, Subbarao Kanchi, Venketesh Sivaramakrishnan, Anusha Elumalai, Saiswaroop Rajaratnam (2026).*

---

## Table of Contents
1. [Scientific Background & Rationale](#-scientific-background--rationale)
2. [Cohort Architecture & Data Provenance](#-cohort-architecture--data-provenance)
3. [Methodological Highlights & Mathematical Definitions](#-methodological-highlights--mathematical-definitions)
4. [Key Biological & Spatial Findings](#-key-biological--spatial-findings)
5. [Repository Structure & File Inventory](#-repository-structure--file-inventory)
6. [Hardware Requirements & Computational Footprint](#-hardware-requirements--computational-footprint)
7. [Environment Installation & Setup](#-environment-installation--setup)
8. [Data Acquisition & Directory Staging](#-data-acquisition--directory-staging)
9. [Step-by-Step Pipeline Execution](#-step-by-step-pipeline-execution)
10. [Detailed Script Specifications](#-detailed-script-specifications)
11. [Publication Deliverables](#-publication-deliverables)
12. [Authors & Affiliations](#-authors--affiliations)
13. [Citation & License](#-citation--license)

---

## 🔬 Scientific Background & Rationale

Glioblastoma (GBM; CNS WHO grade 4, IDH-wildtype) represents the most aggressive primary intrinsic central nervous system malignancy, characterized by extensive cellular heterogeneity, infiltrative margins, and therapy resistance against maximal surgical resection, radiotherapy, and temozolomide (TMZ). Proliferating tumor cells require continuous nucleotide pools for genomic replication and DNA damage repair (DDR).

Nucleotide synthesis involves two biochemical arms that converge at thymidylate synthesis:
1. **The *De Novo* Pyrimidine Pathway:** Driven by **Dihydroorotate Dehydrogenase (`DHODH`)**, an inner mitochondrial membrane enzyme that oxidizes dihydroorotate to orotate, linking pyrimidine assembly to the mitochondrial ubiquinone pool and electron transport chain.
2. **The Mitochondrial One-Carbon Folate Cycle:** Driven by **Serine Hydroxymethyltransferase 2 (`SHMT2`)**, which transfers one-carbon units from serine to tetrahydrofolate (THF), yielding glycine and 5,10-methylene-THF to supply one-carbon units for thymidylate (*TYMS*) and purine synthesis.

Bulk transcriptional profiling often cannot determine whether co-elevated metabolic genes reflect true intracellular co-regulation or simply variations in tumor purity. This repository provides a scalable, single-pass spatial transcriptomics pipeline across **153 10x Genomics Visium tissue sections (535,386 spatial spots)**, demonstrating that *DHODH* and *SHMT2* exhibit tumor-restricted induction and within-niche stoichiometric co-regulation.

---

## 📊 Cohort Architecture & Data Provenance

The spatial analyses integrate three multi-center clinical cohorts:
