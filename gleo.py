# %%
import gc
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr
import seaborn as sns

# ==============================================================================
# 1. DIRECTORY CONFIGURATION & GLOBAL SETUP
# ==============================================================================
data_dir = "/Users/mekalav/Documents/BCM/Dropbox_system/Sai_Glio/data/anndata"
results_dir = "/Users/mekalav/Documents/BCM/Dropbox_system/Sai_Glio/results_data3_Gleo_Website"
os.makedirs(results_dir, exist_ok=True)

# Publication-grade vector PDF settings (embedded TrueType fonts)
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["figure.dpi"] = 300

all_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".h5ad")])
print(
    "================================================================================"
)
print("STARTING SURGICAL SITE & PRIMARY NICHE SPATIAL PIPELINE")
print(f"Directory: {data_dir}")
print(
    f"Found {len(all_files)} Visium slices across 12 IDH-wt Glioblastoma"
    " patients"
)
print(
    "================================================================================\n"
)


# ==============================================================================
# 2. HELPER FUNCTION: PARSE BIOLOGICAL SURGICAL SITES
# ==============================================================================
def get_surgical_site_id(fname):
  """Consolidates technical cryosections (e.g. 1_1 and 1_2) into their

  underlying biological surgical sampling site (~1.25 cm3 biopsy block).
  """
  m_at15 = re.match(r"(AT15).*?-FO-([A-G]\d+)", fname)
  if m_at15:
    return m_at15.group(1), f"{m_at15.group(1)}_Site_{m_at15.group(2)}"

  m_std = re.match(r"(AT\d+).*?-FO-(\d+)", fname)
  if m_std:
    return m_std.group(1), f"{m_std.group(1)}_Site_{m_std.group(2)}"

  p_id = fname.split("-")[0]
  return p_id, f"{p_id}_Site_Unknown"


# ==============================================================================
# 3. STREAMING EXTRACTION & AGGREGATION ACROSS SURGICAL SITES
# ==============================================================================
all_spots_master = []

for idx, fname in enumerate(all_files, 1):
  fpath = os.path.join(data_dir, fname)
  patient_id, site_id = get_surgical_site_id(fname)

  print(
      f"[{idx:02d}/{len(all_files)}] Streaming {fname} -> {site_id}...",
      flush=True,
  )

  # Load Visium AnnData slice
  adata = sc.read_h5ad(fpath)
  adata.var_names_make_unique()

  # Features of interest
  target_genes = ["DHODH", "SHMT2"]
  niche_candidates = [
      "Proliferative",  # Malignant cycling core
      "Vasculature",  # Microvascular proliferation / endothelial
      "Hypoxic 1 (niche)",  # Perinecrotic / hypoxic niche
      "Ambiguous (TAMs)",  # Tumor-associated macrophages & microglia
      "Dev-like (AC)",  # Developmental astrocyte progenitor state
      "Dev-like (Neuron)",  # Developmental neuronal progenitor state
      "White matter",  # Infiltrated / non-malignant brain stroma
      "Grey matter",  # Infiltrated / non-malignant brain stroma
  ]

  avail_genes = [g for g in target_genes if g in adata.var_names]
  avail_niches = [n for n in niche_candidates if n in adata.var_names]

  # Spot-level pull
  spot_df = sc.get.obs_df(adata, keys=avail_genes + avail_niches).copy()

  # Fill any missing niche column with 0.0 for consistent alignment
  for n in niche_candidates:
    if n not in spot_df.columns:
      spot_df[n] = 0.0

  # Classify spots into the 6 Primary Microenvironments
  spot_df["Dev_like_Tumor"] = (
      spot_df["Dev-like (AC)"] + spot_df["Dev-like (Neuron)"]
  )
  spot_df["Brain_Stroma"] = spot_df["White matter"] + spot_df["Grey matter"]

  primary_niches = {
      "Proliferative Tumor Core": spot_df["Proliferative"],
      "Vasculature / Perivascular": spot_df["Vasculature"],
      "Hypoxic Tumor Niche": spot_df["Hypoxic 1 (niche)"],
      "Developmental Progenitor": spot_df["Dev_like_Tumor"],
      "Immune / TAM Infiltrate": spot_df["Ambiguous (TAMs)"],
      "Non-Malignant Brain Stroma": spot_df["Brain_Stroma"],
  }

  niche_matrix = pd.DataFrame(primary_niches, index=spot_df.index)
  spot_df["Primary_Niche"] = niche_matrix.idxmax(axis=1)

  # Handle spots with zero signal across all markers
  zero_mask = niche_matrix.max(axis=1) == 0
  spot_df.loc[zero_mask, "Primary_Niche"] = "Non-Malignant Brain Stroma"

  # Attach biological site and patient identifiers
  spot_df["Patient_ID"] = patient_id
  spot_df["Surgical_Site"] = site_id
  spot_df["Slide_Filename"] = fname

  # Keep only essential columns to preserve memory
  retained_cols = [
      "Patient_ID",
      "Surgical_Site",
      "DHODH",
      "SHMT2",
      "Primary_Niche",
  ]
  all_spots_master.append(spot_df[retained_cols])

  # Clean memory
  del adata, spot_df
  gc.collect()

# Consolidate all spot transcriptomes into unified matrix
print("\nConsolidating spot matrix across all surgical sites...", flush=True)
cohort_df = pd.concat(all_spots_master, ignore_index=True)
total_spots = len(cohort_df)
unique_sites = sorted(cohort_df["Surgical_Site"].unique())
print(f"--> Total Spots Collected: {total_spots:,}")
print(
    "--> Total Unique Surgical Sites Profiled on Visium:"
    f" {len(unique_sites)}\n"
)

# ==============================================================================
# 4. STEP 1: EVALUATE BASELINE METRICS PER SURGICAL SITE
# ==============================================================================
print(
    "================================================================================"
)
print("COMPUTING STEP 1: BASELINE METRICS FOR EACH SURGICAL SITE")
print(
    "================================================================================"
)

site_records = []
for s_name in unique_sites:
  s_sub = cohort_df[cohort_df["Surgical_Site"] == s_name]
  s_count = len(s_sub)
  p_id = s_sub["Patient_ID"].iloc[0]

  d_mean = s_sub["DHODH"].mean()
  d_sem = s_sub["DHODH"].sem()
  d_pct = (s_sub["DHODH"] > 0).mean() * 100

  s_mean = s_sub["SHMT2"].mean()
  s_sem = s_sub["SHMT2"].sem()
  s_pct = (s_sub["SHMT2"] > 0).mean() * 100

  dual_pct = ((s_sub["DHODH"] > 0) & (s_sub["SHMT2"] > 0)).mean() * 100

  r_val, p_val = (
      pearsonr(s_sub["DHODH"], s_sub["SHMT2"])
      if s_count > 10
      else (np.nan, np.nan)
  )

  site_records.append({
      "Patient_ID": p_id,
      "Surgical_Site_ID": s_name,
      "Spots_Collected_n": s_count,
      "DHODH_Mean": d_mean,
      "DHODH_SEM": d_sem,
      "DHODH_Pct_Active": d_pct,
      "SHMT2_Mean": s_mean,
      "SHMT2_SEM": s_sem,
      "SHMT2_Pct_Active": s_pct,
      "Dual_Positive_Pct": dual_pct,
      "Pearson_r": r_val,
      "p_value": p_val,
  })

df_sites_summary = pd.DataFrame(site_records).sort_values(
    "Pearson_r", ascending=True
)

# Save Master Surgical Sites Table
sites_csv_path = os.path.join(
    results_dir, "Table_S_Surgical_Sites_DHODH_SHMT2_Summary.csv"
)
df_sites_summary.to_csv(sites_csv_path, index=False)
print(f"Saved Step 1 Summary Table to: {sites_csv_path}")

# Figure 1: Ranked Lollipop Plot Across All Surgical Sites (PDF)
fig, ax = plt.subplots(figsize=(9, 13), dpi=300)
y_coords = np.arange(len(df_sites_summary))

ax.hlines(
    y=y_coords,
    xmin=0,
    xmax=df_sites_summary["Pearson_r"],
    color="#CFD8DC",
    linewidth=1.2,
)
ax.scatter(
    df_sites_summary["Pearson_r"], y_coords, color="#D95F02", s=50, zorder=3
)
ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)

mean_cohort_r = df_sites_summary["Pearson_r"].mean()
ax.axvline(
    mean_cohort_r,
    color="#B71C1C",
    linestyle=":",
    linewidth=1.5,
    label=f"Cohort Mean r = {mean_cohort_r:.3f}",
)

# Format Y-axis labels with Site ID and exact spot count n
y_labels = [
    f"{row['Surgical_Site_ID']}  (n = {row['Spots_Collected_n']:,})"
    for _, row in df_sites_summary.iterrows()
]
ax.set_yticks(y_coords)
ax.set_yticklabels(y_labels, fontsize=7.5, fontweight="bold")

ax.set_xlabel(
    "Pearson Correlation (r) between DHODH and SHMT2\n[Total Spatial Spots"
    f" Analyzed across Cohort: N = {total_spots:,}]",
    fontsize=10,
    fontweight="bold",
    labelpad=8,
)
ax.set_ylabel(
    "Surgical Sampling Sites [Site ID  (n = Spots Collected per"
    f" Site)]\n[Total Biological Sites Evaluated: N = {len(df_sites_summary)}]",
    fontsize=10,
    fontweight="bold",
    labelpad=8,
)
ax.set_title(
    "Step 1: Conservation of DHODH-SHMT2 Spatial Co-Localization Across"
    " Surgical Sites\n(Primary IDH-wt Glioblastoma Cohort: N ="
    f" {len(df_sites_summary)} Sites, Total Spots N = {total_spots:,})",
    fontsize=11,
    fontweight="bold",
    pad=12,
)

ax.legend(frameon=True, loc="lower right", fontsize=9)
sns.despine()
plt.tight_layout()

fig_step1_path = os.path.join(
    results_dir, "Fig_Step1_Surgical_Sites_Correlation_Lollipop_SpotCounts.pdf"
)
fig.savefig(fig_step1_path, format="pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved Step 1 Figure (PDF) to: {fig_step1_path}\n")

# ==============================================================================
# 5. STEP 2: EVALUATE PRIMARY MICROENVIRONMENTAL NICHES
# ==============================================================================
print(
    "================================================================================"
)
print("COMPUTING STEP 2: PRIMARY MICROENVIRONMENTAL NICHES ACROSS ALL SITES")
print(
    "================================================================================"
)

primary_niche_order = [
    "Proliferative Tumor Core",
    "Vasculature / Perivascular",
    "Hypoxic Tumor Niche",
    "Developmental Progenitor",
    "Immune / TAM Infiltrate",
    "Non-Malignant Brain Stroma",
]

niche_records = []
stroma_sub = cohort_df[cohort_df["Primary_Niche"] == "Non-Malignant Brain Stroma"]
stroma_dhodh_mean = stroma_sub["DHODH"].mean()
stroma_shmt2_mean = stroma_sub["SHMT2"].mean()

for n_name in primary_niche_order:
  sub = cohort_df[cohort_df["Primary_Niche"] == n_name]
  count_n = len(sub)

  d_m = sub["DHODH"].mean()
  d_se = sub["DHODH"].sem()
  s_m = sub["SHMT2"].mean()
  s_se = sub["SHMT2"].sem()

  fold_dh = (d_m / stroma_dhodh_mean) if stroma_dhodh_mean > 0 else np.nan
  fold_sh = (s_m / stroma_shmt2_mean) if stroma_shmt2_mean > 0 else np.nan

  r_sub, p_sub = (
      pearsonr(sub["DHODH"], sub["SHMT2"]) if count_n > 10 else (np.nan, np.nan)
  )

  niche_records.append({
      "Microenvironmental_Niche": n_name,
      "Spots_Collected_n": count_n,
      "Pct_of_Total_Cohort": (count_n / total_spots) * 100,
      "DHODH_Mean": d_m,
      "DHODH_SEM": d_se,
      "DHODH_Fold_vs_Stroma": fold_dh,
      "SHMT2_Mean": s_m,
      "SHMT2_SEM": s_se,
      "SHMT2_Fold_vs_Stroma": fold_sh,
      "Pearson_r": r_sub,
      "p_value": p_sub,
  })

df_niches_summary = pd.DataFrame(niche_records)

# Save Primary Niches Summary Table
niches_csv_path = os.path.join(
    results_dir, "Table_Cohort_Primary_Niches_Summary.csv"
)
df_niches_summary.to_csv(niches_csv_path, index=False)
print(f"Saved Step 2 Primary Niches Table to: {niches_csv_path}")

# Figure 2: Grouped Bar Chart Across Primary Niches (PDF)
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

x_pos = np.arange(len(primary_niche_order))
b_width = 0.35

x_axis_labels = [
    f"{row['Microenvironmental_Niche']}\n(n = {row['Spots_Collected_n']:,}"
    " spots)"
    for _, row in df_niches_summary.iterrows()
]

rect1 = ax.bar(
    x_pos - b_width / 2,
    df_niches_summary["DHODH_Mean"],
    b_width,
    yerr=df_niches_summary["DHODH_SEM"],
    capsize=4,
    label="DHODH",
    color="#E66101",
    edgecolor="black",
    linewidth=0.8,
)

rect2 = ax.bar(
    x_pos + b_width / 2,
    df_niches_summary["SHMT2_Mean"],
    b_width,
    yerr=df_niches_summary["SHMT2_SEM"],
    capsize=4,
    label="SHMT2",
    color="#5E3C99",
    edgecolor="black",
    linewidth=0.8,
)

ax.set_xticks(x_pos)
ax.set_xticklabels(x_axis_labels, fontsize=9, fontweight="bold")
ax.set_xlabel(
    "Primary Glioblastoma Cellular Microenvironments [Spot Count n per Niche]",
    fontsize=11,
    fontweight="bold",
    labelpad=10,
)

ax.set_ylabel(
    "Mean Expression Counts per Spot (± SEM)\n[Total Cohort Spots Collected on"
    f" Y-Axis: N = {total_spots:,}]",
    fontsize=10,
    fontweight="bold",
    labelpad=10,
)

ax.set_title(
    "Step 2: Expression of DHODH and SHMT2 Across Primary Glioblastoma"
    f" Niches\n(Cohort: N = {len(df_sites_summary)} Surgical Sites, Total"
    f" Spatial Spots N = {total_spots:,})",
    fontsize=12,
    fontweight="bold",
    pad=12,
)

ax.legend(frameon=True, fontsize=10, loc="upper right")
sns.despine()
plt.tight_layout()

fig_step2_path = os.path.join(
    results_dir, "Fig_Step2_Primary_Niches_Expression_SpotCounts.pdf"
)
fig.savefig(fig_step2_path, format="pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved Step 2 Niche Bar Chart (PDF) to: {fig_step2_path}")

# Figure 3: Global 2D Hexbin Co-Localization (PDF)
print("Generating Global Hexbin Co-Localization Plot...", flush=True)
global_r, global_p = pearsonr(cohort_df["DHODH"], cohort_df["SHMT2"])

fig, ax = plt.subplots(figsize=(7.5, 6), dpi=300)
hb = ax.hexbin(
    cohort_df["DHODH"],
    cohort_df["SHMT2"],
    gridsize=35,
    cmap="YlOrRd",
    mincnt=1,
    bins="log",
)

cb = fig.colorbar(hb, ax=ax)
cb.set_label("Log10 (Spatial Spot Count)", fontsize=10, fontweight="bold")

ax.set_xlabel(
    "DHODH Expression (Counts per Spot)\n[X-Axis Total Spots Collected: n ="
    f" {total_spots:,}]",
    fontsize=10,
    fontweight="bold",
    labelpad=8,
)
ax.set_ylabel(
    "SHMT2 Expression (Counts per Spot)\n[Y-Axis Total Spots Collected: n ="
    f" {total_spots:,}]",
    fontsize=10,
    fontweight="bold",
    labelpad=8,
)

ax.set_title(
    f"Global Cohort Co-Localization: DHODH vs SHMT2\nPearson r = {global_r:.3f},"
    f" p < 1.0e-300 (Total N = {total_spots:,} Spots across"
    f" {len(df_sites_summary)} Sites)",
    fontsize=11,
    fontweight="bold",
    pad=12,
)

sns.regplot(
    data=cohort_df.sample(min(25000, len(cohort_df)), random_state=42),
    x="DHODH",
    y="SHMT2",
    scatter=False,
    ax=ax,
    color="blue",
    line_kws={
        "linewidth": 1.8,
        "linestyle": "--",
        "label": f"Regression Fit (r = {global_r:.3f})",
    },
)
ax.legend(frameon=True, loc="upper left")

sns.despine()
plt.tight_layout()

fig_hexbin_path = os.path.join(
    results_dir, "Fig_Cohort_Hexbin_Density_SpotCounts_BothAxes.pdf"
)
fig.savefig(fig_hexbin_path, format="pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved Global Hexbin Plot (PDF) to: {fig_hexbin_path}\n")

# ==============================================================================
# 6. PATIENT-LEVEL VISUALIZATION SETTINGS & PALETTES
# ==============================================================================
SPOT_SIZE = 450  # Bold, prominent spots across the tissue
ALPHA_SPOT = 1.0  # 100% opaque spots
ALPHA_IMG = 0.25  # Softens H&E background to 25% opacity
VMAX_PERCENTILE = 98  # Clips top 2% intensity for high dynamic range

niche_palette = {
    "Proliferative Tumor Core": "#E41A1C",  # Vibrant Red
    "Vasculature / Perivascular": "#008837",  # Deep Green
    "Hypoxic Tumor Niche": "#7B3294",  # Rich Purple
    "Developmental Progenitor": "#FF7F00",  # Bright Orange
    "Immune / TAM Infiltrate": "#1F78B4",  # Primary Blue
    "Non-Malignant Brain Stroma": "#90A4AE",  # Soft Slate Grey
}


def parse_site(fname):
  m_at15 = re.match(r"(AT15).*?-FO-([A-G]\d+)", fname)
  if m_at15:
    return m_at15.group(1), f"Site_{m_at15.group(2)}"
  m_std = re.match(r"(AT\d+).*?-FO-(\d+)", fname)
  if m_std:
    return m_std.group(1), f"Site_{m_std.group(2)}"
  p_id = fname.split("-")[0]
  return p_id, "Site_Unknown"


patient_dict = {}
for f in all_files:
  p_id, s_id = parse_site(f)
  if p_id not in patient_dict:
    patient_dict[p_id] = {}
  if s_id not in patient_dict[p_id]:
    patient_dict[p_id][s_id] = []
  patient_dict[p_id][s_id].append(f)

patient_order = sorted(
    patient_dict.keys(), key=lambda x: int(re.search(r"\d+", x).group())
)

print(
    "================================================================================"
)
print(f"Found {len(patient_order)} Patients across {len(all_files)} slices.")
print("Processing each patient: Spatial Grid + Proportion/Expression (PDF)...")
print(
    "================================================================================\n"
)

cohort_composition_records = []

# ==============================================================================
# 7. PATIENT-BY-PATIENT LOOP (12 PATIENTS)
# ==============================================================================
for p_idx, p_id in enumerate(patient_order, 1):
  sites_for_patient = sorted(patient_dict[p_id].keys())
  n_sites = len(sites_for_patient)

  print(
      "--------------------------------------------------------------------------------"
  )
  print(
      f"[{p_idx:02d}/12] Processing Patient {p_id} ({n_sites} Surgical Sites)..."
  )
  print(
      "--------------------------------------------------------------------------------"
  )

  # Select 1 representative section per surgical site
  selected_sections = []
  for s_id in sites_for_patient:
    file_candidates = patient_dict[p_id][s_id]
    chosen_file = file_candidates[0]
    for c in file_candidates:
      if "_1" in c or "S27" in c or "S34" in c:
        chosen_file = c
        break
    selected_sections.append({
        "site": s_id,
        "filename": chosen_file,
        "label": f"{p_id} {s_id}",
    })

  patient_niche_proportions = []
  patient_regional_expr = []

  # Setup figure canvas for spatial grid (n_sites rows x 3 columns)
  fig_spatial, axes_spatial = plt.subplots(
      n_sites, 3, figsize=(18, 5.2 * n_sites), dpi=300
  )
  if n_sites == 1:
    axes_spatial = np.expand_dims(axes_spatial, axis=0)

  for row_idx, sec in enumerate(selected_sections):
    fpath = os.path.join(data_dir, sec["filename"])
    print(f"   -> Streaming {sec['site']} ({sec['filename']})...")

    adata = sc.read_h5ad(fpath)
    adata.var_names_make_unique()
    n_spots = adata.n_obs

    # A. Normalization (CPM) & Log1p Transformation
    if adata.X.max() > 50 or np.all(
        np.mod(adata.X.data if hasattr(adata.X, "data") else adata.X, 1) == 0
    ):
      sc.pp.normalize_total(adata, target_sum=1e4)
      sc.pp.log1p(adata)

    # B. Primary Microenvironmental Niche Classification
    niche_candidates = [
        "Proliferative",
        "Vasculature",
        "Hypoxic 1 (niche)",
        "Ambiguous (TAMs)",
        "Dev-like (AC)",
        "Dev-like (Neuron)",
        "White matter",
        "Grey matter",
    ]
    avail_niches = [n for n in niche_candidates if n in adata.var_names]
    sub_df = sc.get.obs_df(adata, keys=avail_niches).copy()

    for n in niche_candidates:
      if n not in sub_df.columns:
        sub_df[n] = 0.0

    sub_df["Dev_like_Tumor"] = (
        sub_df["Dev-like (AC)"] + sub_df["Dev-like (Neuron)"]
    )
    sub_df["Brain_Stroma"] = sub_df["White matter"] + sub_df["Grey matter"]

    primary_dict = {
        "Proliferative Tumor Core": sub_df["Proliferative"],
        "Vasculature / Perivascular": sub_df["Vasculature"],
        "Hypoxic Tumor Niche": sub_df["Hypoxic 1 (niche)"],
        "Developmental Progenitor": sub_df["Dev_like_Tumor"],
        "Immune / TAM Infiltrate": sub_df["Ambiguous (TAMs)"],
        "Non-Malignant Brain Stroma": sub_df["Brain_Stroma"],
    }
    niche_matrix = pd.DataFrame(primary_dict, index=sub_df.index)
    adata.obs["Primary_Niche"] = niche_matrix.idxmax(axis=1)

    zero_mask = niche_matrix.max(axis=1) == 0
    adata.obs.loc[zero_mask, "Primary_Niche"] = "Non-Malignant Brain Stroma"
    adata.obs["Primary_Niche"] = pd.Categorical(
        adata.obs["Primary_Niche"], categories=primary_niche_order
    )

    # C. Collect Niche Spot Proportions for this Site
    niche_counts = adata.obs["Primary_Niche"].value_counts()
    for n_name in primary_niche_order:
      count = niche_counts.get(n_name, 0)
      prop = (count / n_spots) * 100.0
      patient_niche_proportions.append({
          "Surgical_Site": sec["site"],
          "Microenvironment": n_name,
          "Spots_n": count,
          "Proportion_Pct": prop,
      })
      cohort_composition_records.append({
          "Patient_ID": p_id,
          "Surgical_Site": f"{p_id}_{sec['site']}",
          "Microenvironment": n_name,
          "Spots_n": count,
          "Proportion_Pct": prop,
      })

    # D. Compute Expression per Microenvironment
    expr_df = sc.get.obs_df(adata, keys=["DHODH", "SHMT2"])
    r_val = expr_df["DHODH"].corr(expr_df["SHMT2"])

    combined_obs = expr_df.copy()
    combined_obs["Primary_Niche"] = adata.obs["Primary_Niche"]

    for n_name in primary_niche_order:
      n_sub = combined_obs[combined_obs["Primary_Niche"] == n_name]
      if len(n_sub) > 0:
        dh_sem = n_sub["DHODH"].sem()
        sh_sem = n_sub["SHMT2"].sem()
        patient_regional_expr.append({
            "Surgical_Site": sec["site"],
            "Microenvironment": n_name,
            "DHODH_Mean": n_sub["DHODH"].mean(),
            "DHODH_SEM": 0.0 if np.isnan(dh_sem) else dh_sem,
            "SHMT2_Mean": n_sub["SHMT2"].mean(),
            "SHMT2_SEM": 0.0 if np.isnan(sh_sem) else sh_sem,
        })

    # E. Saturated Color Scale Limits (Upper 98th Percentile)
    dh_vec = adata.obs_vector("DHODH")
    sh_vec = adata.obs_vector("SHMT2")
    dh_pos = dh_vec[dh_vec > 0]
    sh_pos = sh_vec[sh_vec > 0]
    vmax_dh = (
        float(np.percentile(dh_pos, VMAX_PERCENTILE))
        if len(dh_pos) > 0
        else 1.0
    )
    vmax_sh = (
        float(np.percentile(sh_pos, VMAX_PERCENTILE))
        if len(sh_pos) > 0
        else 1.0
    )

    # F. Plot Column 1: DHODH
    ax_dh = axes_spatial[row_idx, 0]
    sc.pl.spatial(
        adata,
        color="DHODH",
        cmap="magma",
        spot_size=SPOT_SIZE,
        alpha=ALPHA_SPOT,
        alpha_img=ALPHA_IMG,
        vmin=0.0,
        vmax=vmax_dh,
        ax=ax_dh,
        show=False,
        colorbar_loc="right",
    )
    ax_dh.set_title(
        f"{sec['label']}: Normalized DHODH\n(n = {n_spots:,} spots)",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )
    ax_dh.set_xlabel("")
    ax_dh.set_ylabel(
        f"{sec['site']}\nr = {r_val:.3f}",
        fontsize=11,
        fontweight="bold",
        labelpad=10,
    )

    # G. Plot Column 2: SHMT2
    ax_sh = axes_spatial[row_idx, 1]
    sc.pl.spatial(
        adata,
        color="SHMT2",
        cmap="viridis",
        spot_size=SPOT_SIZE,
        alpha=ALPHA_SPOT,
        alpha_img=ALPHA_IMG,
        vmin=0.0,
        vmax=vmax_sh,
        ax=ax_sh,
        show=False,
        colorbar_loc="right",
    )
    ax_sh.set_title(
        f"{sec['label']}: Normalized SHMT2\n(Pearson r = {r_val:.3f})",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )
    ax_sh.set_xlabel("")
    ax_sh.set_ylabel("")

    # H. Plot Column 3: Primary Cellular Niches
    ax_reg = axes_spatial[row_idx, 2]
    sc.pl.spatial(
        adata,
        color="Primary_Niche",
        palette=niche_palette,
        spot_size=SPOT_SIZE,
        alpha=ALPHA_SPOT,
        alpha_img=ALPHA_IMG,
        ax=ax_reg,
        show=False,
        legend_loc="right margin" if row_idx == 0 else "none",
    )
    ax_reg.set_title(
        f"{sec['label']}: Microenvironment Niches",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )
    ax_reg.set_xlabel("")
    ax_reg.set_ylabel("")

    del adata, sub_df, expr_df
    gc.collect()

  # Column Top Headers
  axes_spatial[0, 0].annotate(
      "LEFT: NORMALIZED DHODH",
      xy=(0.5, 1.25),
      xycoords="axes fraction",
      ha="center",
      fontsize=12.5,
      fontweight="bold",
      color="#D95F02",
  )
  axes_spatial[0, 1].annotate(
      "MIDDLE: NORMALIZED SHMT2",
      xy=(0.5, 1.25),
      xycoords="axes fraction",
      ha="center",
      fontsize=12.5,
      fontweight="bold",
      color="#5E3C99",
  )
  axes_spatial[0, 2].annotate(
      "RIGHT: CELLULAR NICHES",
      xy=(0.5, 1.25),
      xycoords="axes fraction",
      ha="center",
      fontsize=12.5,
      fontweight="bold",
      color="#008837",
  )

  plt.tight_layout()
  spatial_out_path = os.path.join(
      results_dir, f"Fig_{p_id}_All_Sites_Genes_vs_Regions_Grid_LargeSpots.pdf"
  )
  fig_spatial.savefig(spatial_out_path, format="pdf", bbox_inches="tight")
  plt.close(fig_spatial)
  print(f"   [+] Saved Spatial Grid (PDF): {spatial_out_path}")

  # ==========================================================================
  # 8. COMPANION PROPORTION & REGIONAL EXPRESSION FIGURE (PDF)
  # ==========================================================================
  df_pat_props = pd.DataFrame(patient_niche_proportions)
  df_pat_expr = pd.DataFrame(patient_regional_expr)

  fig_prop, (ax_p1, ax_p2) = plt.subplots(1, 2, figsize=(16, 5.5), dpi=300)

  # Panel A: Stacked Bar Chart of Niche Proportions per Site
  prop_pivot = df_pat_props.pivot_table(
      index="Surgical_Site",
      columns="Microenvironment",
      values="Proportion_Pct",
      aggfunc="mean",
  ).fillna(0)
  prop_pivot = prop_pivot.reindex(columns=primary_niche_order)

  prop_pivot.plot(
      kind="bar",
      stacked=True,
      color=[niche_palette[c] for c in primary_niche_order],
      ax=ax_p1,
      edgecolor="black",
      linewidth=0.7,
      width=0.55,
  )
  ax_p1.set_title(
      f"{p_id}: Microenvironmental Niche Composition (% Spots)",
      fontsize=11,
      fontweight="bold",
      pad=10,
  )
  ax_p1.set_xlabel("Surgical Sampling Site", fontsize=10, fontweight="bold")
  ax_p1.set_ylabel("Proportion of Spots (%)", fontsize=10, fontweight="bold")
  ax_p1.set_ylim(0, 100)
  ax_p1.set_xticklabels(
      ax_p1.get_xticklabels(), rotation=0, fontsize=9.5, fontweight="bold"
  )
  ax_p1.legend(
      bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, fontsize=8.5
  )
  sns.despine(ax=ax_p1)

  # Panel B: Expression of DHODH & SHMT2 across regions
  reg_stats = (
      df_pat_expr.groupby("Microenvironment")
      .agg(
          DHODH_Mean=("DHODH_Mean", "mean"),
          DHODH_SEM=(
              "DHODH_SEM",
              lambda s: np.nanmean(s) if not np.isnan(s).all() else 0.0,
          ),
          SHMT2_Mean=("SHMT2_Mean", "mean"),
          SHMT2_SEM=(
              "SHMT2_SEM",
              lambda s: np.nanmean(s) if not np.isnan(s).all() else 0.0,
          ),
      )
      .reindex(primary_niche_order)
      .dropna(subset=["DHODH_Mean", "SHMT2_Mean"])
  )

  reg_stats["DHODH_SEM"] = reg_stats["DHODH_SEM"].fillna(0.0)
  reg_stats["SHMT2_SEM"] = reg_stats["SHMT2_SEM"].fillna(0.0)

  x_pos = np.arange(len(reg_stats))
  b_w = 0.35

  ax_p2.bar(
      x_pos - b_w / 2,
      reg_stats["DHODH_Mean"],
      b_w,
      yerr=reg_stats["DHODH_SEM"],
      capsize=4,
      label="DHODH",
      color="#E66101",
      edgecolor="black",
      linewidth=0.8,
  )
  ax_p2.bar(
      x_pos + b_w / 2,
      reg_stats["SHMT2_Mean"],
      b_w,
      yerr=reg_stats["SHMT2_SEM"],
      capsize=4,
      label="SHMT2",
      color="#5E3C99",
      edgecolor="black",
      linewidth=0.8,
  )

  short_labels = [
      n.replace("Tumor Core", "Core")
      .replace("Vasculature / Perivascular", "Vasculature")
      .replace("Non-Malignant Brain Stroma", "Brain Stroma")
      for n in reg_stats.index
  ]
  ax_p2.set_xticks(x_pos)
  ax_p2.set_xticklabels(
      short_labels, rotation=25, ha="right", fontsize=9, fontweight="bold"
  )
  ax_p2.set_title(
      f"{p_id}: Mean Normalized Expression Across Microenvironments",
      fontsize=11,
      fontweight="bold",
      pad=10,
  )
  ax_p2.set_xlabel("Primary Microenvironment", fontsize=10, fontweight="bold")
  ax_p2.set_ylabel(
      "Mean Normalized Expression (log CPM ± SEM)",
      fontsize=10,
      fontweight="bold",
  )
  ax_p2.legend(frameon=True, loc="upper right", fontsize=9)
  sns.despine(ax=ax_p2)

  plt.tight_layout()
  prop_out_path = os.path.join(
      results_dir, f"Fig_{p_id}_Niche_Proportions_and_Regional_Expression.pdf"
  )
  fig_prop.savefig(prop_out_path, format="pdf", bbox_inches="tight")
  plt.close(fig_prop)
  print(f"   [+] Saved Proportion/Expression Figure (PDF): {prop_out_path}\n")

# ==============================================================================
# 9. MASTER COHORT-WIDE NICHE PROPORTION OVERVIEW (ALL 49 SITES) (PDF)
# ==============================================================================
print(
    "================================================================================"
)
print(
    "Generating Master Cohort-Wide Niche Proportion Overview across all 49"
    " Sites (PDF)..."
)
print(
    "================================================================================"
)

df_all_cohort_props = pd.DataFrame(cohort_composition_records)
cohort_pivot = df_all_cohort_props.pivot_table(
    index="Surgical_Site",
    columns="Microenvironment",
    values="Proportion_Pct",
    aggfunc="mean",
).fillna(0)
cohort_pivot = cohort_pivot.reindex(columns=primary_niche_order)

fig_cohort, ax_c = plt.subplots(figsize=(24, 6.5), dpi=300)
cohort_pivot.plot(
    kind="bar",
    stacked=True,
    color=[niche_palette[c] for c in primary_niche_order],
    ax=ax_c,
    edgecolor="black",
    linewidth=0.5,
    width=0.75,
)
ax_c.set_title(
    "Microenvironmental Niche Proportions Across All 49 Surgical Sampling Sites"
    " (12 IDH-wt Patients, N = 343,799 Spots)",
    fontsize=13,
    fontweight="bold",
    pad=12,
)
ax_c.set_xlabel(
    "Biological Surgical Sampling Sites (49 Sites)",
    fontsize=11,
    fontweight="bold",
)
ax_c.set_ylabel("Niche Composition (% of Spots)", fontsize=11, fontweight="bold")
ax_c.set_ylim(0, 100)
ax_c.set_xticklabels(
    ax_c.get_xticklabels(), rotation=90, fontsize=8, fontweight="bold"
)
ax_c.legend(
    bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True, fontsize=10
)
sns.despine(ax=ax_c)
plt.tight_layout()

cohort_prop_path = os.path.join(
    results_dir, "Fig_Cohort_All_49_Sites_Niche_Proportions_Master.pdf"
)
fig_cohort.savefig(cohort_prop_path, format="pdf", bbox_inches="tight")
plt.close(fig_cohort)
print(f"--> Saved Master Cohort Niche Overview (PDF) to: {cohort_prop_path}")

print(
    "\n================================================================================"
)
print("ALL 12 PATIENTS COMPLETED SUCCESSFULLY! ALL PDF FIGURES STORED IN:")
print(f"{results_dir}")
print(
    "================================================================================"
)







# %%

# %%
import gc
import os
import re
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr, spearmanr
import seaborn as sns

# ==============================================================================
# 1. DIRECTORY CONFIGURATION & GLOBAL SETUP
# ==============================================================================
data_dir = "/Users/mekalav/Documents/BCM/Dropbox_system/Sai_Glio/data/anndata"
results_dir = (
    "/Users/mekalav/Documents/BCM/Dropbox_system/Sai_Glio/results_data3_revised"
)
os.makedirs(results_dir, exist_ok=True)

# Publication-grade vector PDF settings
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["figure.dpi"] = 300

all_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".h5ad")])
print(
    f"Found {len(all_files)} Visium slices. Starting single-pass extraction..."
)


def parse_metadata(fname):
  """Extracts Patient ID, Biological Site ID, and Slide Name."""
  m_at15 = re.match(r"(AT15).*?-FO-([A-G]\d+)", fname)
  if m_at15:
    p_id = m_at15.group(1)
    s_id = f"{p_id}_Site_{m_at15.group(2)}"
  else:
    m_std = re.match(r"(AT\d+).*?-FO-(\d+)", fname)
    if m_std:
      p_id = m_std.group(1)
      s_id = f"{p_id}_Site_{m_std.group(2)}"
    else:
      p_id = fname.split("-")[0]
      s_id = f"{p_id}_Site_Unknown"
  slide_id = os.path.splitext(fname)[0]
  return p_id, s_id, slide_id


# ==============================================================================
# 2. SINGLE-PASS EXTRACTION (RAW COUNTS + NORMALIZED CP10k + SLIDE METRICS)
# ==============================================================================
slide_stats_list = []
all_spots_raw = []
all_spots_norm = []

for idx, fname in enumerate(all_files, 1):
  fpath = os.path.join(data_dir, fname)
  patient_id, site_id, slide_id = parse_metadata(fname)

  adata = sc.read_h5ad(fpath)
  adata.var_names_make_unique()

  if "DHODH" not in adata.var_names or "SHMT2" not in adata.var_names:
    print(f"[{idx:02d}/{len(all_files)}] Skipping {fname}: Target genes absent.")
    del adata
    continue

  n_spots = adata.n_obs

  # A. Extract Raw Counts
  raw_df = sc.get.obs_df(adata, keys=["DHODH", "SHMT2"]).copy()
  raw_df["Patient_ID"] = patient_id
  raw_df["Surgical_Site"] = site_id
  raw_df["Slide_ID"] = slide_id
  all_spots_raw.append(raw_df)

  # B. Library Normalization (CP10k) & Log1p
  sc.pp.normalize_total(adata, target_sum=1e4)
  sc.pp.log1p(adata)
  norm_df = sc.get.obs_df(adata, keys=["DHODH", "SHMT2"]).copy()
  norm_df["Patient_ID"] = patient_id
  norm_df["Surgical_Site"] = site_id
  norm_df["Slide_ID"] = slide_id
  all_spots_norm.append(norm_df)

  # C. Slide-Level Statistics
  d_norm = norm_df["DHODH"].values
  s_norm = norm_df["SHMT2"].values
  d_raw = raw_df["DHODH"].values
  s_raw = raw_df["SHMT2"].values

  r_norm_slide, p_norm_slide = (
      pearsonr(d_norm, s_norm) if n_spots > 3 else (np.nan, np.nan)
  )
  rho_norm_slide, _ = (
      spearmanr(d_norm, s_norm) if n_spots > 3 else (np.nan, np.nan)
  )
  r_raw_slide, _ = pearsonr(d_raw, s_raw) if n_spots > 3 else (np.nan, np.nan)

  # Fisher z-transformation for 95% Confidence Interval on Pearson r
  if n_spots > 3 and not np.isnan(r_norm_slide) and abs(r_norm_slide) < 1.0:
    z = np.arctanh(r_norm_slide)
    se_z = 1.0 / np.sqrt(n_spots - 3)
    ci_low = np.tanh(z - 1.96 * se_z)
    ci_high = np.tanh(z + 1.96 * se_z)
  else:
    ci_low, ci_high = np.nan, np.nan

  dual_pos_count = np.sum((d_norm > 0) & (s_norm > 0))
  dual_pos_pct = (dual_pos_count / n_spots) * 100.0 if n_spots > 0 else 0.0

  slide_stats_list.append({
      "Slide_Filename": fname,
      "Slide_ID": slide_id,
      "Patient_ID": patient_id,
      "Surgical_Site": site_id,
      "Spot_Count": n_spots,
      "Pearson_r_Norm": r_norm_slide,
      "Pearson_p_Norm": p_norm_slide,
      "CI_95_Low": ci_low,
      "CI_95_High": ci_high,
      "Spearman_rho_Norm": rho_norm_slide,
      "Pearson_r_Raw": r_raw_slide,
      "Dual_Pos_Count": dual_pos_count,
      "Dual_Pos_Pct": dual_pos_pct,
      "DHODH_Mean_Norm": np.mean(d_norm),
      "SHMT2_Mean_Norm": np.mean(s_norm),
  })

  del adata, raw_df, norm_df
  gc.collect()

# Consolidate cohort matrices
cohort_raw_df = pd.concat(all_spots_raw, ignore_index=True)
cohort_norm_df = pd.concat(all_spots_norm, ignore_index=True)
df_slides = pd.DataFrame(slide_stats_list)

# Cohort-level summary metrics
total_spots_n = len(cohort_raw_df)
total_sites_n = df_slides["Surgical_Site"].nunique()
total_patients_n = df_slides["Patient_ID"].nunique()
total_slides_n = len(df_slides)

# Save Master Slide Metrics Table
stats_csv_path = os.path.join(
    results_dir, "Table_Master_Slide_Level_Correlation_Normalized.csv"
)
df_slides.to_csv(stats_csv_path, index=False)
print(f"\n--> Consolidated Metrics:")
print(f"    Total Spots:     {total_spots_n:,}")
print(f"    Surgical Sites:  {total_sites_n}")
print(f"    Patients:        {total_patients_n}")
print(f"    Visium Slices:   {total_slides_n}")
print(f"    Table saved to:  {stats_csv_path}\n")

# ==============================================================================
# 3. FIGURE 1: SIDE-BY-SIDE HEXBIN (RAW COUNTS VS. NORMALIZED EXPRESSION)
# ==============================================================================
# Global Cohort Statistics
r_raw_glob, p_raw_glob = pearsonr(
    cohort_raw_df["DHODH"], cohort_raw_df["SHMT2"]
)
rho_raw_glob, _ = spearmanr(cohort_raw_df["DHODH"], cohort_raw_df["SHMT2"])

r_norm_glob, p_norm_glob = pearsonr(
    cohort_norm_df["DHODH"], cohort_norm_df["SHMT2"]
)
rho_norm_glob, _ = spearmanr(cohort_norm_df["DHODH"], cohort_norm_df["SHMT2"])

# Dual-positive subset
dual_mask = (cohort_norm_df["DHODH"] > 0) & (cohort_norm_df["SHMT2"] > 0)
dual_norm_df = cohort_norm_df[dual_mask]
n_dual_spots = len(dual_norm_df)
pct_dual_spots = (n_dual_spots / total_spots_n) * 100.0

r_dual_glob, p_dual_glob = (
    pearsonr(dual_norm_df["DHODH"], dual_norm_df["SHMT2"])
    if n_dual_spots > 10
    else (np.nan, np.nan)
)
rho_dual_glob, _ = (
    spearmanr(dual_norm_df["DHODH"], dual_norm_df["SHMT2"])
    if n_dual_spots > 10
    else (np.nan, np.nan)
)

fig = plt.figure(figsize=(15.5, 6.6), dpi=300)
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.30)
plt.subplots_adjust(top=0.82, bottom=0.14, left=0.08, right=0.92)

# Global Cohort Header
fig.suptitle(
    "Spatial Co-Localization Dynamics: DHODH vs. SHMT2 across Glioblastoma"
    f" Cohort\n(N = {total_spots_n:,} Visium Spots | {total_sites_n} Surgical"
    f" Sites | {total_patients_n} IDH-wt Patients | {total_slides_n} Slices)",
    fontsize=12,
    fontweight="bold",
    y=0.97,
)

# ------------------------------------------------------------------------------
# PANEL A: RAW UMI COUNTS
# ------------------------------------------------------------------------------
ax1 = fig.add_subplot(gs[0, 0])
hb1 = ax1.hexbin(
    cohort_raw_df["DHODH"],
    cohort_raw_df["SHMT2"],
    gridsize=35,
    cmap="YlOrRd",
    mincnt=1,
    bins="log",
    linewidths=0.2,
    edgecolors="face",
)
sns.regplot(
    data=cohort_raw_df.sample(min(30000, total_spots_n), random_state=42),
    x="DHODH",
    y="SHMT2",
    scatter=False,
    ax=ax1,
    color="#08519C",
    line_kws={
        "linewidth": 2.0,
        "linestyle": "--",
        "label": f"Linear Fit (r = {r_raw_glob:.3f})",
    },
)

cb1 = fig.colorbar(hb1, ax=ax1, fraction=0.046, pad=0.04)
cb1.set_label("Log10(Spot Density)", fontsize=9.5, fontweight="bold")

p_raw_str = "< 1.0e-300" if p_raw_glob < 1e-300 else f"= {p_raw_glob:.2e}"
ax1.set_title(
    "A. Raw UMI Counts (Depth Confounded)\n"
    f"Pearson r = {r_raw_glob:.3f} (p {p_raw_str}) | Spearman ρ ="
    f" {rho_raw_glob:.3f}\n"
    f"[N = {total_spots_n:,} spots across {total_sites_n} sites]",
    fontsize=10,
    fontweight="bold",
    pad=10,
)
ax1.set_xlabel(
    f"Raw DHODH Counts per Spot [Total N = {total_spots_n:,}]",
    fontsize=9.5,
    fontweight="bold",
)
ax1.set_ylabel(
    f"Raw SHMT2 Counts per Spot [Total N = {total_spots_n:,}]",
    fontsize=9.5,
    fontweight="bold",
)
ax1.legend(loc="upper right", frameon=True, fontsize=8.5)
sns.despine(ax=ax1)

# ------------------------------------------------------------------------------
# PANEL B: LIBRARY-NORMALIZED EXPRESSION
# ------------------------------------------------------------------------------
ax2 = fig.add_subplot(gs[0, 1])
hb2 = ax2.hexbin(
    cohort_norm_df["DHODH"],
    cohort_norm_df["SHMT2"],
    gridsize=35,
    cmap="YlOrRd",
    mincnt=1,
    bins="log",
    linewidths=0.2,
    edgecolors="face",
)

# All spots fit (Blue dashed)
sns.regplot(
    data=cohort_norm_df.sample(min(30000, total_spots_n), random_state=42),
    x="DHODH",
    y="SHMT2",
    scatter=False,
    ax=ax2,
    color="#08519C",
    line_kws={
        "linewidth": 2.0,
        "linestyle": "--",
        "label": f"All Spots (r = {r_norm_glob:.3f}, N = {total_spots_n:,})",
    },
)

# Dual-positive fit (Green solid)
sns.regplot(
    data=dual_norm_df.sample(min(25000, n_dual_spots), random_state=42),
    x="DHODH",
    y="SHMT2",
    scatter=False,
    ax=ax2,
    color="#006D2C",
    line_kws={
        "linewidth": 2.2,
        "linestyle": "-",
        "label": (
            f"Dual-Positive Only (r = {r_dual_glob:.3f}, n = {n_dual_spots:,})"
        ),
    },
)

cb2 = fig.colorbar(hb2, ax=ax2, fraction=0.046, pad=0.04)
cb2.set_label("Log10(Spot Density)", fontsize=9.5, fontweight="bold")

p_norm_str = "< 1.0e-300" if p_norm_glob < 1e-300 else f"= {p_norm_glob:.2e}"
p_dual_str = "< 1.0e-300" if p_dual_glob < 1e-300 else f"= {p_dual_glob:.2e}"

ax2.set_title(
    "B. Normalized Expression [log1p(CP10k)]\n"
    f"All Spots: r = {r_norm_glob:.3f} (p {p_norm_str}) | ρ ="
    f" {rho_norm_glob:.3f}\n"
    f"Dual-Positive: r = {r_dual_glob:.3f} (p {p_dual_str}, n ="
    f" {n_dual_spots:,} [{pct_dual_spots:.1f}%])",
    fontsize=10,
    fontweight="bold",
    pad=10,
)
ax2.set_xlabel(
    f"Normalized DHODH [log1p(CP10k)] [Total N = {total_spots_n:,}]",
    fontsize=9.5,
    fontweight="bold",
)
ax2.set_ylabel(
    f"Normalized SHMT2 [log1p(CP10k)] [Total N = {total_spots_n:,}]",
    fontsize=9.5,
    fontweight="bold",
)
ax2.legend(loc="upper right", frameon=True, fontsize=8.5)
sns.despine(ax=ax2)

fig1_path = os.path.join(
    results_dir, "Fig1_Cohort_Correlation_Counts_vs_Normalized_Polished.pdf"
)
fig.savefig(fig1_path, format="pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved Fig 1 to: {fig1_path}")

# ==============================================================================
# 4. FIGURE 2: PATIENT-LEVEL DOT/STRIP PLOT WITH PATIENT MEANS
# ==============================================================================
patient_order = sorted(
    df_slides["Patient_ID"].unique(),
    key=lambda x: int(re.search(r"\d+", x).group())
    if re.search(r"\d+", x)
    else x,
)

fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=300)
mean_cohort_slide_r = df_slides["Pearson_r_Norm"].mean()

ax.axhline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
ax.axhline(
    mean_cohort_slide_r,
    color="#B71C1C",
    linestyle=":",
    linewidth=1.3,
    label=f"Cohort Mean Slide r = {mean_cohort_slide_r:.3f}",
)

sns.stripplot(
    data=df_slides,
    x="Patient_ID",
    y="Pearson_r_Norm",
    order=patient_order,
    size=7.5,
    jitter=0.22,
    palette="Set2",
    edgecolor="black",
    linewidth=0.6,
    alpha=0.9,
    ax=ax,
)

# Overlay Patient Means
patient_means = (
    df_slides.groupby("Patient_ID")["Pearson_r_Norm"]
    .mean()
    .reindex(patient_order)
    .values
)
patient_x = np.arange(len(patient_order))
ax.scatter(
    patient_x,
    patient_means,
    marker="D",
    color="black",
    s=50,
    zorder=5,
    label="Patient Mean r",
)

ax.set_title(
    "Slide-Level Normalized Spatial Co-Localization (DHODH vs. SHMT2) by"
    " Patient\n"
    f"(Cohort N = {total_spots_n:,} Spots across {total_slides_n} Slices and"
    f" {total_sites_n} Sites in {total_patients_n} Patients)",
    fontsize=11,
    fontweight="bold",
    pad=12,
)
ax.set_xlabel(
    f"Patient Identifier (N = {total_patients_n} Patients)",
    fontsize=10,
    fontweight="bold",
    labelpad=8,
)
ax.set_ylabel(
    "Pearson Correlation (r) [log1p(CP10k)]",
    fontsize=10,
    fontweight="bold",
    labelpad=8,
)
ax.set_ylim(-0.15, max(0.65, df_slides["Pearson_r_Norm"].max() + 0.1))
ax.legend(frameon=True, loc="upper right", fontsize=9)
sns.despine(trim=True)
plt.tight_layout()

fig2_path = os.path.join(
    results_dir, "Fig2_Slide_Correlation_by_Patient_Polished.pdf"
)
fig.savefig(fig2_path, format="pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved Fig 2 to: {fig2_path}")

# ==============================================================================
# 5. FIGURE 3: RANKED SLIDE FOREST PLOT (WITH 95% CIs & SPOT COUNTS)
# ==============================================================================
df_ranked = df_slides.sort_values("Pearson_r_Norm", ascending=True).reset_index(
    drop=True
)

fig_height = max(8.5, len(df_ranked) * 0.22)
fig, ax = plt.subplots(figsize=(8.5, fig_height), dpi=300)
y_vals = np.arange(len(df_ranked))

# 95% Confidence Intervals
ax.errorbar(
    x=df_ranked["Pearson_r_Norm"],
    y=y_vals,
    xerr=[
        df_ranked["Pearson_r_Norm"] - df_ranked["CI_95_Low"],
        df_ranked["CI_95_High"] - df_ranked["Pearson_r_Norm"],
    ],
    fmt="none",
    ecolor="#B0BEC5",
    elinewidth=1.0,
    capsize=2,
    zorder=2,
)

# Marker size proportional to spot count
sizes = 25 + 65 * (df_ranked["Spot_Count"] / df_ranked["Spot_Count"].max())
ax.scatter(
    df_ranked["Pearson_r_Norm"],
    y_vals,
    s=sizes,
    c=df_ranked["Pearson_r_Norm"],
    cmap="Spectral_r",
    edgecolor="black",
    linewidth=0.5,
    zorder=3,
)

ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
ax.axvline(
    mean_cohort_slide_r,
    color="#B71C1C",
    linestyle=":",
    linewidth=1.3,
    label=f"Cohort Mean Slide r = {mean_cohort_slide_r:.3f}",
)

# Y-Axis Labels: Patient + Slide + Spot Count n
y_labels = [
    f"{r['Patient_ID']} | {r['Slide_ID'][-14:]} (n={r['Spot_Count']:,})"
    for _, r in df_ranked.iterrows()
]
ax.set_yticks(y_vals)
ax.set_yticklabels(y_labels, fontsize=6.8)

ax.set_xlabel(
    f"Normalized Pearson Correlation (r) [95% CI]\n[Total Spots across All"
    f" Slices: N = {total_spots_n:,}]",
    fontsize=9.5,
    fontweight="bold",
    labelpad=8,
)
ax.set_ylabel(
    f"Individual Visium Slices [Sorted by r | Total N = {total_slides_n}"
    " Slices]",
    fontsize=9.5,
    fontweight="bold",
    labelpad=8,
)
ax.set_title(
    "Ranked Slide-Level DHODH-SHMT2 Correlation Across Glioblastoma Cohort\n"
    f"(N = {total_spots_n:,} Spots | {total_sites_n} Sites | {total_patients_n}"
    f" Patients | Cohort Mean r = {mean_cohort_slide_r:.3f})",
    fontsize=10.5,
    fontweight="bold",
    pad=12,
)
ax.legend(frameon=True, loc="lower right", fontsize=8.5)
sns.despine()
plt.tight_layout()

fig3_path = os.path.join(
    results_dir, "Fig3_Slide_Correlation_Forest_Ranked_Polished.pdf"
)
fig.savefig(fig3_path, format="pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved Fig 3 to: {fig3_path}")

print("\n===================================================================")
print("ALL FIGURES & TABLES COMPLETED SUCCESSFULLY!")
print(f"Directory: {results_dir}")
print("===================================================================")



# %%
# %%
import gc
import os
import re
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import linregress, pearsonr, spearmanr
import seaborn as sns

# ==============================================================================
# 1. DIRECTORY CONFIGURATION & GLOBAL SETUP
# ==============================================================================
data_dir = "/Users/mekalav/Documents/BCM/Dropbox_system/Sai_Glio/data/anndata"
results_dir = (
    "/Users/mekalav/Documents/BCM/Dropbox_system/Sai_Glio/results_data3_revised"
)
os.makedirs(results_dir, exist_ok=True)

# Publication-grade vector PDF settings
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["figure.dpi"] = 300

all_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".h5ad")])
print(
    f"Found {len(all_files)} Visium slices. Starting single-pass extraction..."
)


def parse_metadata(fname):
  """Consistently extracts Patient ID, Biological Site ID, and Slide Name."""
  m_at15 = re.match(r"(AT15).*?-FO-([A-G]\d+)", fname)
  if m_at15:
    p_id = m_at15.group(1)
    s_id = f"{p_id}_Site_{m_at15.group(2)}"
  else:
    m_std = re.match(r"(AT\d+).*?-FO-(\d+)", fname)
    if m_std:
      p_id = m_std.group(1)
      s_id = f"{p_id}_Site_{m_std.group(2)}"
    else:
      p_id = fname.split("-")[0]
      s_id = f"{p_id}_Site_Unknown"
  slide_id = os.path.splitext(fname)[0]
  return p_id, s_id, slide_id


# ==============================================================================
# 2. STREAMING EXTRACTION: RAW COUNTS & LIBRARY-NORMALIZED VALUES
# ==============================================================================
all_spots_raw = []
all_spots_norm = []
unique_sites = set()
unique_patients = set()
total_slides_count = 0

for idx, fname in enumerate(all_files, 1):
  fpath = os.path.join(data_dir, fname)
  patient_id, site_id, slide_id = parse_metadata(fname)

  print(
      f"[{idx:02d}/{len(all_files)}] Streaming: {slide_id} -> {site_id}...",
      flush=True,
  )

  adata = sc.read_h5ad(fpath)
  adata.var_names_make_unique()

  if "DHODH" not in adata.var_names or "SHMT2" not in adata.var_names:
    print(f"   [!] Skipping {fname}: Target gene(s) missing.")
    del adata
    gc.collect()
    continue

  total_slides_count += 1
  unique_sites.add(site_id)
  unique_patients.add(patient_id)

  # 1. Extract Raw Integer Counts
  raw_df = sc.get.obs_df(adata, keys=["DHODH", "SHMT2"]).copy()
  all_spots_raw.append(raw_df)

  # 2. Normalize to 10k CPM (CP10k) and log1p transform
  sc.pp.normalize_total(adata, target_sum=1e4)
  sc.pp.log1p(adata)
  norm_df = sc.get.obs_df(adata, keys=["DHODH", "SHMT2"]).copy()
  all_spots_norm.append(norm_df)

  del adata, raw_df, norm_df
  gc.collect()

# Consolidate cohort matrices across all Visium slices
print("\nConsolidating spot transcriptomes into cohort dataframes...", flush=True)
cohort_raw_df = pd.concat(all_spots_raw, ignore_index=True)
cohort_norm_df = pd.concat(all_spots_norm, ignore_index=True)

total_spots_n = len(cohort_raw_df)
total_sites_n = len(unique_sites)
total_patients_n = len(unique_patients)

# ==============================================================================
# 3. STATISTICAL EVALUATION ACROSS THE COHORT
# ==============================================================================
# A. Raw Counts Statistics
r_raw, p_raw = pearsonr(cohort_raw_df["DHODH"], cohort_raw_df["SHMT2"])
rho_raw, _ = spearmanr(cohort_raw_df["DHODH"], cohort_raw_df["SHMT2"])
slope_raw, inter_raw, _, _, _ = linregress(
    cohort_raw_df["DHODH"], cohort_raw_df["SHMT2"]
)

# B. All Spots Normalized Statistics
r_norm, p_norm = pearsonr(cohort_norm_df["DHODH"], cohort_norm_df["SHMT2"])
rho_norm, _ = spearmanr(cohort_norm_df["DHODH"], cohort_norm_df["SHMT2"])
slope_norm, inter_norm, _, _, _ = linregress(
    cohort_norm_df["DHODH"], cohort_norm_df["SHMT2"]
)

# C. Dual-Positive Only Normalized Statistics
dual_mask = (cohort_norm_df["DHODH"] > 0) & (cohort_norm_df["SHMT2"] > 0)
dual_norm_df = cohort_norm_df[dual_mask]
n_dual_spots = len(dual_norm_df)
pct_dual_spots = (n_dual_spots / total_spots_n) * 100.0

r_dual, p_dual = (
    pearsonr(dual_norm_df["DHODH"], dual_norm_df["SHMT2"])
    if n_dual_spots > 10
    else (np.nan, np.nan)
)
rho_dual, _ = (
    spearmanr(dual_norm_df["DHODH"], dual_norm_df["SHMT2"])
    if n_dual_spots > 10
    else (np.nan, np.nan)
)
slope_dual, inter_dual, _, _, _ = linregress(
    dual_norm_df["DHODH"], dual_norm_df["SHMT2"]
)

print(f"--> Cohort Size: {total_spots_n:,} spots across {total_sites_n} sites")
print(f"--> Raw Counts:   r = {r_raw:.3f} | rho = {rho_raw:.3f}")
print(f"--> Normalized:   r = {r_norm:.3f} | rho = {rho_norm:.3f}")
print(
    f"--> Dual-Active:  r = {r_dual:.3f} (n = {n_dual_spots:,}"
    f" [{pct_dual_spots:.1f}%])"
)

# ==============================================================================
# 4. FIGURE GENERATION: HIGH-RESOLUTION VECTOR PDF
# ==============================================================================
plt.close("all")
fig = plt.figure(figsize=(15.5, 6.8), dpi=300)

# Set GridSpec geometry to guarantee ample space for titles and colorbars
gs = gridspec.GridSpec(
    1, 2, figure=fig, top=0.73, bottom=0.12, left=0.07, right=0.91, wspace=0.32
)

# Supertitle (Cohort-wide metadata overview)
fig.suptitle(
    "Spatial Co-Localization Dynamics: DHODH vs. SHMT2 across Glioblastoma"
    f" Cohort\n(N = {total_spots_n:,} Visium Spots | {total_sites_n} Surgical"
    f" Sites | {total_patients_n} Patients | {total_slides_count} Slices)",
    fontsize=12.5,
    fontweight="bold",
    y=0.93,
)

# Colorbar frequency ticks (log-scale positions matched to count frequency)
cbar_ticks = [1, 10, 100, 1000, 10000, 100000]
cbar_labels = ["1", "10", "100", "1,000", "10,000", "100,000+"]

# ------------------------------------------------------------------------------
# PANEL A: RAW UMI COUNTS
# ------------------------------------------------------------------------------
ax1 = fig.add_subplot(gs[0, 0])
hb1 = ax1.hexbin(
    cohort_raw_df["DHODH"],
    cohort_raw_df["SHMT2"],
    gridsize=35,
    cmap="YlOrRd",
    mincnt=1,
    bins="log",
    linewidths=0.2,
    edgecolors="face",
)

# Linear Fit Line
x_vals_raw = np.linspace(0, 10, 100)
ax1.plot(
    x_vals_raw,
    inter_raw + slope_raw * x_vals_raw,
    color="#08519C",
    linestyle="--",
    linewidth=2.0,
    label=f"Linear Fit (r = {r_raw:.3f})",
)

# Colorbar (Spot count frequency)
cb1 = fig.colorbar(hb1, ax=ax1, fraction=0.046, pad=0.04)
cb1.set_label("Spatial Spot Count per Bin", fontsize=9.5, fontweight="bold")
cb1.locator = FixedLocator(cbar_ticks)
cb1.formatter = FixedFormatter(cbar_labels)
cb1.update_ticks()

p_raw_str = "< 1.0e-300" if p_raw < 1e-300 else f"= {p_raw:.2e}"
ax1.set_title(
    "A. Raw UMI Counts (Depth Confounded)\n"
    f"Pearson r = {r_raw:.3f} (p {p_raw_str}) | Spearman ρ = {rho_raw:.3f}\n"
    "[Discrete integer grid reflects technical depth bias]",
    fontsize=10.0,
    fontweight="bold",
    pad=10,
)
ax1.set_xlabel(
    f"Raw DHODH Counts per Spot (N = {total_spots_n:,})",
    fontsize=9.5,
    fontweight="bold",
)
ax1.set_ylabel(
    f"Raw SHMT2 Counts per Spot (N = {total_spots_n:,})",
    fontsize=9.5,
    fontweight="bold",
)
ax1.set_xlim(-0.5, 11)
ax1.set_ylim(-0.5, 32)
ax1.legend(loc="upper right", frameon=True, fontsize=8.5)
sns.despine(ax=ax1)

# ------------------------------------------------------------------------------
# PANEL B: LIBRARY-NORMALIZED EXPRESSION
# ------------------------------------------------------------------------------
ax2 = fig.add_subplot(gs[0, 1])
hb2 = ax2.hexbin(
    cohort_norm_df["DHODH"],
    cohort_norm_df["SHMT2"],
    gridsize=35,
    cmap="YlOrRd",
    mincnt=1,
    bins="log",
    linewidths=0.2,
    edgecolors="face",
)

# All Spots Fit (Blue dashed line)
x_vals_norm = np.linspace(0, 6, 100)
ax2.plot(
    x_vals_norm,
    inter_norm + slope_norm * x_vals_norm,
    color="#08519C",
    linestyle="--",
    linewidth=2.0,
    label=f"All Spots (r = {r_norm:.3f}, N = {total_spots_n:,})",
)

# Dual-Positive Only Fit (Green solid line)
x_vals_dual = np.linspace(
    dual_norm_df["DHODH"].min(), min(3.5, dual_norm_df["DHODH"].max()), 100
)
ax2.plot(
    x_vals_dual,
    inter_dual + slope_dual * x_vals_dual,
    color="#006D2C",
    linestyle="-",
    linewidth=2.2,
    label=(
        f"Dual-Positive Only (r = {r_dual:.3f}, n = {n_dual_spots:,}"
        f" [{pct_dual_spots:.1f}%])"
    ),
)

# Colorbar (Spot count frequency)
cb2 = fig.colorbar(hb2, ax=ax2, fraction=0.046, pad=0.04)
cb2.set_label("Spatial Spot Count per Bin", fontsize=9.5, fontweight="bold")
cb2.locator = FixedLocator(cbar_ticks)
cb2.formatter = FixedFormatter(cbar_labels)
cb2.update_ticks()

p_norm_str = "< 1.0e-300" if p_norm < 1e-300 else f"= {p_norm:.2e}"
p_dual_str = "< 1.0e-300" if p_dual < 1e-300 else f"= {p_dual:.2e}"

ax2.set_title(
    "B. Library-Normalized Expression [log1p(CP10k)]\n"
    f"All Spots: r = {r_norm:.3f} (p {p_norm_str}) | ρ = {rho_norm:.3f}\n"
    f"Dual-Positive: r = {r_dual:.3f} (p {p_dual_str}, n = {n_dual_spots:,}"
    f" [{pct_dual_spots:.1f}%])",
    fontsize=10.0,
    fontweight="bold",
    pad=10,
)
ax2.set_xlabel(
    f"Normalized DHODH [log1p(CP10k)] (N = {total_spots_n:,})",
    fontsize=9.5,
    fontweight="bold",
)
ax2.set_ylabel(
    f"Normalized SHMT2 [log1p(CP10k)] (N = {total_spots_n:,})",
    fontsize=9.5,
    fontweight="bold",
)
ax2.set_xlim(-0.2, 6.3)
ax2.set_ylim(-0.2, 6.3)
ax2.legend(loc="upper right", frameon=True, fontsize=8.5)
sns.despine(ax=ax2)

# Save Master Vector Figure
fig_out_path = os.path.join(
    results_dir, "Fig1_Cohort_Correlation_Counts_vs_Normalized_FinalClean.pdf"
)
fig.savefig(fig_out_path, format="pdf", bbox_inches="tight")
plt.close(fig)

print(f"\n===================================================================")
print(f"SUCCESSFULLY GENERATED PUBLICATION FIGURE:")
print(f"--> {fig_out_path}")
print(f"===================================================================")
# %%