
#%%
import gc
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

# ==============================================================================
# 1. SETUP PATHS & VECTOR PDF SETTINGS
# ==============================================================================
base_dir = "/Users/mekalav/Documents/BCM/Dropbox_system/Sai_Glio"
data2_dir = os.path.join(base_dir, "data/10XVisium2")
results2_dir = os.path.join(base_dir, "results_data1")
os.makedirs(results2_dir, exist_ok=True)

# Strict vector PDF configuration with embedded TrueType Type 42 fonts
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["figure.dpi"] = 300

# Tunable dot size parameters (decrease or increase as desired)
STRIP_DOT_SIZE = 6.5   # Marker size for jittered dots in Box Plots (Panels A & B)
RANKED_DOT_SIZE = 45   # Marker size for ranked dots in Dot Plots (Panels C & D)

# Candidate paths for the summary metrics CSV
candidate_csvs = [
    os.path.join(
        results2_dir, "Table_All_24_Samples_DHODH_SHMT2_Correlation.csv"
    ),
    os.path.join(
        results2_dir, "Table_Data2_All_24_Cortex_and_Tumor_Summary.csv"
    ),
    os.path.join(base_dir, "Table_All_24_Samples_DHODH_SHMT2_Correlation.csv"),
    os.path.join(base_dir, "Table_Data2_All_24_Cortex_and_Tumor_Summary.csv"),
    "Table_All_24_Samples_DHODH_SHMT2_Correlation.csv",
    "Table_Data2_All_24_Cortex_and_Tumor_Summary.csv",
]

csv_path = None
for path in candidate_csvs:
  if os.path.exists(path):
    csv_path = path
    break

# ==============================================================================
# 2. AUTO-RECOVERY: EXTRACT COUNTS DIRECTLY IF CSV IS MISSING
# ==============================================================================
if csv_path is None:
  print("Summary CSV not detected on disk. Extracting metrics directly from:")
  print(f"  -> {data2_dir}")

  import scanpy as sc

  all_folders = sorted([d for d in os.listdir(data2_dir) if d.startswith("#UKF")])
  target_samples = [
      d
      for d in all_folders
      if ("_C_ST" in d or "_T_ST" in d)
      and ("IDHMutant" not in d)
      and ("_TC_" not in d)
      and ("_TI_" not in d)
  ]

  records = []
  for idx, s_folder in enumerate(target_samples, 1):
    outs_dir = os.path.join(data2_dir, s_folder, "outs")
    clean_name = s_folder.replace("#", "")
    grp = "Cortex" if "_C_ST" in s_folder else "Tumor"
    print(f"  [{idx:02d}/24] Reading {clean_name} ({grp})...", flush=True)

    if not os.path.exists(outs_dir):
      continue

    try:
      adata = sc.read_visium(outs_dir)
      adata.var_names_make_unique()
      n_spots = adata.n_obs

      dh_raw = (
          adata.obs_vector("DHODH")
          if "DHODH" in adata.var_names
          else np.zeros(n_spots)
      )
      sh_raw = (
          adata.obs_vector("SHMT2")
          if "SHMT2" in adata.var_names
          else np.zeros(n_spots)
      )

      dh_pct = float((dh_raw > 0).mean() * 100.0)
      sh_pct = float((sh_raw > 0).mean() * 100.0)

      records.append({
          "Sample_ID": clean_name,
          "Tissue_Group": grp,
          "Total_Spots_n": n_spots,
          "DHODH_Pct_Positive": dh_pct,
          "SHMT2_Pct_Positive": sh_pct,
      })
      del adata, dh_raw, sh_raw
      gc.collect()
    except Exception as e:
      print(f"     Error on {clean_name}: {e}")

  df = pd.DataFrame(records)
  saved_csv = os.path.join(
      results2_dir, "Table_Data2_All_24_Cortex_and_Tumor_Summary.csv"
  )
  df.to_csv(saved_csv, index=False)
  print(f"--> Saved extracted metrics to: {saved_csv}\n")
else:
  print(f"Loading existing metrics from: {csv_path}")
  df = pd.read_csv(csv_path)

# ==============================================================================
# 3. HARMONIZE COLUMN NAMES
# ==============================================================================
if "Sample_ID" not in df.columns:
  if "Sample" in df.columns:
    df["Sample_ID"] = df["Sample"]
  elif "sample_name" in df.columns:
    df["Sample_ID"] = df["sample_name"]

df["Clean_Sample"] = df["Sample_ID"].astype(str).str.replace("#", "")

if "Tissue_Group" not in df.columns and "Group" in df.columns:
  df["Tissue_Group"] = df["Group"]

if "DHODH_Pct_Positive" not in df.columns and "DHODH_positive_percent" in df.columns:
  df["DHODH_Pct_Positive"] = df["DHODH_positive_percent"]

if "SHMT2_Pct_Positive" not in df.columns and "SHMT2_positive_percent" in df.columns:
  df["SHMT2_Pct_Positive"] = df["SHMT2_positive_percent"]

if "Total_Spots_n" not in df.columns and "Total_spots" in df.columns:
  df["Total_Spots_n"] = df["Total_spots"]

# Filter strictly for Cortex and Tumor
df = df[df["Tissue_Group"].isin(["Cortex", "Tumor"])].copy()

cortex_df = df[df["Tissue_Group"] == "Cortex"]
tumor_df = df[df["Tissue_Group"] == "Tumor"]

print(
    f"Samples loaded: Cortex n = {len(cortex_df)}, Tumor n = {len(tumor_df)}"
)

# ==============================================================================
# 4. STATISTICAL CALCULATIONS
# ==============================================================================
# DHODH Stats
t_dh, p_dh = stats.ttest_ind(
    tumor_df["DHODH_Pct_Positive"],
    cortex_df["DHODH_Pct_Positive"],
    equal_var=False,
)
dh_fold = (
    tumor_df["DHODH_Pct_Positive"].mean()
    / cortex_df["DHODH_Pct_Positive"].mean()
)

# SHMT2 Stats
t_sh, p_sh = stats.ttest_ind(
    tumor_df["SHMT2_Pct_Positive"],
    cortex_df["SHMT2_Pct_Positive"],
    equal_var=False,
)
sh_fold = (
    tumor_df["SHMT2_Pct_Positive"].mean()
    / cortex_df["SHMT2_Pct_Positive"].mean()
)

print(
    f"DHODH: Fold-enrichment = {dh_fold:.2f}x, Welch's t = {t_dh:.3f}, p ="
    f" {p_dh:.4e}"
)
print(
    f"SHMT2: Fold-enrichment = {sh_fold:.2f}x, Welch's t = {t_sh:.3f}, p ="
    f" {p_sh:.4e}"
)

# ==============================================================================
# 5. GENERATE 4-PANEL COMPOSITE DISPLAY FIGURE (VECTOR PDF)
# ==============================================================================
fig = plt.figure(figsize=(18, 9), dpi=300)
gs = plt.GridSpec(2, 2, width_ratios=[1.1, 1.3], wspace=0.25, hspace=0.35)
palette = {"Cortex": "#1F78B4", "Tumor": "#E41A1C"}

# --- Panel A: DHODH Box Plot ---
ax_a = fig.add_subplot(gs[0, 0])
sns.boxplot(
    data=df,
    x="Tissue_Group",
    y="DHODH_Pct_Positive",
    palette=palette,
    width=0.45,
    ax=ax_a,
    boxprops=dict(alpha=0.65),
    showfliers=False,
)
sns.stripplot(
    data=df,
    x="Tissue_Group",
    y="DHODH_Pct_Positive",
    palette=palette,
    size=STRIP_DOT_SIZE,
    jitter=0.2,
    edgecolor="black",
    linewidth=0.8,
    ax=ax_a,
)

y_max_dh = df["DHODH_Pct_Positive"].max() * 1.05
ax_a.plot(
    [0, 0, 1, 1],
    [y_max_dh, y_max_dh * 1.06, y_max_dh * 1.06, y_max_dh],
    color="black",
    lw=1.2,
)
ax_a.text(
    0.5,
    y_max_dh * 1.07,
    f"p = {p_dh:.4f} ({dh_fold:.2f}x)",
    ha="center",
    va="bottom",
    fontsize=10.5,
    fontweight="bold",
    color="#B71C1C",
)
ax_a.set_ylim(-0.5, y_max_dh * 1.25)
ax_a.set_title(
    "DHODH Detection Rate: Cortex vs. Tumor",
    fontsize=12,
    fontweight="bold",
    pad=10,
)
ax_a.set_xlabel("Tissue Group", fontsize=11, fontweight="bold")
ax_a.set_ylabel(
    f"DHODH-Positive Spots (%)\n[Cortex n={len(cortex_df)} | Tumor"
    f" n={len(tumor_df)}]",
    fontsize=10.5,
    fontweight="bold",
)
ax_a.set_xticklabels(
    [f"Non-Tumor Cortex\n(n = {len(cortex_df)})", f"Primary Tumor\n(n = {len(tumor_df)})"],
    fontsize=10,
    fontweight="bold",
)
sns.despine(ax=ax_a)
ax_a.text(
    -0.15,
    1.05,
    "A",
    transform=ax_a.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
)

# --- Panel B: SHMT2 Box Plot ---
ax_b = fig.add_subplot(gs[1, 0])
sns.boxplot(
    data=df,
    x="Tissue_Group",
    y="SHMT2_Pct_Positive",
    palette=palette,
    width=0.45,
    ax=ax_b,
    boxprops=dict(alpha=0.65),
    showfliers=False,
)
sns.stripplot(
    data=df,
    x="Tissue_Group",
    y="SHMT2_Pct_Positive",
    palette=palette,
    size=STRIP_DOT_SIZE,
    jitter=0.2,
    edgecolor="black",
    linewidth=0.8,
    ax=ax_b,
)

y_max_sh = df["SHMT2_Pct_Positive"].max() * 1.05
ax_b.plot(
    [0, 0, 1, 1],
    [y_max_sh, y_max_sh * 1.06, y_max_sh * 1.06, y_max_sh],
    color="black",
    lw=1.2,
)
ax_b.text(
    0.5,
    y_max_sh * 1.07,
    f"p = {p_sh:.4f} ({sh_fold:.2f}x)",
    ha="center",
    va="bottom",
    fontsize=10.5,
    fontweight="bold",
    color="#B71C1C",
)
ax_b.set_ylim(-1.5, y_max_sh * 1.25)
ax_b.set_title(
    "SHMT2 Detection Rate: Cortex vs. Tumor",
    fontsize=12,
    fontweight="bold",
    pad=10,
)
ax_b.set_xlabel("Tissue Group", fontsize=11, fontweight="bold")
ax_b.set_ylabel(
    f"SHMT2-Positive Spots (%)\n[Cortex n={len(cortex_df)} | Tumor"
    f" n={len(tumor_df)}]",
    fontsize=10.5,
    fontweight="bold",
)
ax_b.set_xticklabels(
    [f"Non-Tumor Cortex\n(n = {len(cortex_df)})", f"Primary Tumor\n(n = {len(tumor_df)})"],
    fontsize=10,
    fontweight="bold",
)
sns.despine(ax=ax_b)
ax_b.text(
    -0.15,
    1.05,
    "B",
    transform=ax_b.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
)

# --- Panel C: Sample-Wide DHODH Dot Plot ---
df_sorted = df.sort_values("DHODH_Pct_Positive", ascending=True).reset_index(
    drop=True
)
y_pos = np.arange(len(df_sorted))
colors = [
    "#1F78B4" if g == "Cortex" else "#D95F02" for g in df_sorted["Tissue_Group"]
]

ax_c = fig.add_subplot(gs[0, 1])
ax_c.hlines(
    y=y_pos,
    xmin=0,
    xmax=df_sorted["DHODH_Pct_Positive"],
    color="#CFD8DC",
    linewidth=1.2,
)
ax_c.scatter(
    df_sorted["DHODH_Pct_Positive"],
    y_pos,
    color=colors,
    s=RANKED_DOT_SIZE,
    zorder=3,
    edgecolor="black",
    linewidth=0.8,
)

y_labels = [
    f"{row['Clean_Sample']} (n={int(row['Total_Spots_n']):,})"
    for _, row in df_sorted.iterrows()
]
ax_c.set_yticks(y_pos)
ax_c.set_yticklabels(y_labels, fontsize=8, fontweight="bold")
ax_c.set_xlabel("DHODH-Positive Visium Spots (%)", fontsize=10, fontweight="bold")
ax_c.set_title(
    f"Sample-Wide DHODH Expression Across All {len(df_sorted)} Samples",
    fontsize=12,
    fontweight="bold",
    pad=10,
)

c_dh_m = cortex_df["DHODH_Pct_Positive"].mean()
t_dh_m = tumor_df["DHODH_Pct_Positive"].mean()
ax_c.axvline(
    c_dh_m,
    color="#1F78B4",
    linestyle="--",
    linewidth=1.2,
    label=f"Cortex Mean: {c_dh_m:.1f}%",
)
ax_c.axvline(
    t_dh_m,
    color="#D95F02",
    linestyle="--",
    linewidth=1.2,
    label=f"Tumor Mean: {t_dh_m:.1f}%",
)
ax_c.legend(loc="lower right", frameon=True, fontsize=8.5)
sns.despine(ax=ax_c)
ax_c.text(
    -0.25,
    1.05,
    "C",
    transform=ax_c.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
)

# --- Panel D: Sample-Wide SHMT2 Dot Plot ---
ax_d = fig.add_subplot(gs[1, 1])
ax_d.hlines(
    y=y_pos,
    xmin=0,
    xmax=df_sorted["SHMT2_Pct_Positive"],
    color="#CFD8DC",
    linewidth=1.2,
)
ax_d.scatter(
    df_sorted["SHMT2_Pct_Positive"],
    y_pos,
    color=colors,
    s=RANKED_DOT_SIZE,
    zorder=3,
    edgecolor="black",
    linewidth=0.8,
)

ax_d.set_yticks(y_pos)
ax_d.set_yticklabels(y_labels, fontsize=8, fontweight="bold")
ax_d.set_xlabel("SHMT2-Positive Visium Spots (%)", fontsize=10, fontweight="bold")
ax_d.set_title(
    f"Sample-Wide SHMT2 Expression Across All {len(df_sorted)} Samples",
    fontsize=12,
    fontweight="bold",
    pad=10,
)

c_sh_m = cortex_df["SHMT2_Pct_Positive"].mean()
t_sh_m = tumor_df["SHMT2_Pct_Positive"].mean()
ax_d.axvline(
    c_sh_m,
    color="#1F78B4",
    linestyle="--",
    linewidth=1.2,
    label=f"Cortex Mean: {c_sh_m:.1f}%",
)
ax_d.axvline(
    t_sh_m,
    color="#D95F02",
    linestyle="--",
    linewidth=1.2,
    label=f"Tumor Mean: {t_sh_m:.1f}%",
)
ax_d.legend(loc="lower right", frameon=True, fontsize=8.5)
sns.despine(ax=ax_d)
ax_d.text(
    -0.25,
    1.05,
    "D",
    transform=ax_d.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
)

# Save as vector PDF
out_fig_file = os.path.join(
    results2_dir, "Fig_Data2_SampleWide_BoxPlots_and_DotPlots_Composite.pdf"
)
fig.savefig(out_fig_file, format="pdf", bbox_inches="tight")
plt.close(fig)

print("====================================================================")
print(f"SUCCESS: Vector PDF figure generated and saved to:\n  {out_fig_file}")
print("====================================================================")


# %%
