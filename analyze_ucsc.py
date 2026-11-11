#%%
import os
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.gridspec import GridSpec
import scanpy as sc

# ==============================================================================
# 1. SETUP DIRECTORIES & PUBLICATION VECTOR PDF SETTINGS
# ==============================================================================
base_dir = "/Users/mekalav/Documents/BCM/Dropbox_system/Sai_Glio"
data_dir = os.path.join(base_dir, "data/UCSC_GBM_Atlas")
results_dir = os.path.join(base_dir, "results_data2_ucsc")
os.makedirs(results_dir, exist_ok=True)

# Strict vector PDF configuration with embedded TrueType Type 42 fonts
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.dpi'] = 300

# Standardized Ivy GAP Anatomical Niches 10-Color Palette
IVY_PALETTE = {
    'PNZ':   '#2B83BA',  # Perinecrotic Zone (Deep Blue)
    'HBV':   '#D7191C',  # Hyperplastic Blood Vessels (Crimson Red)
    'IGN':   '#FDAE61',  # Infiltrating Glioma Niche (Amber)
    'CT':    '#4DAC26',  # Cellular Tumor Core (Forest Green)
    'MVP':   '#D01C8B',  # Microvascular Proliferation (Magenta)
    'PAN':   '#80CDC1',  # Pseudopalisading Cells (Turquoise)
    'BV':    '#F1B6DA',  # Leading Normal Blood Vessels (Light Pink)
    'IT':    '#B8E186',  # Infiltrating Tumor Border (Light Green)
    'LE_GM': '#CAB2D6',  # Leading Edge Grey Matter (Lavender)
    'LE_WM': '#BDBDBD'   # Leading Edge White Matter (Light Grey)
}

print("================================================================================")
print("UCSC MULTIOMIC GLIOBLASTOMA ATLAS: FULL 32-SLIDE COHORT ANALYSIS")
print("================================================================================\n")

# ==============================================================================
# 2. LOAD ATLAS METADATA, EXPRESSION, AND COORDINATES
# ==============================================================================
print("[1/4] Reading metadata (meta.tsv)...", flush=True)
meta_df = pd.read_csv(os.path.join(data_dir, "meta.tsv"), sep='\t', index_col=0)

print("[2/4] Reading gene symbols & spot barcodes...", flush=True)
features_df = pd.read_csv(os.path.join(data_dir, "features.tsv.gz"), header=None, sep='\t')
raw_genes = features_df[1].values if features_df.shape[1] > 1 else features_df[0].values
clean_genes = [str(g).split('|')[-1].upper() for g in raw_genes]

barcodes_df = pd.read_csv(os.path.join(data_dir, "barcodes.tsv.gz"), header=None, sep='\t')
spot_barcodes = barcodes_df[0].astype(str).values

print("[3/4] Reading expression matrix (matrix.mtx.gz)...", flush=True)
adata = sc.read_mtx(os.path.join(data_dir, "matrix.mtx.gz")).T
adata.obs_names = spot_barcodes
adata.var_names = clean_genes
adata.var_names_make_unique()
adata.obs = meta_df.reindex(adata.obs_names)

# Library-size normalization (CPM) & Log1p
if adata.X.max() > 50:
    print("      Applying library size normalization (target_sum=1e4) and log1p...", flush=True)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

dh_vals = adata.obs_vector('DHODH')
sh_vals = adata.obs_vector('SHMT2')
adata.obs['DHODH'] = dh_vals
adata.obs['SHMT2'] = sh_vals

print("[4/4] Aligning Visium coordinates (Visium.coords.tsv.gz)...", flush=True)
coords_file = os.path.join(data_dir, "Visium.coords.tsv.gz")
coords_raw = pd.read_csv(coords_file, sep='\t', header=None)
try:
    float(coords_raw.iloc[0, 1])
    coords_df = coords_raw.set_index(0)
    coords_df.columns = ['x', 'y']
except (ValueError, TypeError):
    coords_df = pd.read_csv(coords_file, sep='\t', header=0, index_col=0)
    x_c = [c for c in coords_df.columns if 'x' in str(c).lower() or '1' in str(c)][0]
    y_c = [c for c in coords_df.columns if 'y' in str(c).lower() or '2' in str(c)][0]
    coords_df = coords_df[[x_c, y_c]]
    coords_df.columns = ['x', 'y']

coords_aligned = coords_df.reindex(adata.obs_names)
adata.obs['x'] = pd.to_numeric(coords_aligned['x'], errors='coerce')
adata.obs['y'] = pd.to_numeric(coords_aligned['y'], errors='coerce')

print(f"\n--> Assembled Atlas: {adata.n_obs:,} spots across {adata.n_vars:,} genes.")

# ==============================================================================
# 3. COMPUTE SECTION-LEVEL METRICS ACROSS ALL 32 SLIDES
# ==============================================================================
slide_col = 'orig.ident' if 'orig.ident' in adata.obs.columns else 'sample'
all_slides = sorted(adata.obs[slide_col].dropna().unique())
print(f"\nIdentified {len(all_slides)} distinct tissue slides across the cohort.")

slide_records = []
for s_name in all_slides:
    s_df = adata.obs[adata.obs[slide_col] == s_name]
    n_spots = len(s_df)
    if n_spots == 0:
        continue
    
    dh_arr = s_df['DHODH'].values
    sh_arr = s_df['SHMT2'].values
    
    diag = s_df['Diagnosis'].dropna().iloc[0] if 'Diagnosis' in s_df.columns and len(s_df['Diagnosis'].dropna()) > 0 else 'Glioblastoma'
    pat_id = s_df['sample'].dropna().iloc[0] if 'sample' in s_df.columns and len(s_df['sample'].dropna()) > 0 else s_name
    
    # Classify group (GBM vs. Non-GBM Control/Astrocytoma)
    group_label = "Glioblastoma" if "glioblastoma" in str(diag).lower() or "gbm" in str(diag).lower() else "Astrocytoma / Control"
    
    r_val = float(np.corrcoef(dh_arr, sh_arr)[0, 1]) if n_spots > 2 else 0.0
    
    slide_records.append({
        'Slide_ID': s_name,
        'Patient_ID': pat_id,
        'Diagnosis': diag,
        'Tissue_Group': group_label,
        'Total_Spots_n': n_spots,
        'DHODH_Positive_Spots_n': int((dh_arr > 0).sum()),
        'DHODH_Pct_Positive': float((dh_arr > 0).mean() * 100.0),
        'DHODH_Mean_Expr': float(np.mean(dh_arr)),
        'SHMT2_Positive_Spots_n': int((sh_arr > 0).sum()),
        'SHMT2_Pct_Positive': float((sh_arr > 0).mean() * 100.0),
        'SHMT2_Mean_Expr': float(np.mean(sh_arr)),
        'Pearson_r': 0.0 if np.isnan(r_val) else r_val
    })

df_all_slides = pd.DataFrame(slide_records).sort_values('DHODH_Pct_Positive', ascending=True).reset_index(drop=True)
csv_slides_out = os.path.join(results_dir, "Table_UCSC_All_32_Slides_Summary.csv")
df_all_slides.to_csv(csv_slides_out, index=False)
print(f"[+] Exported Master Summary Table across all 32 slides to:\n    {csv_slides_out}")

# ==============================================================================
# 4. FIGURE 1: SAMPLE-WIDE BOX PLOTS & RANKED DOT PLOTS (PDF)
# ==============================================================================
print("\n[Rendering Fig 1] Sample-Wide Box & Dot Plots across all 32 slides...", flush=True)
fig_sw = plt.figure(figsize=(19, 10), dpi=300)
gs = GridSpec(2, 2, width_ratios=[1.05, 1.35], wspace=0.25, hspace=0.35)

palette_grp = {"Glioblastoma": "#E41A1C", "Astrocytoma / Control": "#1F78B4"}

# Stats: GBM vs Non-GBM slides
gbm_df = df_all_slides[df_all_slides['Tissue_Group'] == 'Glioblastoma']
ctrl_df = df_all_slides[df_all_slides['Tissue_Group'] != 'Glioblastoma']

t_dh, p_dh = (stats.ttest_ind(gbm_df['DHODH_Pct_Positive'], ctrl_df['DHODH_Pct_Positive'], equal_var=False) 
              if len(ctrl_df) > 1 else (np.nan, np.nan))
t_sh, p_sh = (stats.ttest_ind(gbm_df['SHMT2_Pct_Positive'], ctrl_df['SHMT2_Pct_Positive'], equal_var=False)
              if len(ctrl_df) > 1 else (np.nan, np.nan))

# --- Panel A: DHODH Box Plot ---
ax_a = fig_sw.add_subplot(gs[0, 0])
sns.boxplot(data=df_all_slides, x='Tissue_Group', y='DHODH_Pct_Positive', palette=palette_grp,
            width=0.45, ax=ax_a, boxprops=dict(alpha=0.65), showfliers=False)
sns.stripplot(data=df_all_slides, x='Tissue_Group', y='DHODH_Pct_Positive', palette=palette_grp,
              size=7.5, jitter=0.2, edgecolor='black', linewidth=0.8, ax=ax_a)
ax_a.set_title("DHODH Detection Rate: Glioblastoma vs. Controls", fontsize=11.5, fontweight='bold', pad=10)
ax_a.set_ylabel("DHODH-Positive Spots (%)", fontsize=10.5, fontweight='bold')
ax_a.set_xlabel("")
if not np.isnan(p_dh):
    y_top = df_all_slides['DHODH_Pct_Positive'].max() * 1.05
    ax_a.plot([0, 0, 1, 1], [y_top, y_top*1.05, y_top*1.05, y_top], color='black', lw=1.2)
    ax_a.text(0.5, y_top*1.06, f"p = {p_dh:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#B71C1C')
sns.despine(ax=ax_a)
ax_a.text(-0.15, 1.05, 'A', transform=ax_a.transAxes, fontsize=15, fontweight='bold', va='top')

# --- Panel B: SHMT2 Box Plot ---
ax_b = fig_sw.add_subplot(gs[1, 0])
sns.boxplot(data=df_all_slides, x='Tissue_Group', y='SHMT2_Pct_Positive', palette=palette_grp,
            width=0.45, ax=ax_b, boxprops=dict(alpha=0.65), showfliers=False)
sns.stripplot(data=df_all_slides, x='Tissue_Group', y='SHMT2_Pct_Positive', palette=palette_grp,
              size=7.5, jitter=0.2, edgecolor='black', linewidth=0.8, ax=ax_b)
ax_b.set_title("SHMT2 Detection Rate: Glioblastoma vs. Controls", fontsize=11.5, fontweight='bold', pad=10)
ax_b.set_ylabel("SHMT2-Positive Spots (%)", fontsize=10.5, fontweight='bold')
ax_b.set_xlabel("Tissue Classification", fontsize=10.5, fontweight='bold')
if not np.isnan(p_sh):
    y_top_sh = df_all_slides['SHMT2_Pct_Positive'].max() * 1.05
    ax_b.plot([0, 0, 1, 1], [y_top_sh, y_top_sh*1.05, y_top_sh*1.05, y_top_sh], color='black', lw=1.2)
    ax_b.text(0.5, y_top_sh*1.06, f"p = {p_sh:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#B71C1C')
sns.despine(ax=ax_b)
ax_b.text(-0.15, 1.05, 'B', transform=ax_b.transAxes, fontsize=15, fontweight='bold', va='top')

# --- Panel C: DHODH Ranked Dot Plot (All 32 Slides) ---
ax_c = fig_sw.add_subplot(gs[0, 1])
y_pos = np.arange(len(df_all_slides))
dot_colors = ['#E41A1C' if g == 'Glioblastoma' else '#1F78B4' for g in df_all_slides['Tissue_Group']]

ax_c.hlines(y=y_pos, xmin=0, xmax=df_all_slides['DHODH_Pct_Positive'], color='#CFD8DC', linewidth=1.1)
ax_c.scatter(df_all_slides['DHODH_Pct_Positive'], y_pos, color=dot_colors, s=50, zorder=3, edgecolor='black', linewidth=0.8)
y_lbls = [f"{row['Slide_ID']} (n={int(row['Total_Spots_n']):,})" for _, row in df_all_slides.iterrows()]
ax_c.set_yticks(y_pos)
ax_c.set_yticklabels(y_lbls, fontsize=7.5, fontweight='bold')
ax_c.set_xlabel("DHODH-Positive Visium Spots (%)", fontsize=10, fontweight='bold')
ax_c.set_title(f"Sample-Wide DHODH Across All {len(df_all_slides)} Slides", fontsize=11.5, fontweight='bold', pad=10)
ax_c.axvline(df_all_slides['DHODH_Pct_Positive'].mean(), color='black', linestyle='--', linewidth=1.1, 
             label=f"Cohort Mean: {df_all_slides['DHODH_Pct_Positive'].mean():.1f}%")
ax_c.legend(loc='lower right', frameon=True, fontsize=8.5)
sns.despine(ax=ax_c)
ax_c.text(-0.25, 1.05, 'C', transform=ax_c.transAxes, fontsize=15, fontweight='bold', va='top')

# --- Panel D: SHMT2 Ranked Dot Plot (All 32 Slides) ---
ax_d = fig_sw.add_subplot(gs[1, 1])
ax_d.hlines(y=y_pos, xmin=0, xmax=df_all_slides['SHMT2_Pct_Positive'], color='#CFD8DC', linewidth=1.1)
ax_d.scatter(df_all_slides['SHMT2_Pct_Positive'], y_pos, color=dot_colors, s=50, zorder=3, edgecolor='black', linewidth=0.8)
ax_d.set_yticks(y_pos)
ax_d.set_yticklabels(y_lbls, fontsize=7.5, fontweight='bold')
ax_d.set_xlabel("SHMT2-Positive Visium Spots (%)", fontsize=10, fontweight='bold')
ax_d.set_title(f"Sample-Wide SHMT2 Across All {len(df_all_slides)} Slides", fontsize=11.5, fontweight='bold', pad=10)
ax_d.axvline(df_all_slides['SHMT2_Pct_Positive'].mean(), color='black', linestyle='--', linewidth=1.1,
             label=f"Cohort Mean: {df_all_slides['SHMT2_Pct_Positive'].mean():.1f}%")
ax_d.legend(loc='lower right', frameon=True, fontsize=8.5)
sns.despine(ax=ax_d)
ax_d.text(-0.25, 1.05, 'D', transform=ax_d.transAxes, fontsize=15, fontweight='bold', va='top')

plt.tight_layout()
fig1_pdf = os.path.join(results_dir, "Fig_UCSC_SampleWide_BoxPlots_and_DotPlots_All_32_Slides.pdf")
fig_sw.savefig(fig1_pdf, format='pdf', bbox_inches='tight')
plt.close(fig_sw)
print(f"[+] Saved Vector PDF: {fig1_pdf}")

# ==============================================================================
# 5. FIGURE 2: NICHE-LEVEL BOX PLOTS ACROSS INDEPENDENT SLIDES (PDF)
# ==============================================================================
print("\n[Rendering Fig 2] Multi-slide Niche Box & Dot Plots...", flush=True)
# Evaluate % positive spots per niche within EACH slide that has >= 20 spots in that niche
slide_niche_data = []
for s_name, s_df in adata.obs.groupby(slide_col):
    if 'AF' not in s_df.columns:
        continue
    for n_name, n_df in s_df.groupby('AF'):
        if len(n_df) >= 20 and str(n_name) not in ['nan', 'None']:
            slide_niche_data.append({
                'Slide_ID': s_name,
                'Anatomical_Niche': str(n_name),
                'Spots_n': len(n_df),
                'DHODH_Pct_Positive': float((n_df['DHODH'] > 0).mean() * 100.0),
                'SHMT2_Pct_Positive': float((n_df['SHMT2'] > 0).mean() * 100.0)
            })

df_slide_niches = pd.DataFrame(slide_niche_data)
# Order niches by cohort mean DHODH
niche_order = ['PNZ', 'HBV', 'IGN', 'CT', 'MVP', 'PAN', 'BV', 'IT', 'LE_GM', 'LE_WM']
niche_order = [n for n in niche_order if n in df_slide_niches['Anatomical_Niche'].unique()]

fig_nbox, (ax_nb1, ax_nb2) = plt.subplots(1, 2, figsize=(16, 5.5), dpi=300)

# DHODH Niche Box + Dot Plot
sns.boxplot(data=df_slide_niches, x='Anatomical_Niche', y='DHODH_Pct_Positive', order=niche_order,
            palette=IVY_PALETTE, ax=ax_nb1, width=0.5, boxprops=dict(alpha=0.65), showfliers=False)
sns.stripplot(data=df_slide_niches, x='Anatomical_Niche', y='DHODH_Pct_Positive', order=niche_order,
              palette=IVY_PALETTE, size=6.5, jitter=0.2, edgecolor='black', linewidth=0.8, ax=ax_nb1)
ax_nb1.set_xticklabels(niche_order, rotation=35, ha='right', fontsize=9.5, fontweight='bold')
ax_nb1.set_ylabel("DHODH-Positive Spots (%) [Per Slide]", fontsize=10.5, fontweight='bold')
ax_nb1.set_xlabel("Ivy GAP Anatomical Niches", fontsize=10.5, fontweight='bold')
ax_nb1.set_title("DHODH Variation Across Slides by Niche\n(Dots = Independent Visium Slides)", fontsize=11.5, fontweight='bold', pad=10)
sns.despine(ax=ax_nb1)

# SHMT2 Niche Box + Dot Plot
sns.boxplot(data=df_slide_niches, x='Anatomical_Niche', y='SHMT2_Pct_Positive', order=niche_order,
            palette=IVY_PALETTE, ax=ax_nb2, width=0.5, boxprops=dict(alpha=0.65), showfliers=False)
sns.stripplot(data=df_slide_niches, x='Anatomical_Niche', y='SHMT2_Pct_Positive', order=niche_order,
              palette=IVY_PALETTE, size=6.5, jitter=0.2, edgecolor='black', linewidth=0.8, ax=ax_nb2)
ax_nb2.set_xticklabels(niche_order, rotation=35, ha='right', fontsize=9.5, fontweight='bold')
ax_nb2.set_ylabel("SHMT2-Positive Spots (%) [Per Slide]", fontsize=10.5, fontweight='bold')
ax_nb2.set_xlabel("Ivy GAP Anatomical Niches", fontsize=10.5, fontweight='bold')
ax_nb2.set_title("SHMT2 Variation Across Slides by Niche\n(Dots = Independent Visium Slides)", fontsize=11.5, fontweight='bold', pad=10)
sns.despine(ax=ax_nb2)

plt.tight_layout()
fig2_pdf = os.path.join(results_dir, "Fig_UCSC_Niche_BoxPlots_and_DotPlots_Across_Slides.pdf")
fig_nbox.savefig(fig2_pdf, format='pdf', bbox_inches='tight')
plt.close(fig_nbox)
print(f"[+] Saved Vector PDF: {fig2_pdf}")

# ==============================================================================
# 6. FIGURE 3: NICHE BAR PLOTS (COHORT-WIDE MEAN ± SEM) (PDF)
# ==============================================================================
print("\n[Rendering Fig 3] Re-verifying Vector Barplots...", flush=True)
csv_af_path = os.path.join(results_dir, "Table_UCSC_Anatomical_Niches_AF_Metrics.csv")
if os.path.exists(csv_af_path):
    df_af_bar = pd.read_csv(csv_af_path)
    fig_bar, (b_ax1, b_ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)
    x_pos = np.arange(len(df_af_bar))
    width = 0.38
    
    b_ax1.bar(x_pos - width/2, df_af_bar['DHODH_Pct_Positive'], width, label='DHODH', color='#E66101', edgecolor='black', linewidth=0.8)
    b_ax1.bar(x_pos + width/2, df_af_bar['SHMT2_Pct_Positive'], width, label='SHMT2', color='#5E3C99', edgecolor='black', linewidth=0.8)
    b_ax1.set_xticks(x_pos)
    b_ax1.set_xticklabels(df_af_bar['Category'], rotation=35, ha='right', fontsize=9.5, fontweight='bold')
    b_ax1.set_ylabel("Spots Positive (%)", fontsize=10.5, fontweight='bold')
    b_ax1.set_title("Detection Rate Across Published Niches (AF)\n[N = 115,914 Spots | UCSC Multiomic Glioblastoma Atlas]", fontsize=11, fontweight='bold', pad=10)
    b_ax1.legend(frameon=True, loc='upper right', fontsize=9.5)
    sns.despine(ax=b_ax1)

    b_ax2.bar(x_pos - width/2, df_af_bar['DHODH_Mean_Expr'], width, yerr=df_af_bar['DHODH_SEM'],
              capsize=4, label='DHODH', color='#E66101', edgecolor='black', linewidth=0.8)
    b_ax2.bar(x_pos + width/2, df_af_bar['SHMT2_Mean_Expr'], width, yerr=df_af_bar['SHMT2_SEM'],
              capsize=4, label='SHMT2', color='#5E3C99', edgecolor='black', linewidth=0.8)
    b_ax2.set_xticks(x_pos)
    b_ax2.set_xticklabels(df_af_bar['Category'], rotation=35, ha='right', fontsize=9.5, fontweight='bold')
    b_ax2.set_ylabel("Mean Expression (log CPM ± SEM)", fontsize=10.5, fontweight='bold')
    b_ax2.set_title("Expression Density Across Published Niches (AF)\n[N = 115,914 Spots | UCSC Multiomic Glioblastoma Atlas]", fontsize=11, fontweight='bold', pad=10)
    b_ax2.legend(frameon=True, loc='upper right', fontsize=9.5)
    sns.despine(ax=b_ax2)

    plt.tight_layout()
    fig3_pdf = os.path.join(results_dir, "Fig_UCSC_Anatomical_Niches_AF_BarPlots.pdf")
    fig_bar.savefig(fig3_pdf, format='pdf', bbox_inches='tight')
    plt.close(fig_bar)
    print(f"[+] Saved Vector PDF: {fig3_pdf}")

#%%
# ==============================================================================
# 7. FIGURE 4: BATCH GENERATE 3-PANEL SPATIAL VECTOR PDFS FOR ALL 32 SLIDES
# ==============================================================================
slides_pdf_dir = os.path.join(results_dir, "all_32_slides_pdf")
os.makedirs(slides_pdf_dir, exist_ok=True)

print(f"\n================================================================================")
print(f"BATCH RENDERING 3-PANEL SPATIAL VECTOR PDFS FOR ALL {len(all_slides)} SLIDES")
print(f"Destination: {slides_pdf_dir}")
print("================================================================================\n")

for idx, s_name in enumerate(all_slides, 1):
    sub = adata.obs[adata.obs[slide_col] == s_name].copy()
    sub = sub.dropna(subset=['x', 'y', 'AF'])
    sub = sub[~sub['AF'].isin(['nan', 'None'])]
    n_slide_spots = len(sub)
    
    if n_slide_spots < 50:
        print(f"[{idx:02d}/{len(all_slides)}] Skipping {s_name}: Insufficient spots ({n_slide_spots}).")
        continue
        
    print(f"[{idx:02d}/{len(all_slides)}] Rendering Slide: {s_name} ({n_slide_spots:,} spots)...", flush=True)
    
    x_coords = sub['x'].values
    y_coords = -sub['y'].values if sub['y'].min() >= 0 else sub['y'].values
    
    dh_sub = sub['DHODH'].values
    sh_sub = sub['SHMT2'].values
    vmax_dh = float(np.percentile(dh_sub[dh_sub > 0], 98)) if (dh_sub > 0).sum() > 0 else 1.0
    vmax_sh = float(np.percentile(sh_sub[sh_sub > 0], 98)) if (sh_sub > 0).sum() > 0 else 1.0
    
    dh_pct = (dh_sub > 0).mean() * 100.0
    sh_pct = (sh_sub > 0).mean() * 100.0
    r_val = float(np.corrcoef(dh_sub, sh_sub)[0, 1]) if len(sub) > 2 else 0.0

    fig_slide, axes_slide = plt.subplots(1, 3, figsize=(18, 5.8), dpi=300)
    
    # Panel 1: Niches
    unique_niches = [n for n in sub['AF'].unique() if n in IVY_PALETTE]
    other_niches = [n for n in sub['AF'].unique() if n not in IVY_PALETTE]
    slide_palette = {n: IVY_PALETTE[n] for n in unique_niches}
    for n in other_niches:
        slide_palette[n] = '#333333'

    sns.scatterplot(
        x=x_coords, y=y_coords, hue=sub['AF'], palette=slide_palette,
        s=15, linewidth=0.2, edgecolor='black', ax=axes_slide[0]
    )
    axes_slide[0].set_title(f"{s_name}: Ivy GAP Anatomical Niches\n(n = {n_slide_spots:,} spots | {sub['AF'].nunique()} regions)", fontsize=11, fontweight='bold', pad=8)
    axes_slide[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, markerscale=2.0, fontsize=8)
    axes_slide[0].set_aspect('equal')
    axes_slide[0].axis('off')
    
    # Panel 2: DHODH
    im_dh = axes_slide[1].scatter(
        x_coords, y_coords, c=dh_sub, cmap='magma',
        s=15, linewidth=0.2, edgecolor='black', vmin=0.0, vmax=vmax_dh
    )
    axes_slide[1].set_title(f"{s_name}: Normalized DHODH\n({dh_pct:.1f}% spots positive)", fontsize=11, fontweight='bold', pad=8)
    axes_slide[1].set_aspect('equal')
    axes_slide[1].axis('off')
    cbar1 = plt.colorbar(im_dh, ax=axes_slide[1], fraction=0.046, pad=0.04)
    cbar1.set_label("log CPM", fontsize=9, fontweight='bold')
    
    # Panel 3: SHMT2
    im_sh = axes_slide[2].scatter(
        x_coords, y_coords, c=sh_sub, cmap='viridis',
        s=35, linewidth=0.2, edgecolor='black', vmin=0.0, vmax=vmax_sh
    )
    axes_slide[2].set_title(f"{s_name}: Normalized SHMT2\n({sh_pct:.1f}% spots pos | r = {r_val:.3f})", fontsize=11, fontweight='bold', pad=8)
    axes_slide[2].set_aspect('equal')
    axes_slide[2].axis('off')
    cbar2 = plt.colorbar(im_sh, ax=axes_slide[2], fraction=0.046, pad=0.04)
    cbar2.set_label("log CPM", fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    pdf_out = os.path.join(slides_pdf_dir, f"Fig_Spatial_Slide_{s_name}_Niches_and_Expression.pdf")
    fig_slide.savefig(pdf_out, format='pdf', bbox_inches='tight')
    plt.close(fig_slide)
    
    del sub, x_coords, y_coords, dh_sub, sh_sub
    gc.collect()

print("\n================================================================================")
print(f"PIPELINE COMPLETE: ALL 32 SLIDES & COHORT FIGURES EXPORTED TO:\n  {results_dir}")
print("================================================================================")


# %%
