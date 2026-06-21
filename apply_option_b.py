# Option B: cut the final tiers on the DISPLAYED 36-month score (risk_score_scaled),
# preserving the validated Phase 6 tier sizes, so tier and displayed score never disagree.
# Regenerates parquet + tier-profiles CSV + tier map + tier-profiles chart for both models.
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.patches as mpatches, geopandas as gpd
from pathlib import Path
B = Path(r'C:\Users\mbeck\OneDrive\Documents\CBL-16_data')
TIER_COL = {1:'#D62728',2:'#FF7F0E',3:'#2CA02C',4:'#1F77B4'}
TIER_LBL = {1:'Tier 1 - Highest demand',2:'Tier 2 - High demand',3:'Tier 3 - Moderate demand',4:'Tier 4 - Low demand'}

CFG = {
 'london-final-light': dict(
    feats=['severity_weighted_count','ntl_mean_radiance','employment_deprivation','resolution_rate'],
    labels=['Risk\nscore','Severity','Night\nlights','Employ.\ndepriv.','Resolution'],
    shp=('glob', B/'data'/'LB_shp'), figsize=(13,9), title='london-final-light'),
 'england_final_light': dict(
    feats=['ntl_mean_radiance','employment_deprivation','resolution_rate'],
    labels=['Risk\nscore','Night\nlights','Employ.\ndepriv.','Resolution'],
    shp=('ew', B/'data'/'Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BGC_V5_-7203918579177758597'/'LSOA_2021_EW_BGC_V5.shp'),
    figsize=(13,15), title='england_final_light'),
}

for model, cfg in CFG.items():
    OUT = B/model/'outputs'/'phase6'
    p6 = pd.read_parquet(OUT/'phase6_supervised_tiers.parquet')          # supervised_tier (calib), high_demand, risk_score_24
    p4 = pd.read_parquet(B/model/'outputs'/'phase4'/'phase4_risk_scores.parquet')
    p4 = p4[['lsoa21cd','risk_score_scaled']+cfg['feats']]
    d = p6[['lsoa21cd','supervised_tier','high_demand','risk_score_24']].merge(p4, on='lsoa21cd', how='left')

    # Option B re-cut: rank on displayed score, preserve calibrated sizes
    sizes = d['supervised_tier'].value_counts().sort_index()
    order = d['risk_score_scaled'].rank(ascending=False, method='first')
    d['tier'] = np.searchsorted(sizes.cumsum().to_numpy(), order.to_numpy(), side='left') + 1
    moved = int((d['tier'] != d['supervised_tier']).sum())
    print(f'\n=== {model} ===  Option B: {moved} labels moved ({moved/len(d)*100:.1f}%), sizes preserved')

    # Per-tier summary CSV (on the FINAL displayed-score tiers)
    prof = d.groupby('tier').agg(n_lsoas=('lsoa21cd','size'),
            mean_risk_score=('risk_score_scaled','mean'), score_min=('risk_score_scaled','min'),
            score_max=('risk_score_scaled','max'),
            hotspot_share_pct=('high_demand', lambda x: x.mean()*100)).reset_index()
    prof['pct'] = prof['n_lsoas']/len(d)*100
    prof = prof.merge(d.groupby('tier')[cfg['feats']].mean().reset_index(), on='tier').round(2)
    prof.to_csv(OUT/'phase6_tier_profiles.csv', index=False)
    print('displayed-score range per tier (now contiguous):')
    print(prof[['tier','n_lsoas','pct','score_min','score_max','hotspot_share_pct']].to_string(index=False))

    # updated parquet (tier = displayed-score Option B; keep calibration label)
    keep = ['lsoa21cd','tier','supervised_tier','risk_score_scaled','risk_score_24','high_demand']+cfg['feats']
    d[keep].to_parquet(OUT/'phase6_supervised_tiers.parquet', index=False)

    # Tier map
    if cfg['shp'][0]=='glob':
        gdf = pd.concat([gpd.read_file(f) for f in cfg['shp'][1].glob('*.shp')], ignore_index=True)
        gdf = gpd.GeoDataFrame(gdf, crs=gpd.read_file(list(cfg['shp'][1].glob('*.shp'))[0]).crs)
    else:
        gdf = gpd.read_file(cfg['shp'][1]); gdf = gdf[gdf['LSOA21CD'].str.startswith('E')].rename(columns={'LSOA21CD':'lsoa21cd'})
    g = gdf.merge(d[['lsoa21cd','tier']], on='lsoa21cd', how='left').to_crs(epsg=4326)
    fig, ax = plt.subplots(figsize=cfg['figsize'])
    for t in range(1,5):
        g[g['tier']==t].plot(ax=ax, color=TIER_COL[t], linewidth=0.03, edgecolor='white', alpha=0.85)
    um=g[g['tier'].isna()]
    if len(um): um.plot(ax=ax, color='#cccccc', linewidth=0.03, edgecolor='white')
    ax.legend(handles=[mpatches.Patch(color=TIER_COL[t], label=TIER_LBL[t]) for t in range(1,5)], loc='lower left' if model.startswith('london') else 'upper left', fontsize=9, framealpha=0.9)
    ax.set_title(f'Priority tiers (displayed-score bands at validated sizes) - {cfg["title"]}', fontsize=14, fontweight='bold')
    ax.set_axis_off(); plt.tight_layout()
    plt.savefig(OUT/'6_supervised_tier_map.png', dpi=150, bbox_inches='tight'); plt.close()

    # Tier profiles chart
    pf = ['risk_score_scaled']+cfg['feats']
    pr = d.groupby('tier')[pf].mean(); nr=(pr-pr.min())/(pr.max()-pr.min())
    sz=d['tier'].value_counts().sort_index(); hot=d.groupby('tier')['high_demand'].mean()*100
    fig, axes = plt.subplots(1,4,figsize=(3.4*4,4.6),sharey=True)
    for t,ax in zip(range(1,5),axes):
        ax.bar(cfg['labels'], nr.loc[t].values, color=TIER_COL[t], edgecolor='white', alpha=0.9)
        n=int(sz.get(t,0))
        ax.set_title(f'Tier {t}\nn={n} ({n/len(d)*100:.0f}%)\nreal hotspots: {hot.get(t,0):.0f}%', fontsize=10, fontweight='bold')
        ax.set_ylim(0,1.12); ax.tick_params(axis='x',labelsize=8); ax.spines[['top','right']].set_visible(False)
        if t==1: ax.set_ylabel('Normalised mean (0=low, 1=high)', fontsize=9)
    plt.suptitle(f'Phase 5 tier profiles - {cfg["title"]} (tiers are displayed-score bands at validated sizes)', fontsize=12, y=1.06)
    plt.tight_layout(); plt.savefig(OUT/'6_tier_profiles.png', dpi=150, bbox_inches='tight'); plt.close()
    print('regenerated: parquet, csv, 6_supervised_tier_map.png, 6_tier_profiles.png')
print('\nDONE')
