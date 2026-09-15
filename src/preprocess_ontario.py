"""
Preprocess Ontario DWSP data into a format compatible with our model.
Converts long-format (one row per measurement) to wide-format (one row per sample).
"""

import pandas as pd
import numpy as np
import os
import glob

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'data', 'ontario_dwsp', 'extracted')
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

# Parameter mapping: Ontario name -> our column name
# Features
FEATURE_MAP = {
    'TEMPERATURE, WATER': 'Temp',
    'PH': 'pH',
    'CARBON; DISSOLVED ORGANIC': 'DOC',
    'BROMIDE': 'Br',
    'NITROGEN; NITRITE': 'NO2-N',
    'NITROGEN; AMMONIA+AMMONIUM': 'NH4-N',
}

# We'll use both free and total chlorine
CHLORINE_PARAMS = ['CHLORINE, FREE    FIELD', 'CHLORINE, TOTAL   FIELD', 'CHLORINE, COMB    FIELD']

# Target variables - individual HAA species + totals
TARGET_MAP = {
    'DICHLOROACETIC ACID': 'DCAA',
    'TRICHLOROACETIC ACID': 'TCAA',
    'MONOBROMOACETIC ACID': 'MBAA',
    'MONOCHLOROACETIC ACID': 'MCAA',
    'DIBROMOACETIC ACID': 'DBAA',
    'HALOACETIC ACIDS - HAA5': 'HAA5',
    'TRIHALOMETHANES; TOTAL': 'TTHM',
}

# All parameters we need (uppercase for matching)
ALL_PARAMS = [p.upper() for p in list(FEATURE_MAP.keys()) + CHLORINE_PARAMS + list(TARGET_MAP.keys())]

print("Loading Ontario DWSP data...")
csv_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
print(f"Found {len(csv_files)} CSV files")

# Column name mapping (different files use different column names)
COL_RENAME = {
    'Year ': 'Year', 'Year': 'Year',
    'Sample Program': 'PROGRAM', 'Sample program': 'PROGRAM', 'PROGRAM': 'PROGRAM',
    'Drinking Water System Name': 'DWS_NAME', 'DWS_NAME': 'DWS_NAME',
    'Drinking Water System #': 'DWS_NUMBER', 'DWS_NUMBER': 'DWS_NUMBER',
    'Sample Type': 'SAMPLE_TYPE', 'SAMPLE_TYPE': 'SAMPLE_TYPE',
    'Sample Location': 'SAMPLE_SITE', 'SAMPLE_SITE': 'SAMPLE_SITE',
    'Station #': 'STATION_NO', 'STATION_NO': 'STATION_NO',
    'Sample Date': 'SAMPLE_DATE', 'SAMPLE_DATE': 'SAMPLE_DATE',
    'Sample Time': 'SAMPLE_TIME', 'SAMPLE_TIME': 'SAMPLE_TIME',
    'Parameter Group': 'PARAMETER_GROUP', 'PARAMETER_GROUP': 'PARAMETER_GROUP',
    'Parameter': 'PARAMETER_NAME', 'PARAMETER_NAME': 'PARAMETER_NAME',
    'Analysis Method': 'ANALYSIS_METHOD', 'ANALYSIS_METHOD': 'ANALYSIS_METHOD',
    'Detection Limit (DL)': 'DWSP_DETECTION_LIMIT', 'DWSP_DETECTION_LIMIT': 'DWSP_DETECTION_LIMIT',
    'Detection Limit Unit': 'DETECTION_LIMIT_UNIT', 'DETECTION_LIMIT_UNIT': 'DETECTION_LIMIT_UNIT',
    'Result': 'RESULT_VALUE', 'RESULT_VALUE': 'RESULT_VALUE',
    'Result Unit': 'RESULT_UNIT', 'RESULT_UNIT': 'RESULT_UNIT',
    'Result Qualifier': 'QUALIFIER', 'QUALIFIER': 'QUALIFIER',
    'Sample ID': 'SAMPLE_ID', 'SAMPLE_ID': 'SAMPLE_ID',
}

dfs = []
for f in csv_files:
    print(f"  Reading {os.path.basename(f)}...")
    df = pd.read_csv(f, encoding='utf-8-sig', low_memory=False)
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    rename_map = {}
    for orig_col in df.columns:
        if orig_col in COL_RENAME:
            rename_map[orig_col] = COL_RENAME[orig_col]
    df = df.rename(columns=rename_map)
    # Normalize parameter names to uppercase
    if 'PARAMETER_NAME' in df.columns:
        df['PARAMETER_NAME'] = df['PARAMETER_NAME'].str.upper().str.strip()
    # Normalize sample type
    if 'SAMPLE_TYPE' in df.columns:
        df['SAMPLE_TYPE'] = df['SAMPLE_TYPE'].str.upper().str.strip()
    # Only keep treated water samples and relevant parameters
    df = df[df['SAMPLE_TYPE'] == 'TREATED WATER']
    # Match parameters (uppercase)
    all_params_upper = [p.upper() for p in ALL_PARAMS]
    df = df[df['PARAMETER_NAME'].isin(all_params_upper)]
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)
print(f"Total filtered rows: {len(df_all):,}")

# Handle '<DL' in RESULT_VALUE column (older files use '<DL' as result)
df_all['RESULT_VALUE'] = df_all['RESULT_VALUE'].replace('<DL', np.nan)
df_all['RESULT_VALUE'] = pd.to_numeric(df_all['RESULT_VALUE'], errors='coerce')

# For values below detection limit (qualifier '<' or NaN result with DL), use half DL
if 'QUALIFIER' in df_all.columns:
    df_all['QUALIFIER'] = df_all['QUALIFIER'].astype(str).str.strip()
    mask_bdl = df_all['QUALIFIER'].isin(['<', 'nan']) & df_all['RESULT_VALUE'].isna()
    if 'DWSP_DETECTION_LIMIT' in df_all.columns:
        dl_vals = pd.to_numeric(df_all.loc[mask_bdl, 'DWSP_DETECTION_LIMIT'], errors='coerce') / 2
        df_all.loc[mask_bdl, 'RESULT_VALUE'] = dl_vals

# Create a unique sample identifier: DWS_NAME + SAMPLE_DATE + STATION_NO
df_all['SAMPLE_KEY'] = (df_all['DWS_NAME'].astype(str) + '|' +
                        df_all['SAMPLE_DATE'].astype(str) + '|' +
                        df_all['STATION_NO'].astype(str))

# Map parameter names to our column names
# Build uppercase lookup dicts
_feat_map_upper = {k.upper(): v for k, v in FEATURE_MAP.items()}
_tgt_map_upper = {k.upper(): v for k, v in TARGET_MAP.items()}
_cl_params_upper = [p.upper() for p in CHLORINE_PARAMS]

def map_param(name):
    name_u = name.upper().strip()
    if name_u in _feat_map_upper:
        return _feat_map_upper[name_u]
    if name_u in _tgt_map_upper:
        return _tgt_map_upper[name_u]
    if name_u in _cl_params_upper:
        return 'Cl2'
    return name

df_all['PARAM_MAPPED'] = df_all['PARAMETER_NAME'].apply(map_param)

# For Cl2, prefer CHLORINE, FREE FIELD; if multiple, take mean
# Pivot to wide format
print("Pivoting to wide format...")
df_wide = df_all.groupby(['SAMPLE_KEY', 'PARAM_MAPPED'])['RESULT_VALUE'].mean().unstack()

print(f"Wide format shape: {df_wide.shape}")
print(f"Columns: {df_wide.columns.tolist()}")

# Check available columns
feature_cols = ['Temp', 'pH', 'DOC', 'Cl2', 'NO2-N', 'NH4-N', 'Br']
target_cols = ['DCAA', 'TCAA', 'HAA5', 'TTHM']  # Main targets

print("\nColumn availability:")
for col in feature_cols + target_cols:
    if col in df_wide.columns:
        n_valid = df_wide[col].notna().sum()
        print(f"  {col}: {n_valid:,} valid values ({n_valid/len(df_wide)*100:.1f}%)")
    else:
        print(f"  {col}: NOT FOUND")

# Keep only rows that have at least one target AND most features
required_targets = ['HAA5']  # Must have HAA5
available_features = [c for c in feature_cols if c in df_wide.columns]
available_targets = [c for c in target_cols if c in df_wide.columns]

print(f"\nAvailable features: {available_features}")
print(f"Available targets: {available_targets}")

# Filter: require HAA5 + at least 4 features
df_filtered = df_wide.dropna(subset=['HAA5'])
df_filtered = df_filtered.dropna(subset=available_features, thresh=len(available_features) - 1)

print(f"After filtering (require HAA5 + most features): {len(df_filtered):,} samples")

# For remaining NaN in features, fill with column median
for col in available_features:
    if df_filtered[col].isna().any():
        median_val = df_filtered[col].median()
        n_fill = df_filtered[col].isna().sum()
        df_filtered[col] = df_filtered[col].fillna(median_val)
        print(f"  Filled {n_fill} NaN in {col} with median={median_val:.3f}")

# Select final columns: targets first (like original data), then features
final_cols = available_targets + available_features
df_final = df_filtered[final_cols].copy()

# Remove outliers (values beyond 3 std from mean)
for col in df_final.columns:
    mean, std = df_final[col].mean(), df_final[col].std()
    mask = (df_final[col] - mean).abs() <= 3 * std
    n_removed = (~mask).sum()
    if n_removed > 0:
        print(f"  Removed {n_removed} outliers from {col}")
    df_final = df_final[mask]

# Remove any remaining NaN
df_final = df_final.dropna()

print(f"\nFinal dataset shape: {df_final.shape}")
print(f"Features: {available_features}")
print(f"Targets: {available_targets}")
print(f"\nDescriptive statistics:")
print(df_final.describe().round(3))

# Save
out_path = os.path.join(OUT_DIR, 'ontario_data.csv')
df_final.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")

# Also save a version carrying water-system and year metadata, so that
# group-based (GroupKFold by system) and chronological validation are possible.
# The SAMPLE_KEY index is "DWS_NAME|SAMPLE_DATE|STATION_NO".
meta = df_final.index.to_series().str.split('|', expand=True)
df_grouped = df_final.copy()
df_grouped['DWS_NAME'] = meta[0].values
df_grouped['SAMPLE_DATE'] = meta[1].values
years = pd.to_datetime(meta[1], errors='coerce').dt.year
if years.isna().any():
    fallback = meta[1].str.extract(r'(\d{4})')[0].astype(float)
    years = years.fillna(fallback)
df_grouped['Year'] = years.astype(int).values
grouped_path = os.path.join(OUT_DIR, 'ontario_data_grouped.csv')
df_grouped.to_csv(grouped_path, index=False)
print(f"Saved grouped version to {grouped_path}")
print(f"  Systems: {df_grouped['DWS_NAME'].nunique()}, Years: {df_grouped['Year'].min()}-{df_grouped['Year'].max()}")
