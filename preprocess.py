# ============================================================
# MASTER PREPROCESSING PIPELINE
# Generates clean master datasets for EDA and ML training
# ============================================================

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MultiLabelBinarizer

# 1. LOAD DATASETS WITH DETERMINISTIC IDENTIFIERS
print("Loading brand datasets...")
nykaa = pd.read_csv("nykaa_campaign_data_with_nulls.csv")
purplle = pd.read_csv("purplle_campaign_data_with_nulls.csv")
tira = pd.read_csv("tira_campaign_data_with_nulls.csv")

# Exact Campaign_ID sequence recovery per brand
nykaa["Campaign_ID"] = [f"NY-CMP-{1000 + i}" for i in range(len(nykaa))]
nykaa["Brand"] = "Nykaa"

purplle["Campaign_ID"] = [f"PU-CMP-{1000 + i}" for i in range(len(purplle))]
purplle["Brand"] = "Purplle"

tira["Campaign_ID"] = [f"TI-CMP-{1000 + i}" for i in range(len(tira))]
tira["Brand"] = "Tira"

# Concatenate all brands
df = pd.concat([nykaa, purplle, tira], ignore_index=True)
df.columns = df.columns.str.strip()
df = df.drop_duplicates().reset_index(drop=True)

# 2. DATE TYPE CONVERSION & IMPUTATION
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce")
df["Date"] = df.groupby("Brand")["Date"].ffill().bfill()

# 3. CATEGORICAL MODAL IMPUTATION (NO 'UNKNOWN' LABELS)
cat_columns = ["Campaign_Type", "Target_Audience", "Language", "Customer_Segment"]
for col in cat_columns:
    mode_val = df[col].mode()[0]
    df[col] = df[col].fillna(mode_val).astype("string")

# Channel_Used: Impute mode grouped by Campaign_Type
channel_mode_by_type = df.groupby("Campaign_Type")["Channel_Used"].apply(
    lambda s: s.mode()[0] if not s.mode().empty else "Instagram"
)
df["Channel_Used"] = df.apply(
    lambda row: channel_mode_by_type.get(row["Campaign_Type"], "Instagram") if pd.isna(row["Channel_Used"]) else row["Channel_Used"],
    axis=1
).astype("string")

# 4. FUNNEL COUNT IMPUTATION & INTEGER TYPE CASTING
count_cols = ["Duration", "Impressions", "Clicks", "Leads", "Conversions"]
for col in count_cols + ["Engagement_Score"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    group_medians = df.groupby(["Brand", "Campaign_Type"])[col].transform("median")
    df[col] = df[col].fillna(group_medians).fillna(df[col].median())

for col in count_cols:
    df[col] = df[col].round().astype("int64")

df["Engagement_Score"] = df["Engagement_Score"].round(2)

# 5. ALGEBRAIC RECOVERY OF FINANCIAL ATTRIBUTES
# Formula: Revenue = Acquisition_Cost * Conversions * (1 + ROI)
df["Acquisition_Cost"] = pd.to_numeric(df["Acquisition_Cost"], errors="coerce")
df["ROI"] = pd.to_numeric(df["ROI"], errors="coerce")
df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")

# Step 5a: Cross-impute Revenue when CAC, Conv, ROI exist
mask_rev = df["Revenue"].isna() & df["ROI"].notna() & df["Acquisition_Cost"].notna()
df.loc[mask_rev, "Revenue"] = df.loc[mask_rev, "Acquisition_Cost"] * df.loc[mask_rev, "Conversions"] * (1 + df.loc[mask_rev, "ROI"])

# Step 5b: Cross-impute CAC when Revenue, Conv, ROI exist
mask_cac = df["Acquisition_Cost"].isna() & df["Revenue"].notna() & df["ROI"].notna()
df.loc[mask_cac, "Acquisition_Cost"] = df.loc[mask_cac, "Revenue"] / (df.loc[mask_cac, "Conversions"] * (1 + df.loc[mask_cac, "ROI"]))

# Step 5c: Cross-impute ROI when Revenue, CAC, Conv exist
mask_roi = df["ROI"].isna() & df["Revenue"].notna() & df["Acquisition_Cost"].notna()
denom = df.loc[mask_roi, "Acquisition_Cost"] * df.loc[mask_roi, "Conversions"]
df.loc[mask_roi, "ROI"] = (df.loc[mask_roi, "Revenue"] - denom) / np.where(denom != 0, denom, 1.0)

# Step 5d: Fill any remaining financial nulls with brand-level medians
for col in ["Acquisition_Cost", "ROI"]:
    df[col] = df[col].fillna(df.groupby("Brand")[col].transform("median"))

mask_rev_final = df["Revenue"].isna()
df.loc[mask_rev_final, "Revenue"] = df.loc[mask_rev_final, "Acquisition_Cost"] * df.loc[mask_rev_final, "Conversions"] * (1 + df.loc[mask_rev_final, "ROI"])

df["Acquisition_Cost"] = df["Acquisition_Cost"].round(2)
df["ROI"] = df["ROI"].round(2)
df["Revenue"] = df["Revenue"].round(2)

# 6. FEATURE ENGINEERING: TARGETS & FINANCIAL METRICS
df["Total_Spend"] = (df["Acquisition_Cost"] * df["Conversions"]).round(2)
df["Profit"] = (df["Revenue"] - df["Total_Spend"]).round(2)
df["Profit_Flag"] = np.where(df["ROI"] >= 1.0, 1, 0).astype("int64")

# 7. FEATURE ENGINEERING: FUNNEL RATIOS & UNIT COSTS
df["CTR"] = np.where(df["Impressions"] > 0, (df["Clicks"] / df["Impressions"]) * 100, 0).round(4)
df["Click_to_Lead_Rate"] = np.where(df["Clicks"] > 0, (df["Leads"] / df["Clicks"]) * 100, 0).clip(0, 100).round(4)
df["Lead_to_Conv_Rate"] = np.where(df["Leads"] > 0, (df["Conversions"] / df["Leads"]) * 100, 0).clip(0, 100).round(4)
df["Conversion_Rate"] = np.where(df["Clicks"] > 0, (df["Conversions"] / df["Clicks"]) * 100, 0).clip(0, 100).round(4)

df["CPC"] = (df["Total_Spend"] / (df["Clicks"] + 1)).round(2)
df["CPL"] = (df["Total_Spend"] / (df["Leads"] + 1)).round(2)
df["Cost_per_Impression"] = (df["Total_Spend"] / (df["Impressions"] + 1)).round(4)
df["Engagement_per_Click"] = (df["Engagement_Score"] * df["Clicks"]).round(2)
df["Engagement_per_Conv"] = (df["Engagement_Score"] * df["Conversions"]).round(2)

# 8. MULTI-LABEL ENCODING FOR CHANNELS
df["Channel_List"] = df["Channel_Used"].apply(
    lambda x: [] if pd.isna(x) or str(x).strip() in ["None", "Unknown", "", "nan"] else [c.strip() for c in str(x).split(",") if c.strip()]
)
df["Channel_Count"] = df["Channel_List"].apply(len).astype("int64")

mlb = MultiLabelBinarizer()
channel_encoded = mlb.fit_transform(df["Channel_List"])
channel_cols = ["Channel_" + c.replace(" ", "_") for c in mlb.classes_]
channel_df = pd.DataFrame(channel_encoded, columns=channel_cols, index=df.index, dtype="int64")

df_final = pd.concat([df.drop(columns=["Channel_List"]), channel_df], axis=1)

# Chronological sorting
df_final = df_final.sort_values(by=["Brand", "Date"]).reset_index(drop=True)

# 9. EXPORT MASTER FILES
# File A: Base clean dataset for EDA
df_final.drop(columns=channel_cols).to_csv("marketing_campaign_cleaned_master.csv", index=False)

# File B: Full feature-engineered dataset for ML training
df_final.to_csv("marketing_campaign_preprocessed_ready.csv", index=False)
joblib.dump(mlb, "mlb_channel_encoder.pkl")

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 60)
print(f"Total Rows              : {len(df_final):,}")
print(f"Total Columns           : {df_final.shape[1]}")
print(f"Missing Values Remaining: {df_final.isnull().sum().sum()}")
print(f"Infinite Values         : {np.isinf(df_final.select_dtypes(include=np.number)).sum().sum()}")
print("Saved clean master for EDA: marketing_campaign_cleaned_master.csv")
print("Saved feature-engineered dataset for ML: marketing_campaign_preprocessed_ready.csv")