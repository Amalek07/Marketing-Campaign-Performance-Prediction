# ============================================================
# EXPLORATORY DATA ANALYSIS (EDA) & VISUALIZATION SUITE
# Multi-Brand Marketing Campaign Performance
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. LOAD CLEANED MASTER DATASETS
df = pd.read_csv("marketing_campaign_cleaned_master.csv")
df_ready = pd.read_csv("marketing_campaign_preprocessed_ready.csv")

# 2. NUMERICAL SUMMARY & DESCRIPTIVE STATISTICS
print("=" * 65)
print("EXPLORATORY DATA ANALYSIS SUMMARY")
print("=" * 65)
print(f"Total Combined Records : {len(df):,}")
print(f"Unique Brands Analyzed : {df['Brand'].nunique()} ({list(df['Brand'].unique())})")

print("\n--- BRAND-WISE PERFORMANCE SUMMARY ---")
brand_perf = df.groupby("Brand").agg(
    Campaign_Count=("Campaign_ID", "count"),
    Avg_Revenue=("Revenue", "mean"),
    Avg_Spend=("Total_Spend", "mean"),
    Avg_Profit=("Profit", "mean"),
    Avg_ROI=("ROI", "mean"),
    Profitable_Ratio=("Profit_Flag", "mean")
).round(2)
print(brand_perf)

print("\n--- CAMPAIGN STRATEGY EFFECTIVENESS ---")
strategy_perf = df.groupby("Campaign_Type").agg(
    Avg_Revenue=("Revenue", "mean"),
    Avg_ROI=("ROI", "mean"),
    Avg_Conversion_Rate=("Conversion_Rate", "mean"),
    Profitable_Ratio=("Profit_Flag", "mean")
).sort_values(by="Avg_ROI", ascending=False).round(2)
print(strategy_perf)

# 3. GENERATE VISUALIZATION SUITE (6-PANEL FIGURE)
plt.figure(figsize=(20, 15))
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

# Plot 1: Brand Financial Performance Comparison
plt.subplot(2, 3, 1)
brand_summary = df.groupby("Brand")[["Revenue", "Total_Spend", "Profit"]].mean().reset_index()
brand_melted = brand_summary.melt(
    id_vars="Brand", 
    value_vars=["Revenue", "Total_Spend", "Profit"], 
    var_name="Metric", 
    value_name="Amount"
)
sns.barplot(data=brand_melted, x="Brand", y="Amount", hue="Metric", palette=["#2b5c8f", "#d95f02", "#2ca02c"])
plt.title("1. Brand Financial Performance (Averages)", fontsize=12, fontweight="bold")
plt.ylabel("Amount (INR)", fontsize=10)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"INR {int(x):,}"))
plt.legend(frameon=True)

# Plot 2: Channel vs Campaign Type ROI Heatmap
plt.subplot(2, 3, 2)
channel_cols = [c for c in df_ready.columns if c.startswith("Channel_") and c != "Channel_Count"]
channel_roi = {}
for c in channel_cols:
    c_name = c.replace("Channel_", "")
    channel_roi[c_name] = df_ready[df_ready[c] == 1].groupby("Campaign_Type")["ROI"].mean()
heatmap_df = pd.DataFrame(channel_roi)
sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Mean ROI'})
plt.title("2. Channel vs Campaign Type ROI Heatmap", fontsize=12, fontweight="bold")
plt.xlabel("Marketing Channel", fontsize=10)
plt.ylabel("Campaign Type", fontsize=10)

# Plot 3: Marketing Funnel CTR Density: Profit vs Loss
plt.subplot(2, 3, 3)
sns.kdeplot(
    data=df, 
    x="CTR", 
    hue="Profit_Flag", 
    common_norm=False, 
    fill=True, 
    palette={0: "#e74c3c", 1: "#27ae60"}, 
    alpha=0.45
)
plt.title("3. CTR Density: Profit (1) vs Loss (0)", fontsize=12, fontweight="bold")
plt.xlabel("Click-Through Rate (%)", fontsize=10)
plt.xlim(0, df["CTR"].quantile(0.99))
plt.legend(title="Outcome", labels=["Profit (ROI >= 1.0)", "Loss (ROI < 1.0)"])

# Plot 4: Average Conversion Rate by Customer Segment
plt.subplot(2, 3, 4)
segment_stats = df.groupby("Customer_Segment").agg(
    Avg_Conv_Rate=("Conversion_Rate", "mean")
).reset_index()
sns.barplot(data=segment_stats, x="Customer_Segment", y="Avg_Conv_Rate", palette="viridis")
plt.title("4. Avg Conversion Rate by Customer Segment", fontsize=12, fontweight="bold")
plt.ylabel("Conversion Rate (%)", fontsize=10)
plt.xlabel("Customer Segment", fontsize=10)
plt.xticks(rotation=20)

# Plot 5: Correlation Matrix across Funnel & Financial Attributes
plt.subplot(2, 3, 5)
corr_features = [
    "Impressions", "Clicks", "Leads", "Conversions", 
    "Acquisition_Cost", "Total_Spend", "Revenue", "ROI", "Engagement_Score"
]
corr_mat = df[corr_features].corr()
sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, annot_kws={"size": 8})
plt.title("5. Feature Correlation Heatmap", fontsize=12, fontweight="bold")
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(fontsize=9)

# Plot 6: Profit vs Loss Class Proportions
plt.subplot(2, 3, 6)
profit_dist = df["Profit_Flag"].value_counts(normalize=True).rename({1: "Profit (ROI >= 1.0)", 0: "Loss (ROI < 1.0)"})
colors = ["#2ecc71", "#e74c3c"]
plt.pie(
    profit_dist.values, 
    labels=profit_dist.index, 
    autopct="%1.1f%%", 
    colors=colors, 
    startangle=140, 
    explode=(0.04, 0)
)
plt.title("6. Overall Campaign Outcome Distribution", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig("eda_marketing_master_suite.png", dpi=300)
plt.show()

print("\nVisualization suite saved to: eda_marketing_master_suite.png")