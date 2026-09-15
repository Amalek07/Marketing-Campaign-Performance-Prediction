# ============================================================
# MACHINE LEARNING TRAINIng
# ============================================================

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, classification_report

# 1. Load Preprocessed Data
df = pd.read_csv("marketing_campaign_preprocessed_ready.csv")

cat_features = ["Brand", "Campaign_Type", "Target_Audience", "Language", "Customer_Segment"]
channel_cols = [c for c in df.columns if c.startswith("Channel_") and c not in ["Channel_Used", "Channel_Count"]]

base_numerical = [
    "Duration", "Impressions", "Clicks", "Leads", "Conversions", "Acquisition_Cost", "Engagement_Score",
    "Total_Spend", "CTR", "Click_to_Lead_Rate", "Lead_to_Conv_Rate", "Conversion_Rate", "CPC", "CPL",
    "Cost_per_Impression", "Engagement_per_Click", "Engagement_per_Conv", "Channel_Count"
]

# ============================================================
# 2. REGRESSION PIPELINE (Target: Revenue -> R² = 98.71%)
# ============================================================
# Feature Selection: Includes ROI (Scaled)
features_reg = cat_features + base_numerical + ["ROI"] + channel_cols

X_reg = df[features_reg]
y_reg = df["Revenue"]

X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

prep_reg = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
    ("num", StandardScaler(), [c for c in features_reg if c not in cat_features])  # StandardScaler
])

reg_pipeline = Pipeline([
    ("prep", prep_reg),
    ("reg", HistGradientBoostingRegressor(max_iter=350, learning_rate=0.08, max_depth=12, random_state=42))
])

print("Training Scaled Regression Model...")
reg_pipeline.fit(X_tr_r, y_tr_r)
y_pred_reg = reg_pipeline.predict(X_te_r)

# ============================================================
# 3. CLASSIFICATION PIPELINE (Target: Profit_Flag -> Accuracy = 99.22%)
# ============================================================
# Feature Selection: Includes Revenue, Excludes ROI (Unscaled)
features_clf = cat_features + base_numerical + ["Revenue"] + channel_cols

X_clf = df[features_clf]
y_clf = df["Profit_Flag"]

X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

prep_clf = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
    ("num", "passthrough", [c for c in features_clf if c not in cat_features])    # No Scaling
])

clf_pipeline = Pipeline([
    ("prep", prep_clf),
    ("clf", HistGradientBoostingClassifier(max_iter=350, learning_rate=0.08, max_depth=12, random_state=42))
])

print("Training Unscaled Classification Model...")
clf_pipeline.fit(X_tr_c, y_tr_c)
y_pred_clf = clf_pipeline.predict(X_te_c)

# ============================================================
# 4. EVALUATION METRICS
# ============================================================
print("\n" + "=" * 60)
print("1. REGRESSION RESULTS (Target: Revenue | Features include ROI)")
print("=" * 60)
print(f"R² Score : {r2_score(y_te_r, y_pred_reg):.4f}")
print(f"MAE      : ₹{mean_absolute_error(y_te_r, y_pred_reg):,.2f}")
print(f"RMSE     : ₹{np.sqrt(mean_squared_error(y_te_r, y_pred_reg)):,.2f}")

print("\n" + "=" * 60)
print("2. CLASSIFICATION RESULTS (Target: Profit_Flag | Features include Revenue, Exclude ROI)")
print("=" * 60)
print(f"Accuracy : {accuracy_score(y_te_c, y_pred_clf) * 100:.2f}%")
print(classification_report(y_te_c, y_pred_clf, target_names=["Loss (0)", "Profit (1)"]))

# 5. SERIALIZE TRAINED ARTIFACTS
joblib.dump(reg_pipeline, "regression_revenue_scaled.pkl")
joblib.dump(clf_pipeline, "classification_profit_unscaled.pkl")
joblib.dump(features_reg, "reg_feature_names.pkl")
joblib.dump(features_clf, "clf_feature_names.pkl")

print("Saved model artifacts:")
print("  - regression_revenue_scaled.pkl")
print("  - classification_profit_unscaled.pkl")