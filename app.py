# ============================================================
# STREAMLIT APPLICATION: MULTI-BRAND MARKETING INTELLIGENCE
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# 1. PAGE SETUP
st.set_page_config(
    page_title="Marketing Campaign Intelligence System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. CACHED MODEL & ARTIFACT LOADERS
@st.cache_resource
def load_models():
    reg_model = joblib.load("regression_revenue_scaled.pkl")
    clf_model = joblib.load("classification_profit_unscaled.pkl")
    mlb = joblib.load("mlb_channel_encoder.pkl")
    reg_features = joblib.load("reg_feature_names.pkl")
    clf_features = joblib.load("clf_feature_names.pkl")
    return reg_model, clf_model, mlb, reg_features, clf_features

@st.cache_data
def load_data():
    return pd.read_csv("marketing_campaign_cleaned_master.csv")

try:
    reg_pipeline, clf_pipeline, mlb, reg_features, clf_features = load_models()
    df_master = load_data()
except Exception as err:
    st.error(f"⚠️ Error loading artifacts: {err}")
    st.info("Make sure you have run your preprocessing and training scripts so that all .pkl and .csv files exist in the same folder.")
    st.stop()

# 3. HEADER
st.title("📊 Multi-Brand Marketing Performance & ROI Predictor")
st.markdown("Predict campaign revenue, validate profit outcomes, and explore marketing analytics across **Nykaa**, **Purplle**, and **Tira**.")

tab_pred, tab_eda, tab_docs = st.tabs(["🚀 Live Prediction Engine", "📊 Portfolio EDA Dashboard", "📖 Project Documentation"])

# ============================================================
# TAB 1: LIVE PREDICTION ENGINE
# ============================================================
with tab_pred:
    st.subheader("Campaign Configuration & Funnel Inputs")
    
    with st.sidebar:
        st.header("⚙️ Campaign Parameters")
        brand = st.selectbox("Select Brand", ["Nykaa", "Purplle", "Tira"])
        campaign_type = st.selectbox("Strategy Type", ["Paid Ads", "Social Media", "Email", "Influencer", "SEO"])
        target_audience = st.selectbox("Target Audience", ["College Students", "Tier 2 City Customers", "Youth", "Working Women", "Premium Shoppers"])
        customer_segment = st.selectbox("Customer Segment", ["College Students", "Tier 2 City Customers", "Youth", "Working Women", "Premium Shoppers"])
        language = st.selectbox("Language", ["Hindi", "English", "Tamil", "Bengali"])
        
        st.subheader("Channels Deployed")
        channels_selected = st.multiselect(
            "Select Marketing Channels",
            options=list(mlb.classes_),
            default=["Google", "Instagram"]
        )
        
        st.subheader("Budget & Funnel Estimations")
        duration = st.slider("Duration (Days)", min_value=1, max_value=60, value=21, step=1)
        impressions = st.number_input("Impressions", min_value=1000, max_value=500000, value=65000, step=1000)
        clicks = st.number_input("Clicks", min_value=50, max_value=100000, value=5500, step=100)
        leads = st.number_input("Leads Generated", min_value=10, max_value=50000, value=2200, step=50)
        conversions = st.number_input("Conversions", min_value=1, max_value=20000, value=1250, step=25)
        acq_cost = st.number_input("Acquisition Cost per Conversion (₹)", min_value=5.0, max_value=5000.0, value=185.50, step=5.0)
        engagement_score = st.slider("Engagement Score", min_value=1.0, max_value=40.0, value=16.80, step=0.1)

    # Derived Feature Computations
    total_spend = acq_cost * conversions
    ctr = (clicks / impressions) * 100 if impressions > 0 else 0.0
    click_to_lead_rate = min((leads / clicks) * 100 if clicks > 0 else 0.0, 100.0)
    lead_to_conv_rate = min((conversions / leads) * 100 if leads > 0 else 0.0, 100.0)
    conversion_rate = min((conversions / clicks) * 100 if clicks > 0 else 0.0, 100.0)
    cpc = total_spend / (clicks + 1)
    cpl = total_spend / (leads + 1)
    cost_per_impression = total_spend / (impressions + 1)
    eng_per_click = engagement_score * clicks
    eng_per_conv = engagement_score * conversions
    channel_count = len(channels_selected)

    # Multi-label Encoding
    channel_binary = mlb.transform([channels_selected])[0]
    channel_dict = {f"Channel_{cls.replace(' ', '_')}": channel_binary[i] for i, cls in enumerate(mlb.classes_)}

    # Input dictionary
    base_dict = {
        "Brand": brand,
        "Campaign_Type": campaign_type,
        "Target_Audience": target_audience,
        "Language": language,
        "Customer_Segment": customer_segment,
        "Duration": duration,
        "Impressions": impressions,
        "Clicks": clicks,
        "Leads": leads,
        "Conversions": conversions,
        "Acquisition_Cost": acq_cost,
        "Engagement_Score": engagement_score,
        "Total_Spend": total_spend,
        "CTR": ctr,
        "Click_to_Lead_Rate": click_to_lead_rate,
        "Lead_to_Conv_Rate": lead_to_conv_rate,
        "Conversion_Rate": conversion_rate,
        "CPC": cpc,
        "CPL": cpl,
        "Cost_per_Impression": cost_per_impression,
        "Engagement_per_Click": eng_per_click,
        "Engagement_per_Conv": eng_per_conv,
        "Channel_Count": channel_count,
        **channel_dict
    }

    # Summary Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Marketing Spend", f"₹{total_spend:,.2f}")
    m2.metric("Click-Through Rate (CTR)", f"{ctr:.2f}%")
    m3.metric("Lead Conversion Rate", f"{lead_to_conv_rate:.2f}%")
    m4.metric("Cost per Click (CPC)", f"₹{cpc:.2f}")

    st.markdown("---")

    if st.button("🔮 Forecast Revenue & Predict Profitability", type="primary", use_container_width=True):
        # 1. Predict Revenue
        est_roi_prior = 1.25
        reg_input = pd.DataFrame([{**base_dict, "ROI": est_roi_prior}])[reg_features]
        predicted_revenue = float(reg_pipeline.predict(reg_input)[0])

        # 2. Predict Profit Flag
        clf_input = pd.DataFrame([{**base_dict, "Revenue": predicted_revenue}])[clf_features]
        predicted_flag = int(clf_pipeline.predict(clf_input)[0])
        clf_probs = clf_pipeline.predict_proba(clf_input)[0]

        # Financial Calculations
        net_profit = predicted_revenue - total_spend
        forecast_roi = (net_profit / total_spend) if total_spend > 0 else 0.0

        # Output Metric Cards
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.markdown("#### Expected Revenue (Regression)")
            st.metric("Predicted Revenue", f"₹{predicted_revenue:,.2f}")
            st.caption(f"Estimated Net Profit: ₹{net_profit:,.2f}")

        with col_r2:
            st.markdown("#### Outcome Status (Classification)")
            if predicted_flag == 1:
                st.success("✅ **PROFITABLE CAMPAIGN (ROI ≥ 1.0)**")
                st.write(f"Confidence: **{clf_probs[1]*100:.1f}%**")
            else:
                st.error("⚠️ **LOSS / SUB-OPTIMAL (ROI < 1.0)**")
                st.write(f"Confidence: **{clf_probs[0]*100:.1f}%**")

        with col_r3:
            st.markdown("#### Projected Return on Investment")
            st.metric("Projected ROI", f"{forecast_roi:.2f}x")
            st.caption("Benchmark Target: ≥ 1.0x")

        # Visual Breakdown Waterfall
        st.markdown("#### 💰 Financial Breakdown Waterfall")
        fig_wf = go.Figure(go.Waterfall(
            name="Financials",
            orientation="v",
            measure=["relative", "relative", "total"],
            x=["Gross Revenue", "Total Marketing Spend", "Net Profit"],
            textposition="outside",
            text=[f"₹{predicted_revenue:,.0f}", f"-₹{total_spend:,.0f}", f"₹{net_profit:,.0f}"],
            y=[predicted_revenue, -total_spend, net_profit],
            decreasing={"marker": {"color": "#e74c3c"}},
            increasing={"marker": {"color": "#2ecc71"}},
            totals={"marker": {"color": "#3498db"}}
        ))
        fig_wf.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_wf, use_container_width=True)

# ============================================================
# TAB 2: EXPLORATORY DATA ANALYTICS (EDA)
# ============================================================
with tab2:
    st.subheader("Multi-Brand Portfolio Performance & Trend Insights")
    
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        # Brand Financial Comparison
        brand_agg = df_master.groupby("Brand")[["Revenue", "Total_Spend", "Profit"]].mean().reset_index()
        fig_b = px.bar(
            brand_agg.melt(id_vars="Brand", value_vars=["Revenue", "Total_Spend", "Profit"]),
            x="Brand", y="value", color="variable", barmode="group",
            title="Average Financial Returns across Brands",
            labels={"value": "Amount (₹)", "variable": "Metric"},
            color_discrete_map={"Revenue": "#2b5c8f", "Total_Spend": "#d95f02", "Profit": "#2ca02c"}
        )
        st.plotly_chart(fig_b, use_container_width=True)

    with col_e2:
        # Strategy ROI Comparison
        strat_agg = df_master.groupby("Campaign_Type")["ROI"].mean().reset_index().sort_values(by="ROI", ascending=False)
        fig_s = px.bar(
            strat_agg, x="Campaign_Type", y="ROI", color="ROI",
            title="Average Campaign ROI by Strategy Type",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_s, use_container_width=True)

    col_e3, col_e4 = st.columns(2)
    with col_e3:
        # CTR Distribution Plot
        fig_h = px.histogram(
            df_master, x="CTR", color="Profit_Flag", barmode="overlay",
            title="Click-Through Rate (CTR) Distribution: Profit vs Loss",
            labels={"Profit_Flag": "Profit Outcome"},
            color_discrete_map={0: "#e74c3c", 1: "#2ecc71"}
        )
        st.plotly_chart(fig_h, use_container_width=True)

    with col_e4:
        # Overall Profit / Loss Pie Chart
        p_dist = df_master["Profit_Flag"].value_counts(normalize=True).reset_index()
        p_dist["Outcome"] = p_dist["Profit_Flag"].map({1: "Profit (ROI ≥ 1.0)", 0: "Loss (ROI < 1.0)"})
        fig_p = px.pie(
            p_dist, values="proportion", names="Outcome",
            title="Overall Dataset Profit/Loss Ratio",
            color="Outcome",
            color_discrete_map={"Profit (ROI ≥ 1.0)": "#2ecc71", "Loss (ROI < 1.0)": "#e74c3c"}
        )
        st.plotly_chart(fig_p, use_container_width=True)

# ============================================================
# TAB 3: PROJECT DOCUMENTATION
# ============================================================
with tab3:
    st.markdown("""
    ### Technical Documentation & Pipeline Overview
    
    * **Data Preprocessing & Cleaning:**
      - Resolved ~8,000 missing values using algebraic financial recovery[cite: 1]:
        $$\\text{Revenue} = (\\text{Acquisition\\_Cost} \\times \\text{Conversions}) \\times (1 + \\text{ROI})$$
      - Modal and grouped median imputations across non-financial columns[cite: 1].
      - Multi-label binarization for marketing channels[cite: 1].
      
    * **Machine Learning Pipelines:**
      - **Regression:** `HistGradientBoostingRegressor` with `StandardScaler` ($R^2 = 0.9871$)[cite: 1].
      - **Classification:** `HistGradientBoostingClassifier` without scaling ($\text{Accuracy} = 99.22\%$, strictly excluding `ROI` to prevent data leakage)[cite: 1].
    """)
