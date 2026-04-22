# =============================================================================
# MOBIUS — INTERACTIVE RISK DASHBOARD
# Pounce Capital Advisors
# Georgia State University | Finance Innovation Showcase
#
# HOW TO RUN:
#   1. Run mobius_pipeline.py first to train all models
#   2. Open terminal in VS Code
#   3. streamlit run mobius_streamlit_dashboard.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (recall_score, precision_score, f1_score,
                             roc_auc_score, accuracy_score,
                             confusion_matrix, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title  = "Mobius | Pounce Capital Advisors",
    page_icon   = "📊",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #F4F6F9; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1A2744;
        color: white;
    }
    section[data-testid="stSidebar"] * { color: white !important; }
    section[data-testid="stSidebar"] .stSelectbox label { color: #ECF0F1 !important; }
    section[data-testid="stSidebar"] .stSlider label { color: #ECF0F1 !important; }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 10px;
    }
    .metric-value { font-size: 32px; font-weight: bold; color: #1A2744; }
    .metric-label { font-size: 13px; color: #7F8C8D; margin-top: 4px; }

    /* Stoplight cards */
    .green-card  { background:#2ECC71; border-radius:10px; padding:20px;
                   text-align:center; color:white; font-weight:bold; }
    .yellow-card { background:#F39C12; border-radius:10px; padding:20px;
                   text-align:center; color:white; font-weight:bold; }
    .red-card    { background:#E74C3C; border-radius:10px; padding:20px;
                   text-align:center; color:white; font-weight:bold; }

    /* Score result */
    .score-green  { background:#2ECC71; border-radius:12px; padding:30px;
                    text-align:center; color:white; }
    .score-yellow { background:#F39C12; border-radius:12px; padding:30px;
                    text-align:center; color:white; }
    .score-red    { background:#E74C3C; border-radius:12px; padding:30px;
                    text-align:center; color:white; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { background-color: white;
                                        border-radius: 10px; padding: 5px; }
    .stTabs [data-baseweb="tab"] { color: #1A2744; font-weight: 500; }

    /* Headers */
    h1, h2, h3 { color: #1A2744; }
    .section-header { color: #1A2744; font-weight: bold;
                      border-bottom: 2px solid #3498DB;
                      padding-bottom: 5px; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD & PREPARE DATA (cached so it only runs once)
# =============================================================================

@st.cache_resource
def load_and_train():
    """Loads data, engineers features, trains all models. Cached."""

    import os
    # Works locally and on Streamlit Cloud
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "american_bankruptcy.csv")
    df = pd.read_csv(csv_path)

    col_map = {
        "X1":"Current_Assets","X2":"Cost_of_Goods_Sold",
        "X3":"Depreciation_Amortization","X4":"EBITDA",
        "X5":"Inventory","X6":"Net_Income","X7":"Total_Receivables",
        "X8":"Market_Value","X9":"Net_Sales","X10":"Total_Assets",
        "X11":"Total_Long_Term_Debt","X12":"EBIT","X13":"Gross_Profit",
        "X14":"Total_Current_Liabilities","X15":"Retained_Earnings",
        "X16":"Total_Revenue","X17":"Total_Liabilities",
        "X18":"Total_Operating_Expense",
    }
    df = df.rename(columns=col_map)
    df["Bankrupt"] = (df["status_label"] == "failed").astype(int)

    def safe_div(n, d): return n / d.replace(0, np.nan)

    df["Debt_Ratio"]              = safe_div(df["Total_Liabilities"],       df["Total_Assets"])
    df["LT_Debt_Ratio"]           = safe_div(df["Total_Long_Term_Debt"],    df["Total_Assets"])
    df["Current_Ratio"]           = safe_div(df["Current_Assets"],          df["Total_Current_Liabilities"])
    df["ROA"]                     = safe_div(df["Net_Income"],              df["Total_Assets"])
    df["Net_Profit_Margin"]       = safe_div(df["Net_Income"],              df["Total_Revenue"])
    df["EBITDA_Margin"]           = safe_div(df["EBITDA"],                  df["Total_Revenue"])
    df["Retained_Earnings_Ratio"] = safe_div(df["Retained_Earnings"],       df["Total_Assets"])
    df["Operating_Margin"]        = safe_div(df["EBIT"],                    df["Total_Revenue"])
    df["Asset_Turnover"]          = safe_div(df["Total_Revenue"],           df["Total_Assets"])

    raw_features = [
        "Current_Assets","Cost_of_Goods_Sold","Depreciation_Amortization",
        "EBITDA","Inventory","Net_Income","Total_Receivables","Market_Value",
        "Net_Sales","Total_Assets","Total_Long_Term_Debt","EBIT","Gross_Profit",
        "Total_Current_Liabilities","Retained_Earnings","Total_Revenue",
        "Total_Liabilities","Total_Operating_Expense"
    ]
    ratio_features = [
        "Debt_Ratio","LT_Debt_Ratio","Current_Ratio","ROA",
        "Net_Profit_Margin","EBITDA_Margin","Retained_Earnings_Ratio",
        "Operating_Margin","Asset_Turnover"
    ]
    all_features = raw_features + ratio_features

    category_map = {
        "Current_Assets":"Liquidity","Total_Current_Liabilities":"Liquidity",
        "Current_Ratio":"Liquidity","Total_Liabilities":"Leverage",
        "Total_Long_Term_Debt":"Leverage","Market_Value":"Leverage",
        "Total_Assets":"Leverage","Debt_Ratio":"Leverage","LT_Debt_Ratio":"Leverage",
        "Net_Income":"Profitability","EBITDA":"Profitability","EBIT":"Profitability",
        "Gross_Profit":"Profitability","Retained_Earnings":"Profitability",
        "Depreciation_Amortization":"Profitability","ROA":"Profitability",
        "Net_Profit_Margin":"Profitability","EBITDA_Margin":"Profitability",
        "Retained_Earnings_Ratio":"Profitability","Operating_Margin":"Profitability",
        "Cost_of_Goods_Sold":"Efficiency","Inventory":"Efficiency",
        "Total_Receivables":"Efficiency","Net_Sales":"Efficiency",
        "Total_Revenue":"Efficiency","Total_Operating_Expense":"Efficiency",
        "Asset_Turnover":"Efficiency",
    }

    df_model = df[all_features + ["Bankrupt","year"]].copy()
    df_model = df_model.replace([np.inf,-np.inf], np.nan)
    df_model = df_model.fillna(df_model.median(numeric_only=True))

    X = df_model[all_features]
    y = df_model["Bankrupt"]
    year = df_model["year"]

    X_train = X[year<=2013];     y_train = y[year<=2013]
    X_val   = X[(year>=2014)&(year<=2015)]
    y_val   = y[(year>=2014)&(year<=2015)]
    X_test  = X[year>=2016];     y_test  = y[year>=2016]

    smote = SMOTE(k_neighbors=5, random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_bal)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    def find_thresh(y_true, y_prob):
        best_t, best_r = 0.10, 0
        for t in np.arange(0.10, 0.91, 0.01):
            r = recall_score(y_true, (y_prob>=t).astype(int), zero_division=0)
            if r > best_r: best_r, best_t = r, t
        return best_t

    # LASSO
    lasso = LogisticRegression(penalty="l1", solver="liblinear",
                               C=0.05, max_iter=2000, random_state=42)
    lasso.fit(X_train_s, y_train_bal)
    lasso_val_prob = lasso.predict_proba(X_val_s)[:,1]
    lasso_thresh   = find_thresh(y_val, lasso_val_prob)
    lasso_yellow_t = np.percentile(lasso_val_prob, 70)
    lasso_red_t    = np.percentile(lasso_val_prob, 90)
    lasso_prob     = lasso.predict_proba(X_test_s)[:,1]
    lasso_pred     = (lasso_prob >= lasso_thresh).astype(int)
    lasso_coef     = pd.Series(lasso.coef_[0], index=all_features)

    # Neural Network — fixed params, fast startup
    nn_model = MLPClassifier(
        hidden_layer_sizes=(15, 10), alpha=0.01,
        max_iter=200, random_state=42, early_stopping=True)
    nn_model.fit(X_train_s, y_train_bal)
    nn_val_prob   = nn_model.predict_proba(X_val_s)[:,1]
    nn_yellow_t   = np.percentile(nn_val_prob, 70)
    nn_red_t      = np.percentile(nn_val_prob, 90)
    nn_prob       = nn_model.predict_proba(X_test_s)[:,1]
    nn_thresh     = 0.10
    nn_pred       = (nn_prob >= nn_thresh).astype(int)
    nn_best_params = {"hidden_layer_sizes": (15, 10), "alpha": 0.01}

    # Decision Tree — fixed params, fast startup
    tree_model = DecisionTreeClassifier(
        max_depth=4, min_samples_split=20, min_samples_leaf=10, random_state=42)
    tree_model.fit(X_train_bal, y_train_bal)
    tree_val_prob = tree_model.predict_proba(X_val)[:,1]
    tree_yellow_t = np.percentile(tree_val_prob, 70)
    tree_red_t    = np.percentile(tree_val_prob, 90)
    tree_prob     = tree_model.predict_proba(X_test)[:,1]
    tree_thresh   = 0.10
    tree_pred     = (tree_prob >= tree_thresh).astype(int)
    tree_best_params = {"max_depth": 4, "min_samples_split": 20, "min_samples_leaf": 10}

    # Financial category importance
    coef_abs = lasso_coef.abs()
    coef_abs = coef_abs[coef_abs > 0]
    cat_imp = {}
    for feat, imp in coef_abs.items():
        cat = category_map.get(feat, "Other")
        cat_imp[cat] = cat_imp.get(cat, 0) + imp
    total_imp = sum(cat_imp.values())
    cat_df = pd.DataFrame([
        {"Category":k, "Pct":v/total_imp*100}
        for k,v in cat_imp.items()
    ]).sort_values("Pct", ascending=False)

    return {
        "X_test": X_test, "y_test": y_test,
        "X_test_s": X_test_s,
        "scaler": scaler,
        "lasso": lasso, "lasso_prob": lasso_prob,
        "lasso_pred": lasso_pred, "lasso_thresh": lasso_thresh,
        "lasso_yellow_t": lasso_yellow_t, "lasso_red_t": lasso_red_t,
        "lasso_coef": lasso_coef,
        "nn_model": nn_model, "nn_prob": nn_prob,
        "nn_pred": nn_pred, "nn_thresh": nn_thresh,
        "nn_yellow_t": nn_yellow_t, "nn_red_t": nn_red_t,
        "tree_model": tree_model, "tree_prob": tree_prob,
        "tree_pred": tree_pred, "tree_thresh": tree_thresh,
        "tree_yellow_t": tree_yellow_t, "tree_red_t": tree_red_t,
        "tree_imp": pd.Series(tree_model.feature_importances_, index=all_features),
        "cat_df": cat_df, "all_features": all_features,
        "category_map": category_map,
        "nn_best_params": nn_best_params,
        "tree_best_params": tree_best_params,
    }

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_model_data(data, model_name):
    """Returns prob, pred, thresh, yellow_t, red_t for selected model."""
    key = {"LASSO":"lasso", "Neural Network":"nn", "Decision Tree":"tree"}[model_name]
    return (data[f"{key}_prob"], data[f"{key}_pred"],
            data[f"{key}_thresh"], data[f"{key}_yellow_t"], data[f"{key}_red_t"])

def apply_stoplight(prob, yellow_t, red_t):
    zones = []
    for p in prob:
        if p >= red_t:    zones.append("RED")
        elif p >= yellow_t: zones.append("YELLOW")
        else:             zones.append("GREEN")
    return zones

def stoplight_counts(prob, y_true, yellow_t, red_t):
    zones = apply_stoplight(prob, yellow_t, red_t)
    df = pd.DataFrame({"Zone":zones, "Actual":y_true.values})
    out = {}
    for z in ["GREEN","YELLOW","RED"]:
        sub = df[df["Zone"]==z]
        out[z] = {"total":len(sub),
                  "bankrupt":sub["Actual"].sum(),
                  "healthy":(sub["Actual"]==0).sum(),
                  "rate":sub["Actual"].mean()*100 if len(sub)>0 else 0}
    return out

def make_cm_fig(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    labels = [[f"{cm[i,j]:,}<br>({cm_pct[i,j]:.1f}%)" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(
        z=cm, x=["Healthy","Bankrupt"], y=["Healthy","Bankrupt"],
        colorscale="Blues", showscale=False,
        text=labels, texttemplate="%{text}",
        textfont={"size":14, "color":"white"}
    ))
    fig.update_layout(title=title, xaxis_title="Predicted",
                      yaxis_title="Actual", height=350,
                      margin=dict(t=50,b=30,l=60,r=20))
    return fig

def make_roc_fig(data):
    fig = go.Figure()
    colors = {"LASSO":"steelblue","Neural Network":"firebrick","Decision Tree":"forestgreen"}
    for name, key in [("LASSO","lasso"),("Neural Network","nn"),("Decision Tree","tree")]:
        prob = data[f"{key}_prob"]
        fpr, tpr, _ = roc_curve(data["y_test"], prob)
        auc = roc_auc_score(data["y_test"], prob)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                 name=f"{name} (AUC={auc:.3f})",
                                 line=dict(color=colors[name], width=2.5)))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                             name="Random guess (AUC=0.500)",
                             line=dict(color="gray", dash="dash", width=1.5)))
    fig.update_layout(
        title="Combined ROC Curve — All Three Models",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate (Recall)",
        legend=dict(x=0.5, y=0.05, xanchor="center"),
        height=450, margin=dict(t=50,b=50,l=60,r=20)
    )
    return fig

# =============================================================================
# MAIN APP
# =============================================================================

# Header
st.markdown(f"""
<div style='background:#1A2744; border-radius:12px; padding:25px 30px;
            margin-bottom:25px;'>
    <span style='font-size:36px; font-weight:bold; color:#F39C12;'>MOBIUS</span>
    <span style='font-size:18px; color:#ECF0F1; margin-left:15px;'>
        Bankruptcy Early Warning System
    </span><br>
    <span style='font-size:13px; color:#7F8C8D;'>
        Pounce Capital Advisors &nbsp;|&nbsp;
        Georgia State University &nbsp;|&nbsp;
        Finance Innovation Showcase
    </span>
</div>
""", unsafe_allow_html=True)

# Load data with spinner
with st.spinner("Loading Mobius... Training models on 78,682 firms. This takes ~2 minutes on first load."):
    data = load_and_train()

st.success("Mobius is ready.")

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Controls")
    st.markdown("---")

    model_select = st.selectbox(
        "Active Model",
        ["LASSO", "Neural Network", "Decision Tree"],
        index=0
    )

    st.markdown("---")
    st.markdown("**Stoplight Thresholds**")
    yellow_thresh = st.slider("YELLOW cutoff", 0.10, 0.60, 0.30, 0.01,
                              format="%.2f")
    red_thresh    = st.slider("RED cutoff",    0.30, 0.90, 0.60, 0.01,
                              format="%.2f")

    if red_thresh <= yellow_thresh:
        st.error("RED must be above YELLOW")

    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown("📊 78,682 US firms")
    st.markdown("📅 1999–2018")
    st.markdown("🏦 5,220 bankruptcies (6.6%)")
    st.markdown("---")
    st.markdown("**Split**")
    st.markdown("🎓 Train: 1999–2013")
    st.markdown("✅ Validate: 2014–2015")
    st.markdown("🔒 Holdout: 2016–2018")

# Get active model data
prob, pred, thresh, base_yt, base_rt = get_model_data(data, model_select)
y_test = data["y_test"]

# Use slider thresholds if they're valid
yt = yellow_thresh if red_thresh > yellow_thresh else base_yt
rt = red_thresh    if red_thresh > yellow_thresh else base_rt

# =============================================================================
# TABS
# =============================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "📈 Model Performance",
    "🚦 Stoplight System",
    "🔍 Backtest",
    "💡 Risk Predictor",
    "📂 Category Analysis"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    st.markdown("### Key Metrics")

    lasso_recall = recall_score(y_test, data["lasso_pred"])
    lasso_auc    = roc_auc_score(y_test, data["lasso_prob"])
    sl_counts    = stoplight_counts(data["lasso_prob"], y_test,
                                    data["lasso_yellow_t"], data["lasso_red_t"])
    system_recall = (sl_counts["RED"]["bankrupt"] +
                     sl_counts["YELLOW"]["bankrupt"]) / y_test.sum() * 100

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value' style='color:#2ECC71;'>
                {system_recall:.1f}%</div>
            <div class='metric-label'>System Recall</div>
            <div style='font-size:11px;color:#BDC3C7;'>Bankruptcies caught</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value' style='color:#3498DB;'>
                {lasso_auc:.3f}</div>
            <div class='metric-label'>Best AUC</div>
            <div style='font-size:11px;color:#BDC3C7;'>LASSO model</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value' style='color:#1A2744;'>78,682</div>
            <div class='metric-label'>Firms Analyzed</div>
            <div style='font-size:11px;color:#BDC3C7;'>1999–2018</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        green_rate = sl_counts["GREEN"]["rate"]
        red_rate   = sl_counts["RED"]["rate"]
        multiplier = round(red_rate / green_rate) if green_rate > 0 else 0
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value' style='color:#E74C3C;'>~{multiplier}x</div>
            <div class='metric-label'>Risk Multiplier</div>
            <div style='font-size:11px;color:#BDC3C7;'>RED vs GREEN zone</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### How Mobius Works")
        st.markdown("""
**Step 1 — LASSO Feature Selection**
Identifies the most predictive financial ratios from 27 variables.

**Step 2 — Model Scoring**
Three independent models each output a bankruptcy probability score (0–100%).

**Step 3 — Stoplight Classification**
Each score is assigned to a risk zone using data-driven thresholds.

**Step 4 — Decision Tree Explanation**
Translates predictions into plain if/then rules for credit officers.
        """)

    with col2:
        st.markdown("### Risk Classification Guide")
        st.markdown(f"""
<div class='green-card' style='margin-bottom:10px;'>
    🟢 &nbsp; GREEN — Low Risk<br>
    <span style='font-size:13px; font-weight:normal;'>
        Proceed with standard review
    </span>
</div>
<div class='yellow-card' style='margin-bottom:10px;'>
    🟡 &nbsp; YELLOW — Watch Zone<br>
    <span style='font-size:13px; font-weight:normal;'>
        Request additional financials. Monitor closely.
    </span>
</div>
<div class='red-card'>
    🔴 &nbsp; RED — High Risk<br>
    <span style='font-size:13px; font-weight:normal;'>
        Escalate for senior approval or decline.
    </span>
</div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    st.markdown("### Model Comparison — All Metrics")

    results = []
    for name, key in [("LASSO","lasso"),("Neural Network","nn"),("Decision Tree","tree")]:
        p = data[f"{key}_prob"]
        d = data[f"{key}_pred"]
        results.append({
            "Model"    : name,
            "Recall ↑" : round(recall_score(y_test, d), 4),
            "Precision": round(precision_score(y_test, d, zero_division=0), 4),
            "F1"       : round(f1_score(y_test, d, zero_division=0), 4),
            "AUC"      : round(roc_auc_score(y_test, p), 4),
            "Accuracy" : round(accuracy_score(y_test, d), 4),
        })
    comp_df = pd.DataFrame(results)

    st.dataframe(
        comp_df.style
            .highlight_max(subset=["Recall ↑","AUC","F1"], color="#d4f0d4")
            .format({"Recall ↑":"{:.4f}","Precision":"{:.4f}",
                     "F1":"{:.4f}","AUC":"{:.4f}","Accuracy":"{:.4f}"}),
        use_container_width=True, height=150
    )

    st.markdown("---")
    st.markdown("### ROC Curve")
    st.plotly_chart(make_roc_fig(data), use_container_width=True)

    st.markdown("---")
    st.markdown("### Confusion Matrices")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(make_cm_fig(y_test, data["lasso_pred"],
                        "LASSO"), use_container_width=True)
    with col2:
        st.plotly_chart(make_cm_fig(y_test, data["nn_pred"],
                        "Neural Network"), use_container_width=True)
    with col3:
        st.plotly_chart(make_cm_fig(y_test, data["tree_pred"],
                        "Decision Tree"), use_container_width=True)

    st.markdown("---")
    st.markdown("### What These Metrics Mean")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.error("**Recall (Primary)**\n\n% of actual bankruptcies the model caught. This is the most important metric — missing a bankruptcy costs far more than a false alarm.")
    with col2:
        st.info("**AUC**\n\nHow well the model separates bankrupt from healthy firms. 0.5 = random guess. 1.0 = perfect.")
    with col3:
        st.success("**Precision**\n\nOf all firms flagged as bankrupt, how many actually were. Lower precision = more false alarms — an acceptable tradeoff when recall is prioritized.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: STOPLIGHT SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

with tab3:
    st.markdown(f"### Stoplight System — {model_select}")
    st.info("Adjust the YELLOW and RED threshold sliders in the sidebar to see how zone distributions change in real time.")

    zones   = apply_stoplight(prob, yt, rt)
    zone_df = pd.DataFrame({"Zone":zones, "Actual":y_test.values, "Prob":prob})

    n_green  = (zone_df["Zone"]=="GREEN").sum()
    n_yellow = (zone_df["Zone"]=="YELLOW").sum()
    n_red    = (zone_df["Zone"]=="RED").sum()
    total    = len(zone_df)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class='green-card'>
            <div style='font-size:40px;'>{n_green:,}</div>
            <div style='font-size:18px;'>GREEN ZONE</div>
            <div style='font-size:12px; opacity:0.85; margin-top:5px;'>
                &lt; {yt*100:.0f}% probability<br>
                {n_green/total*100:.1f}% of firms
            </div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='yellow-card'>
            <div style='font-size:40px;'>{n_yellow:,}</div>
            <div style='font-size:18px;'>YELLOW ZONE</div>
            <div style='font-size:12px; opacity:0.85; margin-top:5px;'>
                {yt*100:.0f}–{rt*100:.0f}% probability<br>
                {n_yellow/total*100:.1f}% of firms
            </div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class='red-card'>
            <div style='font-size:40px;'>{n_red:,}</div>
            <div style='font-size:18px;'>RED ZONE</div>
            <div style='font-size:12px; opacity:0.85; margin-top:5px;'>
                ≥ {rt*100:.0f}% probability<br>
                {n_red/total*100:.1f}% of firms
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        # Stoplight stacked bar
        fig = go.Figure()
        zone_colors = {"GREEN":"#2ECC71","YELLOW":"#F39C12","RED":"#E74C3C"}
        for zone in ["GREEN","YELLOW","RED"]:
            sub = zone_df[zone_df["Zone"]==zone]
            h = (sub["Actual"]==0).sum()
            b = (sub["Actual"]==1).sum()
            fig.add_trace(go.Bar(name="Healthy",  x=[zone], y=[h],
                                 marker_color="steelblue", showlegend=(zone=="GREEN")))
            fig.add_trace(go.Bar(name="Bankrupt", x=[zone], y=[b],
                                 marker_color="#E74C3C", showlegend=(zone=="GREEN"),
                                 base=[h]))
        fig.update_layout(barmode="stack", title="Firms by Zone and Outcome",
                          height=400, legend=dict(x=0.7,y=0.95))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Probability distribution
        fig = go.Figure()
        for zone, color in [("GREEN","#2ECC71"),("YELLOW","#F39C12"),("RED","#E74C3C")]:
            sub_prob = zone_df[zone_df["Zone"]==zone]["Prob"]
            fig.add_trace(go.Histogram(x=sub_prob, name=zone,
                                       marker_color=color, opacity=0.7,
                                       histnorm="probability density",
                                       nbinsx=40))
        fig.add_vline(x=yt, line_dash="dash", line_color="#F39C12",
                      annotation_text=f"YELLOW ({yt:.2f})")
        fig.add_vline(x=rt, line_dash="dash", line_color="#E74C3C",
                      annotation_text=f"RED ({rt:.2f})")
        fig.update_layout(title="Probability Score Distribution",
                          xaxis_title="Bankruptcy Probability",
                          yaxis_title="Density", height=400,
                          barmode="overlay")
        st.plotly_chart(fig, use_container_width=True)

    # Zone stats table
    st.markdown("### Zone Statistics")
    zone_stats = []
    for zone in ["GREEN","YELLOW","RED"]:
        sub = zone_df[zone_df["Zone"]==zone]
        zone_stats.append({
            "Zone"          : zone,
            "Total Firms"   : f"{len(sub):,}",
            "Bankruptcies"  : int(sub["Actual"].sum()),
            "Healthy"       : int((sub["Actual"]==0).sum()),
            "Bankruptcy Rate": f"{sub['Actual'].mean()*100:.1f}%",
            "Avg Score"     : f"{sub['Prob'].mean()*100:.1f}%"
        })
    st.dataframe(pd.DataFrame(zone_stats), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: BACKTEST
# ─────────────────────────────────────────────────────────────────────────────

with tab4:
    st.markdown(f"### Holdout Backtest — {model_select}")
    st.info("Models trained on 1999–2013, thresholds tuned on 2014–2015, backtested on 2016–2018 (never seen during training).")

    n_total  = y_test.sum()
    sl       = stoplight_counts(prob, y_test, yt, rt)
    n_red_b  = sl["RED"]["bankrupt"]
    n_yel_b  = sl["YELLOW"]["bankrupt"]
    n_miss   = sl["GREEN"]["bankrupt"]
    sys_rec  = (n_red_b + n_yel_b) / n_total * 100
    n_yel_t  = sl["YELLOW"]["total"]
    yel_prec = n_yel_b / n_yel_t * 100 if n_yel_t > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value' style='color:#2ECC71;'>{sys_rec:.1f}%</div>
            <div class='metric-label'>System Recall</div>
            <div style='font-size:11px;color:#BDC3C7;'>RED + YELLOW caught</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value' style='color:#E74C3C;'>{n_red_b}</div>
            <div class='metric-label'>Caught in RED</div>
            <div style='font-size:11px;color:#BDC3C7;'>
                {n_red_b/n_total*100:.1f}% of bankruptcies
            </div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value' style='color:#F39C12;'>{n_yel_b}</div>
            <div class='metric-label'>Caught in YELLOW</div>
            <div style='font-size:11px;color:#BDC3C7;'>Early warnings</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value' style='color:#7F8C8D;'>{n_miss}</div>
            <div class='metric-label'>Missed</div>
            <div style='font-size:11px;color:#BDC3C7;'>
                {n_miss/n_total*100:.1f}% of bankruptcies
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        # Backtest bar chart
        fig = go.Figure()
        zone_data_bt = []
        for zone in ["GREEN","YELLOW","RED"]:
            zone_data_bt.append({
                "Zone": zone,
                "Healthy" : sl[zone]["healthy"],
                "Bankrupt": sl[zone]["bankrupt"]
            })

        for outcome, color in [("Healthy","steelblue"),("Bankrupt","#E74C3C")]:
            fig.add_trace(go.Bar(
                name=outcome,
                x=[d["Zone"] for d in zone_data_bt],
                y=[d[outcome] for d in zone_data_bt],
                marker_color=color, opacity=0.85
            ))

        fig.update_layout(barmode="stack",
                          title=f"Backtest — Firms by Zone ({model_select})",
                          xaxis_title="Risk Zone", yaxis_title="Number of Firms",
                          height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Bankruptcy rate by zone
        fig = go.Figure(go.Bar(
            x=["GREEN","YELLOW","RED"],
            y=[sl["GREEN"]["rate"], sl["YELLOW"]["rate"], sl["RED"]["rate"]],
            marker_color=["#2ECC71","#F39C12","#E74C3C"],
            text=[f"{sl[z]['rate']:.1f}%" for z in ["GREEN","YELLOW","RED"]],
            textposition="outside"
        ))
        fig.update_layout(title="Bankruptcy Rate by Zone",
                          yaxis_title="Bankruptcy Rate (%)",
                          height=400, yaxis_range=[0, sl["RED"]["rate"]*1.3])
        st.plotly_chart(fig, use_container_width=True)

    # Backtest summary table
    st.markdown("### Backtest Summary")
    bt_data = []
    for zone in ["GREEN","YELLOW","RED"]:
        bt_data.append({
            "Zone"           : zone,
            "Total Firms"    : f"{sl[zone]['total']:,}",
            "Bankruptcies"   : sl[zone]["bankrupt"],
            "Bankruptcy Rate": f"{sl[zone]['rate']:.1f}%",
        })
    bt_data.append({
        "Zone"          : "TOTAL CAUGHT",
        "Total Firms"   : "—",
        "Bankruptcies"  : n_red_b + n_yel_b,
        "Bankruptcy Rate": f"{sys_rec:.1f}% system recall"
    })
    st.dataframe(pd.DataFrame(bt_data), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: RISK PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────

with tab5:
    st.markdown("### Company Risk Scorer")
    st.info("Enter a company's financial data to get an instant Mobius risk score and stoplight classification.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Financial Inputs**")
        ca   = st.number_input("Current Assets ($M)",          value=100.0, step=10.0)
        ta   = st.number_input("Total Assets ($M)",            value=500.0, step=50.0)
        tl   = st.number_input("Total Liabilities ($M)",       value=300.0, step=25.0)
        cl   = st.number_input("Current Liabilities ($M)",     value=150.0, step=10.0)
        ni   = st.number_input("Net Income ($M)",               value=20.0,  step=5.0)
        rev  = st.number_input("Total Revenue ($M)",            value=400.0, step=25.0)
        eb   = st.number_input("EBITDA ($M)",                   value=60.0,  step=5.0)
        ltd  = st.number_input("Long-term Debt ($M)",           value=100.0, step=10.0)
        re   = st.number_input("Retained Earnings ($M)",        value=50.0,  step=5.0)
        ebit = st.number_input("EBIT ($M)",                     value=45.0,  step=5.0)
        score_btn = st.button("🔍 Score This Company",
                              use_container_width=True,
                              type="primary")

    with col2:
        if score_btn:
            all_features = data["all_features"]
            scaler       = data["scaler"]
            lasso        = data["lasso"]

            firm_vals = {f: 0.0 for f in all_features}
            firm_vals.update({
                "Current_Assets"           : ca,
                "Cost_of_Goods_Sold"       : rev * 0.6,
                "Depreciation_Amortization": eb - ebit,
                "EBITDA"                   : eb,
                "Inventory"                : ca * 0.3,
                "Net_Income"               : ni,
                "Total_Receivables"        : ca * 0.4,
                "Market_Value"             : ta * 1.2,
                "Net_Sales"                : rev,
                "Total_Assets"             : ta,
                "Total_Long_Term_Debt"     : ltd,
                "EBIT"                     : ebit,
                "Gross_Profit"             : rev * 0.35,
                "Total_Current_Liabilities": cl,
                "Retained_Earnings"        : re,
                "Total_Revenue"            : rev,
                "Total_Liabilities"        : tl,
                "Total_Operating_Expense"  : rev - ebit,
                "Debt_Ratio"               : tl/ta if ta != 0 else 0,
                "LT_Debt_Ratio"            : ltd/ta if ta != 0 else 0,
                "Current_Ratio"            : ca/cl if cl != 0 else 0,
                "ROA"                      : ni/ta if ta != 0 else 0,
                "Net_Profit_Margin"        : ni/rev if rev != 0 else 0,
                "EBITDA_Margin"            : eb/rev if rev != 0 else 0,
                "Retained_Earnings_Ratio"  : re/ta if ta != 0 else 0,
                "Operating_Margin"         : ebit/rev if rev != 0 else 0,
                "Asset_Turnover"           : rev/ta if ta != 0 else 0,
            })

            firm_df  = pd.DataFrame([firm_vals])[all_features]
            firm_s   = scaler.transform(firm_df)
            firm_p   = lasso.predict_proba(firm_s)[0, 1]
            firm_pct = round(firm_p * 100, 1)

            lasso_yt = data["lasso_yellow_t"]
            lasso_rt = data["lasso_red_t"]

            if firm_p >= lasso_rt:
                zone_label  = "🔴 RED — HIGH RISK"
                zone_action = "Escalate for senior approval or decline."
                card_class  = "score-red"
            elif firm_p >= lasso_yt:
                zone_label  = "🟡 YELLOW — WATCH ZONE"
                zone_action = "Request additional financials. Monitor closely."
                card_class  = "score-yellow"
            else:
                zone_label  = "🟢 GREEN — LOW RISK"
                zone_action = "Proceed with standard review."
                card_class  = "score-green"

            st.markdown(f"""
<div class='{card_class}'>
    <div style='font-size:56px; font-weight:bold;'>{firm_pct}%</div>
    <div style='font-size:16px; opacity:0.9;'>Bankruptcy Probability</div>
    <div style='font-size:22px; font-weight:bold; margin-top:10px;
                background:rgba(0,0,0,0.15); border-radius:8px; padding:8px;'>
        {zone_label}
    </div>
    <div style='font-size:14px; margin-top:8px; opacity:0.9;'>{zone_action}</div>
</div>
            """, unsafe_allow_html=True)

            st.markdown("**Calculated Ratios:**")
            ratios_df = pd.DataFrame({
                "Ratio"  : ["Debt Ratio","Current Ratio","ROA","EBITDA Margin","Operating Margin"],
                "Value"  : [round(tl/ta,3) if ta else 0,
                            round(ca/cl,3) if cl else 0,
                            round(ni/ta,3) if ta else 0,
                            round(eb/rev,3) if rev else 0,
                            round(ebit/rev,3) if rev else 0]
            })
            st.dataframe(ratios_df, use_container_width=True, hide_index=True)
        else:
            st.markdown("""
<div style='background:white; border-radius:10px; padding:40px;
            text-align:center; color:#7F8C8D; border:2px dashed #BDC3C7;'>
    <div style='font-size:48px;'>📊</div>
    <div style='font-size:16px; margin-top:10px;'>
        Enter financial data and click Score to get a Mobius risk assessment
    </div>
</div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6: CATEGORY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

with tab6:
    st.markdown("### What Drives Bankruptcy? Financial Category Importance")
    st.info("LASSO identified the most predictive variables from 27 financial features. This shows which financial category contributes most to bankruptcy prediction.")

    cat_df = data["cat_df"]

    col1, col2 = st.columns([1.5, 1])

    with col1:
        cat_colors = {
            "Liquidity"    : "#3498DB",
            "Leverage"     : "#E74C3C",
            "Profitability": "#2ECC71",
            "Efficiency"   : "#F39C12",
            "Other"        : "#95A5A6"
        }
        fig = go.Figure(go.Bar(
            x=cat_df["Pct"],
            y=cat_df["Category"],
            orientation="h",
            marker_color=[cat_colors.get(c,"gray") for c in cat_df["Category"]],
            text=[f"{p:.1f}%" for p in cat_df["Pct"]],
            textposition="outside",
            opacity=0.85
        ))
        fig.update_layout(
            title="Cumulative Importance by Financial Category",
            xaxis_title="% of Total Predictive Power",
            height=400,
            xaxis_range=[0, cat_df["Pct"].max() * 1.3],
            margin=dict(t=50,b=30,l=120,r=60)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        top_cat = cat_df.iloc[0]["Category"]
        top_pct = cat_df.iloc[0]["Pct"]
        st.markdown(f"### {top_cat} is the #1 Signal")
        st.markdown(f"""
**{top_pct:.1f}%** of all predictive power comes from {top_cat.lower()} ratios.

---

**The key insight:**

Companies don't go bankrupt because they're unprofitable long-term. They go bankrupt because they **run out of cash right now**.

Liquidity — the ability to pay today's bills — is the earliest and strongest warning signal.

---

**For credit officers:**

Always check the **Current Ratio** and **cash position** first. A company with strong revenue but deteriorating liquidity is already in the danger zone.
        """)

    st.markdown("---")
    st.markdown("### Top Variables by Category")

    lasso_coef   = data["lasso_coef"]
    category_map = data["category_map"]
    coef_abs     = lasso_coef.abs()
    coef_abs     = coef_abs[coef_abs > 0]

    coef_df = pd.DataFrame({
        "Variable": coef_abs.index,
        "Importance": coef_abs.values,
        "Category": [category_map.get(f,"Other") for f in coef_abs.index]
    }).sort_values("Importance", ascending=False)

    fig = px.bar(
        coef_df.groupby("Category").apply(
            lambda x: x.nlargest(5,"Importance")).reset_index(drop=True),
        x="Importance", y="Variable", color="Category", orientation="h",
        facet_col="Category", facet_col_wrap=2,
        color_discrete_map=cat_colors,
        title="Top 5 Variables per Category — LASSO Coefficient Magnitude",
        height=600
    )
    fig.update_yaxes(matches=None)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#7F8C8D; font-size:12px; padding:10px;'>
    MOBIUS — Bankruptcy Early Warning System &nbsp;|&nbsp;
    Pounce Capital Advisors &nbsp;|&nbsp;
    Georgia State University | Robinson College of Business &nbsp;|&nbsp;
    Finance Innovation Showcase
</div>
""", unsafe_allow_html=True)
