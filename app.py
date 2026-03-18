import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="TAC Intelligence Lab",
    layout="wide",
    page_icon="📊"
)

# =========================================================
# STYLE
# =========================================================
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}
.kpi-card {
    background: linear-gradient(135deg, #133c7a 0%, #1f4fa8 100%);
    padding: 1rem 1rem;
    border-radius: 18px;
    color: white;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    min-height: 118px;
    margin-bottom: 0.6rem;
}
.kpi-title {
    font-size: 0.9rem;
    opacity: 0.9;
    margin-bottom: 0.3rem;
}
.kpi-value {
    font-size: 2.1rem;
    font-weight: 700;
    line-height: 1.05;
}
.kpi-sub {
    font-size: 0.82rem;
    opacity: 0.9;
    margin-top: 0.35rem;
}
.section-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    padding: 0.9rem 1rem;
    border-radius: 14px;
    margin-bottom: 1rem;
}
.insight-banner {
    background: #eef4ff;
    border-left: 5px solid #1d4ed8;
    padding: 0.9rem 1rem;
    border-radius: 12px;
    margin-bottom: 1rem;
}
.small-note {
    color: #475569;
    font-size: 0.92rem;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# CONFIG
# =========================================================
DEFAULT_FILE = "TAC_dummy_data_150.xlsx"
DEFAULT_SHEET = "TAC_Data"

# =========================================================
# HELPERS
# =========================================================
def fmt_cur(x):
    try:
        return f"£{x:,.0f}"
    except Exception:
        return "-"

def fmt_num(x, decimals=1):
    try:
        return f"{x:,.{decimals}f}"
    except Exception:
        return "-"

def fmt_pct(x):
    try:
        return f"{x:.1%}"
    except Exception:
        return "-"

def kpi_card(title, value, sub=""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        xls = pd.ExcelFile(uploaded_file)
    else:
        xls = pd.ExcelFile(DEFAULT_FILE)

    sheet_name = DEFAULT_SHEET if DEFAULT_SHEET in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(xls, sheet_name=sheet_name)
    return df

def enrich_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Dates
    for col in ["Current_Month", "Start_Date", "Expected_End_Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Numeric fields
    numeric_cols = [
        "FTE", "Day_Rate_GBP", "Monthly_Cost_GBP", "Perm_Equivalent_Annual_Cost_GBP",
        "Tenure_Months", "Annualised_TAC_Cost_GBP", "Premium_vs_Perm_GBP", "Risk_Score"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Text fields
    text_cols = [
        "TAC_ID", "Directorate", "Team", "Role_Title", "Engagement_Reason", "IR35_Status",
        "Band_Equivalent", "Vacancy_Linked", "Cover_Type", "Service_Criticality",
        "Manager_or_Service_Owner", "Status_Action", "Recommended_Action", "Risk_Band",
        "Long_Tenure_Flag", "High_Cost_Flag"
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unspecified").astype(str).str.strip()

    # Backfill calculations
    df["Monthly_Cost_GBP"] = np.where(
        df["Monthly_Cost_GBP"].isna() & df["Day_Rate_GBP"].notna(),
        df["Day_Rate_GBP"] * 21.7 * df["FTE"].fillna(1),
        df["Monthly_Cost_GBP"]
    )

    df["Annualised_TAC_Cost_GBP"] = np.where(
        df["Annualised_TAC_Cost_GBP"].isna(),
        df["Monthly_Cost_GBP"].fillna(0) * 12,
        df["Annualised_TAC_Cost_GBP"]
    )

    reference_month = df["Current_Month"].dropna().max()
    if pd.notna(reference_month):
        df["Tenure_Months"] = np.where(
            df["Tenure_Months"].isna() & df["Start_Date"].notna(),
            ((reference_month.year - df["Start_Date"].dt.year) * 12 + (reference_month.month - df["Start_Date"].dt.month)).clip(lower=0),
            df["Tenure_Months"]
        )

    df["Premium_vs_Perm_GBP"] = np.where(
        df["Premium_vs_Perm_GBP"].isna(),
        df["Annualised_TAC_Cost_GBP"].fillna(0) - df["Perm_Equivalent_Annual_Cost_GBP"].fillna(0),
        df["Premium_vs_Perm_GBP"]
    )

    df["Long_Tenure_Flag"] = np.where(
        df["Long_Tenure_Flag"].isin(["Unspecified", "", "nan"]),
        np.where(df["Tenure_Months"].fillna(0) >= 12, "Yes", "No"),
        df["Long_Tenure_Flag"]
    )

    df["High_Cost_Flag"] = np.where(
        df["High_Cost_Flag"].isin(["Unspecified", "", "nan"]),
        np.where(df["Day_Rate_GBP"].fillna(0) >= 650, "Yes", "No"),
        df["High_Cost_Flag"]
    )

    # Risk scoring if weak/missing
    risk_score = pd.Series(0, index=df.index, dtype=float)
    risk_score += np.where(df["Tenure_Months"].fillna(0) >= 24, 30, np.where(df["Tenure_Months"].fillna(0) >= 12, 18, 6))
    risk_score += np.where(df["Day_Rate_GBP"].fillna(0) >= 800, 25, np.where(df["Day_Rate_GBP"].fillna(0) >= 650, 15, 5))
    risk_score += np.where(df["IR35_Status"].eq("Outside"), 12, 4)
    risk_score += np.where(df["Vacancy_Linked"].eq("Yes"), 12, 4)
    risk_score += np.where(df["Service_Criticality"].eq("High"), 8, np.where(df["Service_Criticality"].eq("Medium"), 5, 2))
    risk_score += np.where(df["Recommended_Action"].eq("Convert to Permanent"), 8, 0)
    risk_score += np.where(df["Recommended_Action"].eq("Exit / Decommission"), 10, 0)

    df["Risk_Score"] = np.where(df["Risk_Score"].isna(), risk_score, df["Risk_Score"])

    df["Risk_Band"] = np.where(
        df["Risk_Band"].isin(["Unspecified", "", "nan"]),
        np.select(
            [df["Risk_Score"] >= 70, df["Risk_Score"] >= 45],
            ["High", "Medium"],
            default="Low"
        ),
        df["Risk_Band"]
    )

    df["Convert_to_Perm_Saving_GBP"] = np.where(
        df["Premium_vs_Perm_GBP"].fillna(0) > 0,
        df["Premium_vs_Perm_GBP"].fillna(0),
        0
    )

    df["Tenure_Bucket"] = pd.cut(
        df["Tenure_Months"].fillna(0),
        bins=[-0.1, 6, 12, 24, 36, 999],
        labels=["0–6 months", "6–12 months", "1–2 years", "2–3 years", "3+ years"]
    )

    df["High_Earner_Bucket"] = np.where(df["Day_Rate_GBP"].fillna(0) > 650, "Over £650", "£650 or less")

    return df

def apply_filters(df):
    st.sidebar.header("Filters")

    dir_options = ["All"] + sorted(df["Directorate"].dropna().unique().tolist())
    team_options = ["All"] + sorted(df["Team"].dropna().unique().tolist())
    ir35_options = ["All"] + sorted(df["IR35_Status"].dropna().unique().tolist())
    risk_options = ["All"] + sorted(df["Risk_Band"].dropna().unique().tolist())
    reason_options = ["All"] + sorted(df["Engagement_Reason"].dropna().unique().tolist())

    selected_dir = st.sidebar.selectbox("Directorate", dir_options)
    selected_team = st.sidebar.selectbox("Team", team_options)
    selected_ir35 = st.sidebar.selectbox("IR35 Status", ir35_options)
    selected_risk = st.sidebar.selectbox("Risk Band", risk_options)
    selected_reason = st.sidebar.selectbox("Engagement Reason", reason_options)

    filtered = df.copy()

    if selected_dir != "All":
        filtered = filtered[filtered["Directorate"] == selected_dir]
    if selected_team != "All":
        filtered = filtered[filtered["Team"] == selected_team]
    if selected_ir35 != "All":
        filtered = filtered[filtered["IR35_Status"] == selected_ir35]
    if selected_risk != "All":
        filtered = filtered[filtered["Risk_Band"] == selected_risk]
    if selected_reason != "All":
        filtered = filtered[filtered["Engagement_Reason"] == selected_reason]

    return filtered

def headline_narrative(df, total_workforce_fte, total_employee_monthly_cost):
    if df.empty:
        return "No TACs match the current filters."

    tac_count = len(df)
    tac_monthly = df["Monthly_Cost_GBP"].sum()
    tac_pct_workforce = tac_count / total_workforce_fte if total_workforce_fte else 0
    tac_pct_cost = tac_monthly / total_employee_monthly_cost if total_employee_monthly_cost else 0
    long_tenure = (df["Long_Tenure_Flag"] == "Yes").sum()
    outside_ir35 = (df["IR35_Status"] == "Outside").sum()
    top_dir = (
        df.groupby("Directorate")["Monthly_Cost_GBP"].sum()
        .sort_values(ascending=False)
        .index[0]
    )

    return (
        f"This filtered TAC population contains {tac_count} TACs with a monthly cost of {fmt_cur(tac_monthly)}. "
        f"That represents {tac_pct_workforce:.1%} of workforce and {tac_pct_cost:.1%} of workforce cost based on the assumptions in the sidebar. "
        f"{long_tenure} TACs are over 12 months, {outside_ir35} are outside IR35, and the highest-cost directorate in scope is {top_dir}."
    )

# =========================================================
# POWER BI STYLE SUMMARY PAGE
# =========================================================
def show_dashboard(df, total_workforce_fte, total_employee_monthly_cost):
    st.subheader("TAC Dashboard")
    st.markdown(
        f'<div class="insight-banner"><strong>TAC insight:</strong> {headline_narrative(df, total_workforce_fte, total_employee_monthly_cost)}</div>',
        unsafe_allow_html=True
    )

    tac_count = len(df)
    tac_monthly_cost = df["Monthly_Cost_GBP"].sum()
    tac_annual_cost = df["Annualised_TAC_Cost_GBP"].sum()
    tac_pct_workforce = tac_count / total_workforce_fte if total_workforce_fte else 0
    tac_pct_workforce_cost = tac_monthly_cost / total_employee_monthly_cost if total_employee_monthly_cost else 0
    avg_monthly_tac_cost = df["Monthly_Cost_GBP"].mean()
    avg_monthly_emp_cost = total_employee_monthly_cost / total_workforce_fte if total_workforce_fte else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi_card("TAC Cost This Month", fmt_cur(tac_monthly_cost), f"Annualised: {fmt_cur(tac_annual_cost)}")
    with c2:
        kpi_card("TAC % of Workforce", fmt_pct(tac_pct_workforce), f"TACs: {tac_count} / workforce {fmt_num(total_workforce_fte,0)}")
    with c3:
        kpi_card("TAC % of Workforce Cost", fmt_pct(tac_pct_workforce_cost), f"TAC cost vs employee monthly cost")
    with c4:
        kpi_card("Monthly TAC Cost (Avg)", fmt_cur(avg_monthly_tac_cost), "Monthly TAC cost ÷ TAC count")
    with c5:
        kpi_card("Monthly Emp Cost (Avg)", fmt_cur(avg_monthly_emp_cost), "Monthly employee cost ÷ workforce FTE")

    left, right = st.columns([1.1, 1])

    with left:
        st.markdown("### TAC Workforce Profile & Compliance")
        row1_col1, row1_col2 = st.columns(2)

        with row1_col1:
            tenure_dist = (
                df.groupby("Tenure_Bucket", as_index=False)["TAC_ID"]
                .count()
                .rename(columns={"TAC_ID": "Count"})
            )
            fig = px.bar(tenure_dist, x="Tenure_Bucket", y="Count", text_auto=True)
            fig.update_layout(title="TAC Tenure Distribution", height=360, xaxis_title="", yaxis_title="Count")
            st.plotly_chart(fig, width="stretch")

        with row1_col2:
            gt1 = df[df["Tenure_Months"].fillna(0) >= 12]
            reason = (
                gt1.groupby("Engagement_Reason", as_index=False)["TAC_ID"]
                .count()
                .rename(columns={"TAC_ID": "Count"})
            )
            fig = px.pie(reason, names="Engagement_Reason", values="Count", hole=0.55)
            fig.update_layout(title="TACs > 1 year: Engagement Reason", height=360)
            st.plotly_chart(fig, width="stretch")

        row2_col1, row2_col2 = st.columns(2)

        with row2_col1:
            avg_tenure = df["Tenure_Months"].mean() / 12 if len(df) else 0
            max_tenure = df["Tenure_Months"].max() / 12 if len(df) else 0
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_tenure,
                number={"suffix": " yrs", "valueformat": ".1f"},
                title={"text": "Tenure Snapshot: Avg"},
                gauge={
                    "axis": {"range": [None, max(4.5, max_tenure + 0.3)]},
                    "bar": {"color": "#ef4444"},
                    "steps": [
                        {"range": [0, 1], "color": "#f3f4f6"},
                        {"range": [1, 2], "color": "#e5e7eb"},
                        {"range": [2, 3], "color": "#d1d5db"},
                        {"range": [3, max(4.5, max_tenure + 0.3)], "color": "#f9fafb"},
                    ],
                }
            ))
            fig.add_annotation(
                x=0.95, y=0.06, xref="paper", yref="paper",
                text=f"Max: {max_tenure:.2f} yrs", showarrow=False
            )
            fig.update_layout(height=320)
            st.plotly_chart(fig, width="stretch")

        with row2_col2:
            ir35 = (
                df.groupby("IR35_Status", as_index=False)["TAC_ID"]
                .count()
                .rename(columns={"TAC_ID": "Count"})
            )
            fig = px.pie(ir35, names="IR35_Status", values="Count", hole=0.55)
            fig.update_layout(title="IR35 Status", height=320)
            st.plotly_chart(fig, width="stretch")

    with right:
        st.markdown("### TAC Cost Comparisons")
        top_col1, top_col2 = st.columns(2)

        with top_col1:
            high_earners = (
                df.groupby("High_Earner_Bucket", as_index=False)["TAC_ID"]
                .count()
                .rename(columns={"TAC_ID": "Count"})
            )
            fig = px.pie(high_earners, names="High_Earner_Bucket", values="Count", hole=0.55)
            fig.update_layout(title="High Earners (Day Cost)", height=320)
            st.plotly_chart(fig, width="stretch")

        with top_col2:
            band = (
                df.groupby("Band_Equivalent", as_index=False)["Monthly_Cost_GBP"]
                .mean()
                .rename(columns={"Monthly_Cost_GBP": "AvgMonthlyCost"})
                .sort_values("AvgMonthlyCost", ascending=False)
            )
            fig = px.bar(band, x="Band_Equivalent", y="AvgMonthlyCost", text_auto=".2s")
            fig.update_layout(title="Average Cost by Band", height=320, xaxis_title="", yaxis_title="Avg monthly cost (£)")
            st.plotly_chart(fig, width="stretch")

        st.markdown("### Projected Saving Opportunities - Scenario Modelling over 12 months")
        save_col1, save_col2, save_col3 = st.columns(3)

        replace_10 = df["Convert_to_Perm_Saving_GBP"].sum() * 0.10
        reduce_10 = df["Annualised_TAC_Cost_GBP"].sum() * 0.10
        premium_step4 = df[df["Day_Rate_GBP"].fillna(0) >= 650]["Annualised_TAC_Cost_GBP"].sum() * 0.20

        with save_col1:
            kpi_card("Replace 10%", fmt_cur(replace_10), "Convert-to-perm saving proxy")
        with save_col2:
            kpi_card("Reduce by 10%", fmt_cur(reduce_10), "Straight reduction scenario")
        with save_col3:
            kpi_card("Premium (Step 4)", fmt_cur(premium_step4), "Premium-cost reduction scenario")

    st.markdown(
        '<div class="small-note"><strong>Note:</strong> workforce comparisons are based on the workforce assumptions entered in the sidebar. This lets you mirror the Power BI view while testing alternative assumptions.</div>',
        unsafe_allow_html=True
    )

# =========================================================
# EXTRA PAGES
# =========================================================
def show_cost_analysis(df):
    st.subheader("Cost & Headcount Analysis")

    by_dir = (
        df.groupby("Directorate", as_index=False)
        .agg(
            TAC_Count=("TAC_ID", "count"),
            Monthly_Cost=("Monthly_Cost_GBP", "sum"),
            Annualised_Cost=("Annualised_TAC_Cost_GBP", "sum"),
            Avg_Day_Rate=("Day_Rate_GBP", "mean")
        )
        .sort_values("Annualised_Cost", ascending=False)
    )
    st.dataframe(by_dir, width="stretch", height=400)

    left, right = st.columns(2)
    with left:
        fig = px.bar(by_dir, x="Directorate", y="Annualised_Cost", text_auto=".2s")
        fig.update_layout(height=430, xaxis_title="", yaxis_title="Annualised cost (£)")
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = px.scatter(
            by_dir,
            x="TAC_Count",
            y="Annualised_Cost",
            size="Avg_Day_Rate",
            text="Directorate"
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(height=430, xaxis_title="TAC count", yaxis_title="Annualised cost (£)")
        st.plotly_chart(fig, width="stretch")

def show_risk_compliance(df):
    st.subheader("Risk & Compliance")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("High Risk TACs", f"{(df['Risk_Band']=='High').sum():,}", "Risk score driven")
    with c2:
        kpi_card("Outside IR35", f"{(df['IR35_Status']=='Outside').sum():,}", "Compliance focus")
    with c3:
        kpi_card("Over 12 months", f"{(df['Tenure_Months'].fillna(0)>=12).sum():,}", "Long-tenure flag")
    with c4:
        kpi_card("High cost TACs", f"{(df['High_Cost_Flag']=='Yes').sum():,}", "£650/day+")

    left, right = st.columns(2)
    with left:
        risk_dist = (
            df.groupby("Risk_Band", as_index=False)["TAC_ID"]
            .count()
            .rename(columns={"TAC_ID": "Count"})
        )
        fig = px.bar(risk_dist, x="Risk_Band", y="Count", text_auto=True)
        fig.update_layout(height=400, xaxis_title="", yaxis_title="Count")
        st.plotly_chart(fig, width="stretch")

    with right:
        risk_dir = (
            df.groupby("Directorate", as_index=False)["Risk_Score"]
            .mean()
            .sort_values("Risk_Score", ascending=False)
        )
        fig = px.bar(risk_dir, x="Directorate", y="Risk_Score", text_auto=".1f")
        fig.update_layout(height=400, xaxis_title="", yaxis_title="Average risk score")
        st.plotly_chart(fig, width="stretch")

    st.markdown("### Highest Risk TACs")
    cols = [
        "TAC_ID", "Directorate", "Team", "Role_Title", "Tenure_Months", "Day_Rate_GBP",
        "IR35_Status", "Annualised_TAC_Cost_GBP", "Premium_vs_Perm_GBP", "Risk_Score",
        "Risk_Band", "Recommended_Action"
    ]
    st.dataframe(
        df.sort_values(["Risk_Score", "Annualised_TAC_Cost_GBP"], ascending=False)[cols],
        width="stretch",
        height=500
    )

def show_role_explorer(df):
    st.subheader("Role Explorer")

    search = st.text_input("Search TAC ID, role title, manager, team, or directorate")
    view = df.copy()

    if search:
        mask = (
            view["TAC_ID"].astype(str).str.contains(search, case=False, na=False)
            | view["Role_Title"].astype(str).str.contains(search, case=False, na=False)
            | view["Manager_or_Service_Owner"].astype(str).str.contains(search, case=False, na=False)
            | view["Team"].astype(str).str.contains(search, case=False, na=False)
            | view["Directorate"].astype(str).str.contains(search, case=False, na=False)
        )
        view = view[mask]

    st.dataframe(
        view[[
            "TAC_ID", "Directorate", "Team", "Role_Title", "Start_Date", "Expected_End_Date",
            "Tenure_Months", "Day_Rate_GBP", "Monthly_Cost_GBP", "Annualised_TAC_Cost_GBP",
            "Perm_Equivalent_Annual_Cost_GBP", "Premium_vs_Perm_GBP",
            "IR35_Status", "Engagement_Reason", "Band_Equivalent",
            "Status_Action", "Recommended_Action", "Risk_Score", "Risk_Band"
        ]].sort_values(["Risk_Score", "Annualised_TAC_Cost_GBP"], ascending=False),
        width="stretch",
        height=560
    )

def show_scenario_lab(df):
    st.subheader("Scenario Lab")
    st.markdown(
        '<div class="section-box">Use these sliders to test a more flexible version of the Power BI saving cards. This is a scenario tool, not a final budget-setting model.</div>',
        unsafe_allow_html=True
    )

    base_cost = df["Annualised_TAC_Cost_GBP"].sum()
    convert_pool = df[df["Recommended_Action"].isin(["Convert to Permanent", "Review"])].copy()
    long_pool = df[df["Tenure_Months"].fillna(0) >= 12].copy()
    premium_pool = df[df["Day_Rate_GBP"].fillna(0) >= 650].copy()

    left, right = st.columns(2)
    with left:
        convert_pct = st.slider("Convert suitable TACs to permanent (%)", 0, 100, 35, 5)
        long_reduce_pct = st.slider("Reduce long-tenure TAC population (%)", 0, 100, 20, 5)
    with right:
        premium_cap = st.slider("Cap premium day rates above (£)", 500, 900, 650, 25)
        premium_reduce_pct = st.slider("Reduce premium-cost TAC population (%)", 0, 100, 15, 5)

    convert_saving = convert_pool["Convert_to_Perm_Saving_GBP"].sum() * (convert_pct / 100)
    long_saving = long_pool["Annualised_TAC_Cost_GBP"].sum() * 0.35 * (long_reduce_pct / 100)
    premium_rate_saving = ((premium_pool["Day_Rate_GBP"] - premium_cap).clip(lower=0) * 21.7 * 12 * premium_pool["FTE"].fillna(1)).sum()
    premium_population_saving = premium_pool["Annualised_TAC_Cost_GBP"].sum() * 0.30 * (premium_reduce_pct / 100)

    gross_saving = convert_saving + long_saving + premium_rate_saving + premium_population_saving
    residual_cost = max(0, base_cost - gross_saving)
    equivalent_tacs = gross_saving / (df["Annualised_TAC_Cost_GBP"].mean() if len(df) else 1)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Current annual TAC cost", fmt_cur(base_cost), "Baseline")
    with c2:
        kpi_card("Gross annual saving", fmt_cur(gross_saving), "Scenario estimate")
    with c3:
        kpi_card("Residual run-rate", fmt_cur(residual_cost), "Remaining annual cost")
    with c4:
        kpi_card("Indicative TAC equivalent", fmt_num(equivalent_tacs), "Approximate TACs affected")

    scenario_df = pd.DataFrame({
        "Lever": [
            "Convert to permanent",
            "Reduce long-tenure TACs",
            "Cap premium day rates",
            "Reduce premium-cost population"
        ],
        "Saving_GBP": [
            convert_saving,
            long_saving,
            premium_rate_saving,
            premium_population_saving
        ]
    })

    fig = px.bar(scenario_df, x="Lever", y="Saving_GBP", text_auto=".2s")
    fig.update_layout(height=440, xaxis_title="", yaxis_title="Estimated annual saving (£)")
    st.plotly_chart(fig, width="stretch")

def show_recommendations(df):
    st.subheader("Recommendations")

    tab1, tab2, tab3 = st.tabs(["Top review now", "Top convert-to-perm", "Top premium-cost"])

    with tab1:
        review_now = df.sort_values(["Risk_Score", "Annualised_TAC_Cost_GBP"], ascending=False).head(15)
        st.dataframe(
            review_now[[
                "TAC_ID", "Directorate", "Role_Title", "Annualised_TAC_Cost_GBP",
                "Risk_Score", "Risk_Band", "Recommended_Action"
            ]],
            width="stretch",
            height=420
        )

    with tab2:
        convert_now = df.sort_values("Convert_to_Perm_Saving_GBP", ascending=False).head(15)
        st.dataframe(
            convert_now[[
                "TAC_ID", "Directorate", "Role_Title",
                "Annualised_TAC_Cost_GBP", "Perm_Equivalent_Annual_Cost_GBP",
                "Convert_to_Perm_Saving_GBP", "Recommended_Action"
            ]],
            width="stretch",
            height=420
        )

    with tab3:
        premium_now = df.sort_values("Day_Rate_GBP", ascending=False).head(15)
        st.dataframe(
            premium_now[[
                "TAC_ID", "Directorate", "Role_Title",
                "Day_Rate_GBP", "Annualised_TAC_Cost_GBP", "Tenure_Months",
                "Recommended_Action"
            ]],
            width="stretch",
            height=420
        )

def show_data_quality(df):
    st.subheader("Data Quality & Assumptions")

    quality = pd.DataFrame({
        "Check": [
            "Missing TAC ID",
            "Missing Directorate",
            "Missing Team",
            "Missing Role Title",
            "Missing Start Date",
            "Missing Day Rate",
            "Missing Monthly Cost",
            "Missing Perm Equivalent Cost",
            "Missing Risk Score"
        ],
        "Count": [
            df["TAC_ID"].eq("Unspecified").sum(),
            df["Directorate"].eq("Unspecified").sum(),
            df["Team"].eq("Unspecified").sum(),
            df["Role_Title"].eq("Unspecified").sum(),
            df["Start_Date"].isna().sum(),
            df["Day_Rate_GBP"].isna().sum(),
            df["Monthly_Cost_GBP"].isna().sum(),
            df["Perm_Equivalent_Annual_Cost_GBP"].isna().sum(),
            df["Risk_Score"].isna().sum()
        ]
    })
    st.dataframe(quality, width="stretch")

    st.markdown("""
**Assumptions used in this app**
- Monthly TAC cost is backfilled from `day rate × 21.7 working days × FTE` where missing.
- Annual TAC cost is monthly cost × 12 where missing.
- Premium vs permanent cost is annual TAC cost minus permanent equivalent annual cost.
- Long tenure is set at 12 months or more.
- High-cost flag is set at £650/day or above.
- Risk score is calculated from tenure, day rate, IR35, vacancy linkage, service criticality, and recommended action where required.
- Workforce comparisons use the workforce assumptions entered in the sidebar.
""")

    with st.expander("Show sample data"):
        st.dataframe(df.head(100), width="stretch", height=450)

# =========================================================
# APP
# =========================================================
st.title("TAC Intelligence Lab")
st.caption("Contingent workforce dashboard, cost analysis, compliance, and scenario modelling")

uploaded_file = st.sidebar.file_uploader("Upload TAC Excel file", type=["xlsx"])

st.sidebar.header("Workforce assumptions")
total_workforce_fte = st.sidebar.number_input("Total workforce FTE", min_value=1.0, value=692.0, step=1.0)
total_employee_monthly_cost = st.sidebar.number_input("Total employee monthly cost (£)", min_value=1.0, value=3586106.0, step=1000.0)

page = st.sidebar.radio(
    "Navigate",
    [
        "TAC Dashboard",
        "Cost & Headcount Analysis",
        "Risk & Compliance",
        "Role Explorer",
        "Scenario Lab",
        "Recommendations",
        "Data Quality"
    ]
)

try:
    raw_df = load_data(uploaded_file)
    df = enrich_data(raw_df)
    filtered_df = apply_filters(df)

    if filtered_df.empty:
        st.warning("No TACs match the current filter selection.")
        st.stop()

    if page == "TAC Dashboard":
        show_dashboard(filtered_df, total_workforce_fte, total_employee_monthly_cost)
    elif page == "Cost & Headcount Analysis":
        show_cost_analysis(filtered_df)
    elif page == "Risk & Compliance":
        show_risk_compliance(filtered_df)
    elif page == "Role Explorer":
        show_role_explorer(filtered_df)
    elif page == "Scenario Lab":
        show_scenario_lab(filtered_df)
    elif page == "Recommendations":
        show_recommendations(filtered_df)
    elif page == "Data Quality":
        show_data_quality(filtered_df)

    st.download_button(
        "Download filtered TAC data as CSV",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="tac_filtered_output.csv",
        mime="text/csv"
    )

except FileNotFoundError:
    st.error(f"Could not find `{DEFAULT_FILE}`. Upload the workbook in the sidebar or place it in the same folder as app.py.")
except Exception as e:
    st.exception(e)