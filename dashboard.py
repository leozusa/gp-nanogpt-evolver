import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import time
import json
from config import RESULTS_CSV

st.set_page_config(page_title="nanoGPT Evolver Dashboard", layout="wide")

# Initialize session state for the dropdown
if 'selected_individual_id' not in st.session_state:
    st.session_state.selected_individual_id = None

def load_data():
    if os.path.exists(RESULTS_CSV):
        try:
            df = pd.read_csv(RESULTS_CSV, on_bad_lines='skip')
            numeric_cols = ["fitness", "val_bpb", "num_params_M", "mfu_percent", "num_steps", "generation"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            
            df = df.dropna(subset=["fitness"])
            
            if "timestamp" in df.columns:
                df = df.sort_values("timestamp").reset_index(drop=True)
                df["eval_index"] = df.index + 1 
                df["rolling_best_fitness"] = df["fitness"].cummax()
            
            return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

st.title("🧬 nanoGPT Evolver Dashboard v1.0 (Steady-State)")
st.markdown("Real-time asynchronous neuroevolution tracking")

st.sidebar.header("Controls")
refresh_btn = st.sidebar.button("Refresh Data")
auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=True)

df = load_data()

if df.empty:
    st.info("Waiting for first generation of results... Check results/results.csv")
else:
    best_ind = df.loc[df["fitness"].idxmax()]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Global Best Fitness", f"{float(best_ind['fitness']):.4f}")
    col2.metric("Best val_bpb", f"{float(best_ind['val_bpb']):.4f}")
    col3.metric("Total Evaluations", len(df))
    col4.metric("Deepest Lineage (Gen)", int(df["generation"].max()))

    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Evolution Trajectory")
        fig_fitness = go.Figure()
        fig_fitness.add_trace(go.Scatter(
            x=df["eval_index"], y=df["fitness"],
            mode='markers',
            name='Individual Runs',
            marker=dict(color='rgba(135, 206, 250, 0.5)', size=6)
        ))
        fig_fitness.add_trace(go.Scatter(
            x=df["eval_index"], y=df["rolling_best_fitness"],
            mode='lines',
            name='Global Best',
            line=dict(color='yellow', width=3)
        ))
        fig_fitness.update_layout(
            title="Fitness over Total Evaluations",
            xaxis_title="Total Models Evaluated",
            yaxis_title="Fitness (-val_bpb)",
            showlegend=True
        )
        st.plotly_chart(fig_fitness, use_container_width=True)

    with c2:
        st.subheader("Moving Average BPB")
        df["rolling_bpb"] = df["val_bpb"].rolling(window=10, min_periods=1).mean()
        fig_bpb = px.line(df, x="eval_index", y="rolling_bpb", 
                        title="Population Health (10-Model Rolling Avg BPB)")
        fig_bpb.update_layout(xaxis_title="Total Models Evaluated", yaxis_title="Average BPB")
        st.plotly_chart(fig_bpb, use_container_width=True)

    st.subheader("🏆 Top 10 Individuals")
    top_10 = df.sort_values(by="fitness", ascending=False).head(10)
    st.dataframe(top_10[["individual_id", "fitness", "val_bpb", "n_layer", "n_head", "n_embd", "matrix_lr", "mfu_percent", "generation"]])

    st.subheader("Parameter Analysis")
    numeric_df = df.select_dtypes(include=['float64', 'int64']).drop(["eval_index", "rolling_best_fitness", "timestamp"], errors='ignore')
    if not numeric_df.empty and len(df) > 1:
        corr = numeric_df.corr()["fitness"].sort_values(ascending=False).drop("fitness", errors='ignore')
        fig_corr = px.bar(x=corr.index, y=corr.values, labels={"x": "Parameter", "y": "Correlation with Fitness"},
                         title="Parameter Importance (Correlation)")
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.write("Need more data for correlation analysis.")

    # --- Individual Explorer with persistent state ---
    st.subheader("🔍 Individual Detail")
    
    unique_ids = df["individual_id"].unique().tolist()
    
    # Determine the index of the previously selected ID (if it exists in the current list)
    try:
        if st.session_state.selected_individual_id in unique_ids:
            default_idx = unique_ids.index(st.session_state.selected_individual_id)
        else:
            default_idx = 0
    except ValueError:
        default_idx = 0

    # Callback to update session state when user makes a selection
    def update_selected_id():
        st.session_state.selected_individual_id = st.session_state.dropdown_id

    # The dropdown itself
    selected_id = st.selectbox(
        "Select Individual ID", 
        options=unique_ids, 
        index=default_idx, 
        key="dropdown_id",
        on_change=update_selected_id
    )

    # Fallback to update state if this is the very first render and user hasn't clicked
    if st.session_state.selected_individual_id is None:
         st.session_state.selected_individual_id = selected_id

    selected_ind = df[df["individual_id"] == selected_id].iloc[0]
    st.json(selected_ind.to_dict())

    st.subheader("💾 Export Best Config")
    best_json = json.dumps(best_ind.drop(["eval_index", "rolling_best_fitness", "rolling_bpb"], errors="ignore").to_dict(), indent=2)
    st.download_button(
        label="Download Best Config (JSON)",
        data=best_json,
        file_name="best_config.json",
        mime="application/json"
    )

if auto_refresh or refresh_btn:
    if not refresh_btn: 
        time.sleep(5)
    st.rerun()
