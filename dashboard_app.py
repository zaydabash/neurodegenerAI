"""
Consolidated High-Fidelity Dashboard for Neuro-Trends Suite.
"""


import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Neuro-Trends Suite Dashboard",
    page_icon="monitor",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Configuration (These will come from environment in prod)
CORE_API_URL = os.getenv("CORE_API_URL", "http://127.0.0.1:8000")


# Main components
def show_analytics_overview() -> None:
    st.title("Unified Analytics Overview")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Patients", "1,280", "12%")
    with col2:
        st.metric("Analyzed MRIs", "450", "5%")
    with col3:
        st.metric("Trend Topics", "24", "3")
    with col4:
        st.metric("System Uptime", "99.9%", "Online")

    st.markdown("### System Activity")
    # Placeholder for activity graph
    chart_data = pd.DataFrame(
        np.random.randn(20, 3), columns=["Neuro-API", "Trends-API", "UI-Latency"]
    )
    st.line_chart(chart_data)


def show_neuro_explorer() -> None:
    st.title("NeuroDegenerAI Explorer")

    tabs = st.tabs(["Biomarkers", "MRI Analysis", "EEG Decoding"])

    with tabs[0]:
        st.header("Tabular Biomarker Analysis")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 50, 100, 70)
            sex = st.selectbox("Sex", ["Female", "Male"])
            mmse = st.slider("MMSE Score", 0, 30, 25)
        with col2:
            apoe4 = st.selectbox("APOE4 Alleles", [0, 1, 2])
            abeta = st.number_input("Abeta (pg/mL)", 0.0, 1000.0, 200.0)
            tau = st.number_input("Tau (pg/mL)", 0.0, 1000.0, 300.0)

        if st.button("Analyze Biomarkers", key="btn_tabular"):
            with st.spinner("Analyzing patterns..."):
                payload = {
                    "age": age,
                    "sex": 1 if sex == "Male" else 0,
                    "apoe4": apoe4,
                    "mmse": mmse,
                    "abeta": abeta,
                    "tau": tau,
                }
                res = requests.post(f"{CORE_API_URL}/v1/neuro/tabular", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    st.success(f"Prediction: {data['label']}")
                    st.metric("Probability", f"{data['probability']:.2%}")
                    st.progress(data["confidence"], text="Confidence Score")
                else:
                    st.error("API Error")

    with tabs[1]:
        st.header("MRI Structural Analysis")
        uploaded_file = st.file_uploader(
            "Upload T1-weighted MRI (.nii, .nii.gz)", type=["nii", "gz"]
        )
        if uploaded_file:
            st.info("File uploaded. Ready for processing.")
            if st.button("Run CNN Analysis"):
                st.warning(
                    "MRI Processing requires deep learning backend. Simulated for demo."
                )

    with tabs[2]:
        st.header("EEG Time-Series Decoding")
        state_type = st.selectbox(
            "Simulate EEG State", ["Normal", "Sleep", "Anomalous"]
        )
        if st.button("Generate & Decode EEG"):
            # Migration of EEG logic from previous demo
            from neurodegenerai.src.data.eeg_gen import EEGGenerator

            gen = EEGGenerator()

            with st.spinner(f"Simulating {state_type} state..."):
                if state_type == "Normal":
                    eeg_data = gen.generate_normal_state()
                elif state_type == "Sleep":
                    eeg_data = gen.generate_sleep_state()
                else:
                    eeg_data = gen.generate_anomalous_state()

                # Plotly Visualization
                fig = go.Figure()
                for i in range(min(8, eeg_data.shape[0])):
                    fig.add_trace(
                        go.Scatter(y=eeg_data[i] + (i * 100), name=f"CH {i+1}")
                    )

                fig.update_layout(
                    title="Multi-channel EEG Signal", height=400, showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

                # Call API
                res = requests.post(
                    f"{CORE_API_URL}/v1/neuro/eeg", json={"data": eeg_data.tolist()}
                )
                if res.status_code == 200:
                    data = res.json()
                    st.success(f"Decoded State: {data['label']}")
                    st.metric("Confidence", f"{data['confidence']:.2%}")
                else:
                    st.error("EEG Decoding Failed")


def show_trend_monitor() -> None:
    st.title("Real-Time Trend Monitor")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Top Emerging Topics")
        res = requests.get(f"{CORE_API_URL}/v1/trends/top")
        if res.status_code == 200:
            topics = res.json()["topics"]
            for t in topics:
                with st.expander(f"{t['topic']} (Score: {t['trending_score']:.2f})"):
                    st.write(f"**Keywords:** {', '.join(t['keywords'])}")
                    st.write(
                        f"**Volume:** {t['volume']} | **Growth:** {t['growth_rate']:.1%}"
                    )
                    st.write("**Top Posts:**")
                    for p in t["representative_posts"]:
                        st.markdown(f"> {p}")
        else:
            st.error("Failed to fetch trending topics.")

    with col2:
        st.subheader("Intelligence Search")
        query = st.text_input("Search social data...")
        if query:
            res = requests.post(
                f"{CORE_API_URL}/v1/trends/search", json={"query": query}
            )
            if res.status_code == 200:
                results = res.json()["results"]
                st.write(f"Found {len(results)} matches.")
                for r in results:
                    st.markdown(f"**{r['source']}** ({r['timestamp']})")
                    st.write(r["text"])
                    st.markdown("---")


def show_system_health() -> None:
    st.title("System Configuration & Health")
    # Authentication, Logs, DB Status
    st.info("Hardened Infrastructure Monitoring.")


# Navigation
with st.sidebar:
    st.image(
        "https://via.placeholder.com/200x100/1f77b4/ffffff?text=Neuro-Trends", width=200
    )
    st.markdown("## Navigation")
    page = st.radio(
        "Go to", ["Overview", "Neuro Analysis", "Trend Monitor", "System Health"]
    )
    st.markdown("---")
    st.markdown("### User Control")
    if st.button("Logout"):
        st.write("Logged out")

# Router
if page == "Overview":
    show_analytics_overview()
elif page == "Neuro Analysis":
    show_neuro_explorer()
elif page == "Trend Monitor":
    show_trend_monitor()
elif page == "System Health":
    show_system_health()
