import streamlit as st
import requests
import pandas as pd

# ── Page Config ───────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🏦",
    layout="centered"
)

# ── Load Dataset ──────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\Y.PHANI MANJUNATH\OneDrive\Desktop\ML_PROJECT\fraud-detection\data\creditcard.csv")
    return df

df       = load_data()
fraud_df = df[df['Class'] == 1]
legit_df = df[df['Class'] == 0]

# ── Title ─────────────────────────────────────────────
st.title("🏦 Credit Card Fraud Detection System")
st.markdown("### Real-Time Transaction Analysis powered by ML")
st.markdown("---")

# ── Sample Buttons ────────────────────────────────────
st.subheader("🧪 Load Sample Transaction")
col1, col2 = st.columns(2)

with col1:
    if st.button("✅ Load Legitimate Sample", use_container_width=True):
        sample = legit_df.sample(1).iloc[0]
        st.session_state['features'] = sample.drop('Class').values.tolist()
        st.session_state['amount']   = float(sample['Amount'])
        st.session_state['time']     = float(sample['Time'])
        st.session_state['label']    = "legitimate"

with col2:
    if st.button("🚨 Load Fraud Sample", use_container_width=True):
        sample = fraud_df.sample(1).iloc[0]
        st.session_state['features'] = sample.drop('Class').values.tolist()
        st.session_state['amount']   = float(sample['Amount'])
        st.session_state['time']     = float(sample['Time'])
        st.session_state['label']    = "fraud"

# ── Show Loaded Sample Info ───────────────────────────
if 'label' in st.session_state:
    if st.session_state['label'] == "fraud":
        st.warning(f"⚠️ Fraud sample loaded — Amount: €{st.session_state['amount']:.2f}")
    else:
        st.info(f"ℹ️ Legitimate sample loaded — Amount: €{st.session_state['amount']:.2f}")

st.markdown("---")

# ── Show Transaction Details ──────────────────────────
if 'features' in st.session_state:
    st.subheader("📝 Transaction Details")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Amount", f"€{st.session_state['amount']:.2f}")
    with col2:
        st.metric("Time",   f"{st.session_state['time']:.0f}s")
    st.markdown("---")

# ── Predict Button ────────────────────────────────────
if st.button("🔍 Check Transaction", use_container_width=True):

    # Check sample loaded
    if 'features' not in st.session_state:
        st.warning("⚠️ Please load a sample first!")
        st.stop()

    # Send exact features to API
    features = st.session_state['features'].copy()

    # ── Call FastAPI ──────────────────────────────────
    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json={"features": features}
        )

        if response.status_code == 200:
            result = response.json()

            st.markdown("---")
            st.subheader("📊 Result")

            # ── Show Result ───────────────────────────
            if result["result"] == "🚨 FRAUD DETECTED":
                st.error(f"## {result['result']}")
            else:
                st.success(f"## {result['result']}")

            # ── Metrics ───────────────────────────────
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Amount",            f"€{st.session_state['amount']:.2f}")
            with col2:
                st.metric("Time",              f"{st.session_state['time']:.0f}s")
            with col3:
                st.metric("Fraud Probability", f"{result['fraud_probability']:.4f}")

        else:
            st.error("❌ API Error!")

    except Exception as e:
        st.error("❌ FastAPI is not running!")
        st.info("👉 Open terminal and run: uvicorn main:app --reload")
