from pathlib import Path
import json
import sys

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# 1. Project paths and settings
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
META_PATH = ROOT / "models" / "model_metadata.json"

sys.path.insert(0, str(ROOT / "src"))
from data_utils import load_dataset

st.set_page_config(
    page_title="Machine Health Monitor",
    page_icon="🏭",
    layout="wide",
)

# --------------------------------------------------
# 2. Custom colours and simple styling
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #F3F6FA;
        color: #172B4D;
    }

    [data-testid="stSidebar"] {
        background-color: #10243A;
    }

    [data-testid="stSidebar"] * {
        color: #F1F5F9;
    }

    .main-title {
        font-size: 32px;
        font-weight: 700;
        color: #10243A;
        margin-bottom: 4px;
    }

    .subtitle {
        font-size: 15px;
        color: #64748B;
        margin-bottom: 22px;
    }

    .metric-card {
        background: #FFFFFF;
        border: 1px solid #DFE7F0;
        border-left: 5px solid #159A9C;
        border-radius: 10px;
        padding: 18px;
        min-height: 115px;
        box-shadow: 0 2px 7px rgba(16, 36, 58, 0.05);
    }

    .metric-label {
        color: #64748B;
        font-size: 14px;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #10243A;
        font-size: 25px;
        font-weight: 700;
        overflow-wrap: anywhere;
    }

    .section-heading {
        color: #10243A;
        font-size: 21px;
        font-weight: 650;
        margin-top: 12px;
        margin-bottom: 12px;
    }

    .info-note {
        background: #E6F5F4;
        border-left: 4px solid #159A9C;
        padding: 12px;
        border-radius: 5px;
        color: #174B50;
        margin: 12px 0;
    }

    div.stButton > button[kind="primary"] {
        background-color: #087F8C;
        border: none;
        color: white;
        border-radius: 7px;
        font-weight: 600;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #066873;
        color: white;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #DFE7F0;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 3. Load the trained model
# --------------------------------------------------

st.markdown(
    '<div class="main-title">Machine Health Monitor</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent Predictive Maintenance using Machine Learning'
    '</div>',
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists() or not META_PATH.exists():
    st.error(
        "Trained model not found. Run "
        "`python src/train_model.py` first."
    )
    st.stop()

@st.cache_resource
def load_model_and_metadata():
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    return model, metadata

model, metadata = load_model_and_metadata()
features = metadata["features"]
model_name = metadata["best_model"]

# --------------------------------------------------
# 4. Sidebar
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 🏭 Machine Monitor")
    st.caption("Predictive Maintenance System")
    st.divider()

    st.markdown("### Project details")
    st.write("**Model:**", model_name)
    st.write("**Target:** Machine failure")
    st.write("**Input:** Machine sensor readings")

    st.divider()
    st.markdown("### Navigation")
    st.caption("Choose a section from the tabs on the main page.")

    st.divider()
    st.caption("Academic project | M.Tech AI & DS")

# --------------------------------------------------
# 5. Load dataset and summary cards
# --------------------------------------------------

try:
    df = load_dataset()
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

if "Machine failure" not in df.columns:
    st.error("The dataset does not contain the Machine failure column.")
    st.stop()

total_rows = len(df)
failure_rate = df["Machine failure"].mean() * 100
total_columns = len(df.columns)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Records</div>
        <div class="metric-value">{total_rows:,}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Failure Rate</div>
        <div class="metric-value">{failure_rate:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Input Features</div>
        <div class="metric-value">{len(features)}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# --------------------------------------------------
# 6. Main dashboard tabs
# --------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "🔍 Machine Prediction",
    "📊 Dataset Analysis",
    "📁 Batch Prediction",
])

# ==================================================
# TAB 1: Predict one machine
# ==================================================

with tab1:
    st.markdown(
        '<div class="section-heading">Check Machine Health</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Enter the sensor readings to estimate the probability "
        "of machine failure."
    )

    defaults = {
        "Type": "M",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100,
    }

    left, right = st.columns(2)
    values = {}

    for index, feature in enumerate(features):
        column = left if index % 2 == 0 else right

        with column:
            if feature == "Type":
                values[feature] = st.selectbox(
                    "Machine Type",
                    options=["L", "M", "H"],
                    index=1,
                    help="Select a machine type present in the dataset.",
                )

            elif "speed" in feature.lower():
                values[feature] = st.number_input(
                    feature,
                    min_value=0,
                    max_value=10000,
                    value=int(defaults.get(feature, 1500)),
                )

            elif "wear" in feature.lower():
                values[feature] = st.number_input(
                    feature,
                    min_value=0,
                    max_value=1000,
                    value=int(defaults.get(feature, 100)),
                )

            else:
                values[feature] = st.number_input(
                    feature,
                    value=float(defaults.get(feature, 0.0)),
                )

    threshold = st.slider(
        "Failure alert threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help="The model prediction is positive when its estimated "
             "probability reaches this threshold.",
    )

    if st.button("Check Machine Status", type="primary"):
        input_df = pd.DataFrame([values], columns=features)

        probability = float(model.predict_proba(input_df)[0, 1])
        predicted_failure = probability >= threshold

        st.divider()
        st.markdown("### Prediction Result")

        result1, result2 = st.columns(2)

        with result1:
            st.metric(
                "Estimated Failure Probability",
                f"{probability:.1%}",
            )

        with result2:
            st.metric(
                "Selected Threshold",
                f"{threshold:.0%}",
            )

        if predicted_failure:
            st.error(
                "HIGH ATTENTION: The model has flagged this reading. "
                "A maintenance inspection may be appropriate."
            )
        elif probability >= max(0.0, threshold - 0.15):
            st.warning(
                "MONITOR: The estimated probability is near the alert "
                "threshold. Review the readings."
            )
        else:
            st.success(
                "NO ALERT: The model did not flag this reading "
                "at the selected threshold."
            )

        st.caption(
            "This is a model estimate, not a guarantee. "
            "Do not use it as the only basis for real-world "
            "maintenance or safety decisions."
        )

# ==================================================
# TAB 2: Dataset analysis
# ==================================================

with tab2:
    st.markdown(
        '<div class="section-heading">Explore Machine Data</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Review the dataset, failure distribution and numerical statistics."
    )

    chart_col, detail_col = st.columns([1, 1])

    with chart_col:
        st.markdown("#### Machine Failure Distribution")

        counts = (
            df["Machine failure"]
            .value_counts()
            .reindex([0, 1], fill_value=0)
        )

        counts.index = ["No Failure", "Failure"]
        st.bar_chart(counts, color="#159A9C")

    with detail_col:
        st.markdown("#### Data Quality")

        quality = pd.DataFrame({
            "Measure": [
                "Rows",
                "Columns",
                "Missing cells",
                "Duplicate rows",
            ],
            "Value": [
                len(df),
                len(df.columns),
                int(df.isna().sum().sum()),
                int(df.duplicated().sum()),
            ],
        })

        st.dataframe(
            quality,
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("#### Sensor Readings")

    numeric_cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    available_cols = [c for c in numeric_cols if c in df.columns]

    if available_cols:
        selected_sensor = st.selectbox(
            "Choose a sensor",
            available_cols,
        )

        st.line_chart(
            df[selected_sensor].reset_index(drop=True).head(500),
            color="#159A9C",
        )

        st.caption(
            "The chart shows the first 500 dataset records in row order. "
            "This dataset does not provide a validated chronological "
            "machine history for this chart."
        )

    st.markdown("#### Numerical Summary")

    st.dataframe(
        df.describe().T.round(2),
        use_container_width=True,
    )

    with st.expander("View sample dataset records"):
        st.dataframe(
            df.head(30),
            use_container_width=True,
        )

# ==================================================
# TAB 3: Batch predictions
# ==================================================

with tab3:
    st.markdown(
        '<div class="section-heading">Predict Multiple Machines</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Upload a CSV containing the same input feature columns "
        "used to train the model."
    )

    st.info(
        "Do not include the target column or failure-mode label columns "
        "as model inputs."
    )

    uploaded_file = st.file_uploader(
        "Upload machine readings CSV",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)

            missing_columns = [
                feature for feature in features
                if feature not in batch_df.columns
            ]

            if missing_columns:
                st.error(
                    "These required columns are missing: "
                    + ", ".join(missing_columns)
                )

            elif batch_df.empty:
                st.warning("The uploaded CSV has no records.")

            else:
                batch_features = batch_df[features].copy()

                probabilities = model.predict_proba(
                    batch_features
                )[:, 1]

                results_df = batch_df.copy()
                results_df["Failure Probability"] = probabilities
                results_df["Predicted Failure"] = (
                    probabilities >= 0.50
                ).astype(int)

                st.success(
                    f"Predictions generated for {len(results_df):,} records."
                )

                st.dataframe(
                    results_df.head(100),
                    use_container_width=True,
                )

                st.download_button(
                    label="Download Prediction Results",
                    data=results_df.to_csv(index=False).encode("utf-8"),
                    file_name="machine_failure_predictions.csv",
                    mime="text/csv",
                    type="primary",
                )

        except Exception as exc:
            st.error(f"Could not process the uploaded CSV: {exc}")

# --------------------------------------------------
# 7. Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Dataset: UCI AI4I 2020 Predictive Maintenance Dataset | "
    "Educational machine-learning demonstration"
)
