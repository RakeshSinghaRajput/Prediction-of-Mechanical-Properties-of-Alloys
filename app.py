import streamlit as st
import numpy as np
import pandas as pd
import joblib
import optuna
from PIL import Image

# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Alloy Property Prediction",
    page_icon="⚙",
    layout="wide"
)
# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #f5f7fa;
}
/* Top header */
header[data-testid="stHeader"] {
    background-color: #f5f7fa;
}
/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #dce6f2;
}

/* Main title */
.main-title {
    font-size: 48px;
    font-weight: 800;
    color: #003366;
    text-align: center;
    margin-top: 10px;
}

/* Subtitle */
.sub-title {
    font-size: 24px;
    color: #666666;
    text-align: center;
    margin-bottom: 30px;
}

/* Section headings */
.section-title {
    font-size: 32px;
    font-weight: 700;
    color: #003366;
    margin-top: 20px;
}

/* Button Design */
.stButton > button {
    background-color: #0059b3;
    color: white;
    font-size: 20px;
    border-radius: 10px;
    height: 3em;
    border: none;
    padding: 0 30px;
}

.stButton>button:hover {
    background-color: #EB3F36;
    color: white;
}

/* Dataframe spacing */
.block-container {
    padding-top: 1rem;
}

/* Metric cards */
[data-testid="metric-container"] {
    background-color: white;
    border: 1px solid #d9d9d9;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOGO + HEADER
# =========================================================

logo = Image.open("logo.png")

col1, col2, col3 = st.columns([3,1,3])

with col2:
    st.image(logo, width=220)

st.markdown(
    """
    <div class='main-title'>
        Machine Learning-Based Design of High Temperature Alloy
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class='sub-title'>
        Alloy Composition Optimization
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

#
# =========================================================
# LOAD FILES
# =========================================================

model = joblib.load("Gredient_Boost_Model.pkl")

X_scaler = joblib.load("X_scaled_gb(updated).pkl")

y_scaler = joblib.load("y_scaled_gb(updated).pkl")

train_cols = joblib.load("gradient_boosting_column.pkl")


# =========================================================
# TARGET INPUT
# =========================================================

st.sidebar.header("Target Properties")

target_uts = st.sidebar.number_input(
    "Target UTS",
    value=243.0
)

target_ys = st.sidebar.number_input(
    "Target YS",
    value=194.0
)

target_el = st.sidebar.number_input(
    "Target EL",
    value=6.2
)

target = np.array([
    target_uts,
    target_ys,
    target_el
])

target_scaled = y_scaler.transform([target])[0]


# =========================================================
# TEMPERATURE
# =========================================================

st.sidebar.header("Temperature")

temperature = st.sidebar.selectbox(
    "Select Temperature",
    [200, 250, 300]
)


# =========================================================
# HEAT TREATMENT
# =========================================================

st.sidebar.header("Heat Treatment")

heat_treatment = {

    'S4': st.sidebar.checkbox("S4", value=False),

    'S8': st.sidebar.checkbox("S8", value=False),

    'T6': st.sidebar.checkbox("T6", value=True),

    'T7': st.sidebar.checkbox("T7", value=False),

    'W': st.sidebar.checkbox("W", value=False),

    'A': st.sidebar.checkbox("A", value=False),

    'A1': st.sidebar.checkbox("A1", value=False),

    'A2': st.sidebar.checkbox("A2", value=False),

    'A3': st.sidebar.checkbox("A3", value=False),

    'A4': st.sidebar.checkbox("A4", value=False)
}


# =========================================================
# ELEMENT RANGES
# =========================================================

st.sidebar.header("Element Ranges")

elements = {
    'Cu': (0.0, 8.0),
    'Mn': (0.0, 2.0),
    'Mg': (0.0, 2.0),
    'Fe': (0.0, 1.5),
    'Si': (0.0, 1.0),
    'Zr': (0.0, 0.5),
    'Ti': (0.0, 0.3),
    'Sc': (0.0, 0.5),
    'Ni': (0.0, 3.0),
    'V': (0.0, 0.2),
    'Ag': (0.0, 0.2)
}

element_ranges = {}

for element, limits in elements.items():

    st.sidebar.subheader(element)

    low = st.sidebar.number_input(
        f"{element} Min",
        min_value=0.0,
        max_value=100.0,
        value=float(limits[0]),
        key=f"{element}_min"
    )

    high = st.sidebar.number_input(
        f"{element} Max",
        min_value=0.0,
        max_value=100.0,
        value=float(limits[1]),
        key=f"{element}_max"
    )

    element_ranges[element] = (low, high)


# =========================================================
# NUMBER OF TRIALS
# =========================================================

n_trials = st.sidebar.slider(
    "Number of Trials",
    100,
    5000,
    2000
)


# =========================================================
# OPTIMIZATION BUTTON
# =========================================================

col1, col2, col3 = st.columns([2,1,2])

with col2:
    optimize_button = st.button("Find Elements Composition")

if optimize_button:

    progress_bar = st.progress(0)

    status_text = st.empty()

    # =====================================================
    # OBJECTIVE FUNCTION
    # =====================================================

    def objective(trial):

        params = {}

        # =================================================
        # ELEMENT OPTIMIZATION
        # =================================================

        for element, limits in element_ranges.items():

            low, high = limits

            params[element] = trial.suggest_float(
                element,
                low,
                high
                
                
            )

        # =================================================
        # AL BALANCE
        # =================================================

        total = sum(params.values())

        al_value = 100 - total

        if al_value <= 0:

            return 1e6

        params['Al'] = al_value

        # =================================================
        # ADD HEAT TREATMENT
        # =================================================

        for k, v in heat_treatment.items():

            params[k] = int(v)

        # =================================================
        # DATAFRAME
        # =================================================

        df_input = pd.DataFrame([params])

        # =================================================
        # TEMPERATURE ONE HOT
        # =================================================

        for t in [200, 250, 300]:

            df_input[f'Temperature (C)_{t}'] = (
                1 if temperature == t else 0
            )

        # =================================================
        # MATCH COLUMNS
        # =================================================

        df_input = df_input.reindex(
            columns=train_cols,
            fill_value=0
        )

        # =================================================
        # SCALE INPUT
        # =================================================

        X_scaled_input = X_scaler.transform(df_input)

        # =================================================
        # PREDICT
        # =================================================

        pred_scaled = model.predict(X_scaled_input)

        # =================================================
        # SIMPLE ERROR
        # =================================================

        error = np.mean(
            (pred_scaled[0] - target_scaled) ** 2
        )

        return error


    # =====================================================
    # CALLBACK FOR PROGRESS BAR
    # =====================================================

    def callback(study, trial):

        progress = (trial.number + 1) / n_trials

        progress_bar.progress(progress)

        status_text.text(
            f"Trial {trial.number + 1}/{n_trials}"
        )


    # =====================================================
    # OPTIMIZATION
    # =====================================================

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=50)
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[callback]
    )


    # =====================================================
    # BEST PARAMETERS
    # =====================================================

    best_params = study.best_params

    final_params = best_params.copy()

    total = sum(final_params.values())

    final_params['Al'] = 100 - total

    # Add heat treatment
    for k, v in heat_treatment.items():

        final_params[k] = int(v)


    # =====================================================
    # FINAL DATAFRAME
    # =====================================================

    best_df = pd.DataFrame([final_params])

    for t in [200, 250, 300]:

        best_df[f'Temperature (C)_{t}'] = (
            1 if temperature == t else 0
        )

    best_df = best_df.reindex(
        columns=train_cols,
        fill_value=0
    )


    # =====================================================
    # FINAL PREDICTION
    # =====================================================

    X_scaled_best = X_scaler.transform(best_df)

    pred_scaled = model.predict(X_scaled_best)

    final_pred = y_scaler.inverse_transform(
        pred_scaled
    )[0]


    # =====================================================
    # OUTPUT
    # =====================================================

    st.success("Optimization Completed")

    # -----------------------------------------------------
    # TARGET
    # -----------------------------------------------------

    st.subheader("Target Properties")

    target_df = pd.DataFrame({

        "Property": ["UTS", "YS", "EL"],

        "Target": [
            target[0],
            target[1],
            target[2]
        ],

        "Predicted": [
            final_pred[0],
            final_pred[1],
            final_pred[2]
        ]
    })

    st.dataframe(target_df)


    # -----------------------------------------------------
    # BEST COMPOSITION
    # -----------------------------------------------------

    st.subheader("Best Composition")

    composition_data = []

    for k, v in final_params.items():

        if k not in [
            'S4','S8','T6','T7',
            'W','A','A1','A2','A3','A4'
        ]:

            composition_data.append({
                "Element": k,
                "Value": round(v, 4)
            })

    composition_df = pd.DataFrame(composition_data)

    st.dataframe(composition_df)

