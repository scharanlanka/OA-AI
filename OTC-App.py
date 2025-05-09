import streamlit as st
import pandas as pd
import joblib
import requests
from io import BytesIO

# ─── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_URL = "https://otc-only-model.s3.amazonaws.com/otc_classifier_no_postpain.pkl"

@st.cache(allow_output_mutation=True, show_spinner=False)
def load_artifacts():
    otc_pre = joblib.load("otc_preprocessor_no_postpain.pkl")
    pain_model = joblib.load("pain_reduction_model.pkl")
    weeks_model = joblib.load("weeks_to_effect_model.pkl")

    r = requests.get(MODEL_URL)
    r.raise_for_status()
    otc_model = joblib.load(BytesIO(r.content))

    df = pd.read_csv("OTC-Data.csv", skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        'Best OTC': 'best_otc', 'OTCSleep': 'otc_sleep', 'OTC Cause': 'otc_cause',
        'OTC PainLocation': 'otc_pain_location', 'OTC PainTime': 'otc_pain_time',
        'OTC CocomtSymptom': 'otc_cocomt_symptom', 'Gender': 'gender',
        'Age': 'age', 'Height': 'height', 'Weight': 'weight',
        'Ethnicity': 'ethnicity', 'Race': 'race'
    })
    return otc_pre, otc_model, pain_model, weeks_model, df

# Load artifacts
otc_pre, otc_model, pain_model, weeks_model, df_full = load_artifacts()

st.title("\U0001F3E5 OTC Knee Pain Recommender")

# ─── PATIENT PROFILE ───────────────────────────────────────────────────────────
st.header("Patient Profile")
age       = st.text_input("Age", value="", placeholder="Greater than 50 required")
gender = st.selectbox("Gender", [ "Male", "Female", "Non-binary / third gender", "Prefer not to say"])
race = st.selectbox("Race", [ "White", "Black or African American", "Asian", "Native American", "Other"])
ethnicity = st.selectbox("Hispanic Origin/Ethnicity", ["Yes", "No"])

weight    = st.text_input("Weight (lbs)", value="", placeholder="e.g. 150")
height    = st.text_input("Height (inches)", value="", placeholder="e.g. 65")

# ─── PAIN LEVEL ────────────────────────────────────────────────────────────────
st.header("Pain Level")
pain_level = st.text_input("Current pain level (1 = low, 10 = high)", value="", placeholder="Enter 1–10")

# ─── CONTEXTUAL FIELDS ─────────────────────────────────────────────────────────
st.header("Pain Context")
pain_location = st.selectbox("Where do you feel your knee pain?", [
    "", "In the front of your knee", "All over the knee", "Close to the surface above or behind your knee",
    "Deeper inside your knee", "In multiple parts of your knee or leg", "None of the above"])

pain_time = st.selectbox("When do you feel pain?", [
    "", "When moving or bending (better with rest)", "First thing in the morning", "More pain at night after activity",
    "During bad weather", "When stressed/anxious/tired", "When unwell", "None of the above"])

symptoms = st.multiselect("Accompanying symptoms", [
    "", "Dull pain", "Throbbing pain", "Sharp pain", "Swelling", "Stiffness", "Redness and warmth",
    "Instability or weakness", "Popping or crunching noises", "Limited range of motion",
    "Locking of the knee joint", "Inability to bear weight", "Fever", "Disabling pain", "Others", "None"])

st.header("Sleep & Other Joint Pain")
sleep = st.selectbox("Do you experience any of these?", ["", "Abnormal sleep pattern", "Pain at other joint(s)", "None of the above"])

st.header("Likely Cause of Pain")
cause = st.selectbox("What caused your knee pain?", [
    "", "Overweight or obesity", "Injuries (ligaments/cartilage/bone fractures)",
    "Medical conditions (arthritis, gout, infections, tendonitis, bursitis)",
    "Aging (osteoarthritis)", "Repeated stress (overuse)",
    "Other conditions (patellofemoral syndrome, lupus, rheumatoid arthritis)",
    "None of the above", "Don’t know"])

# ─── PREDICTION ────────────────────────────────────────────────────────────────
if st.button("Get OTC Recommendations"):
    required = [age, gender, race, ethnicity, weight, height,
                pain_level, pain_location, pain_time, sleep, cause]
    if not all(required):
        st.error("Please fill in every field.")
    else:
        try:
            age_v = int(age)
            w_v   = float(weight)
            h_v   = float(height)
            pl_v  = int(pain_level)
        except ValueError:
            st.error("Age, weight, height, and pain level must be numeric.")
            st.stop()
        
        if age_v < 50:
            st.error("This tool is designed for patients aged 50 and above.")
            st.stop()

        input_df = pd.DataFrame([{
            'otc_prepain': pl_v,
            'age':         age_v,
            'height':      h_v,
            'weight':      w_v,
            'gender':      gender,
            'race':        race,
            'ethnicity':   ethnicity,
            'otc_pain_location': pain_location,
            'otc_pain_time':     pain_time,
            'otc_cocomt_symptom': ",".join(symptoms) if symptoms else "",
            'otc_sleep':          sleep,
            'otc_cause':          cause
        }])

        # OTC prediction
        Xp = otc_pre.transform(input_df)
        probs = otc_model.predict_proba(Xp)[0]
        classes = otc_model.classes_
        top3 = probs.argsort()[-3:][::-1]

        st.subheader("Top 3 OTC Recommendations")
        for i in top3:
            st.write(f"- **{classes[i]}**: { 310*probs[i]:.1f}% confidence")

        # Pain reduction + weeks prediction (top-1 only)
        try:
            reg_input = input_df[['otc_prepain', 'age', 'height', 'weight', 'gender', 'ethnicity', 'race']].copy()
            reg_input['otc_usetime'] = 4  # simulate 4 weeks usage

            predicted_reduction = pain_model.predict(reg_input)[0]
            predicted_weeks = weeks_model.predict(reg_input)[0]

            st.success(f"\n\n✨ By following the above recommendation, you may reduce your pain by **{predicted_reduction:.1f} points** in about **{predicted_weeks:.1f} weeks**.")
        except Exception as e:
            st.warning(f"Could not estimate pain reduction or weeks to effect: {e}")
