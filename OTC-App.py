import streamlit as st
import pandas as pd
import joblib
# Load artifacts (preprocessor and classifier)
@st.cache(allow_output_mutation=True)
def load_artifacts():
    pre = joblib.load('otc_preprocessor_no_postpain.pkl')
    clf = joblib.load('otc_classifier_no_postpain.pkl')
    df = pd.read_csv('OTC-Data.csv', skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        'Best OTC':'best_otc','OTCSleep':'otc_sleep','OTC Cause':'otc_cause',
        'OTC PainLocation':'otc_pain_location','OTC PainTime':'otc_pain_time',
        'OTC CocomtSymptom':'otc_cocomt_symptom','Gender':'gender',
        'Age':'age','Height':'height','Weight':'weight',
        'Ethnicity':'ethnicity','Race':'race'
    })
    vc = df['best_otc'].value_counts()
    rare = vc[vc < 5].index
    df['best_otc'] = df['best_otc'].apply(lambda x: 'Other' if x in rare else x)
    return pre, clf, df

preprocessor, model, df_full = load_artifacts()

st.title("🏥 OTC Knee Pain Recommender")

# Input fields without defaults
st.header("Patient Profile")
age = st.text_input("Age", value="", placeholder="Enter your age in years")
gender_opts = ["", *list(df_full['gender'].unique())]
gender = st.selectbox("Gender", gender_opts)
race_opts = ["", *list(df_full['race'].unique())]
race = st.selectbox("Race", race_opts)
ethnicity_opts = ["", *list(df_full['ethnicity'].unique())]
ethnicity = st.selectbox("Hispanic Origin/Ethnicity", ethnicity_opts)
weight = st.text_input("Weight (lbs)", value="", placeholder="Enter weight in lbs")
height = st.text_input("Height (inches)", value="", placeholder="Enter height in inches")

st.header("Pain Level")
pain_level = st.text_input("Current pain level (1 = low, 10 = high)", value="", placeholder="Enter a number 1–10")

st.header("Pain Context")
loc_opts = ["", 'In the front of your knee', 'All over the knee',
            'Close to the surface above or behind your knee',
            'Deeper inside your knee', 'In multiple parts of your knee or leg',
            'None of the above']
pain_location = st.selectbox("Where do you feel your knee pain?", loc_opts)

time_opts = ["", 'When you are moving or bending, better with rest',
             'First thing in the morning when you wake up',
             'More pain at night after activity', 'During bad weather',
             'When stressed/anxious/tired', 'When unwell', 'None of the above']
pain_time = st.selectbox("When do you feel pain?", time_opts)

st.header("Accompanying Symptoms")
symp_opts = ['', 'Dull pain','Throbbing pain','Sharp pain','Swelling','Stiffness',
             'Redness and warmth','Instability or weakness','Popping or crunching noises',
             'Limited range of motion','Locking of the knee joint','Inability to bear weight',
             'Fever','Disabling pain','Others','None']
symptoms = st.multiselect("Select any that apply:", symp_opts)
symptom_str = ",".join(symptoms) if symptoms else ""

st.header("Sleep & Other Joint Pain")
sleep_opts = ['', 'Abnormal sleep pattern','Pain at other joint(s)','None of the above']
sleep = st.selectbox("Do you experience any of these?", sleep_opts)

st.header("Likely Cause of Pain")
cause_opts = ['', 'Overweight or obesity','Injuries (ligaments, cartilage, bone fractures)',
              'Medical conditions (arthritis, gout, infections, tendonitis, bursitis)',
              'Aging (osteoarthritis)','Repeated stress (overuse)',
              'Other conditions (patellofemoral pain syndrome, lupus, rheumatoid arthritis)',
              'None of the above','Don’t know']
cause = st.selectbox("What caused your knee pain?", cause_opts)

# Predict
if st.button("Get OTC Recommendations"):
    # Validate all inputs
    if not all([age, gender, race, ethnicity, weight, height, pain_level,
                pain_location, pain_time, sleep, cause]):
        st.error("Please fill in all fields before submitting.")
    else:
        # Convert numeric
        try:
            age_val = int(age)
            weight_val = float(weight)
            height_val = float(height)
            pain_val = int(pain_level)
        except ValueError:
            st.error("Please enter valid numbers for age, weight, height, and pain level.")
            st.stop()

        input_df = pd.DataFrame([{  
            'otc_prepain': pain_val,
            'age': age_val,
            'height': height_val,
            'weight': weight_val,
            'gender': gender,
            'race': race,
            'ethnicity': ethnicity,
            'otc_pain_location': pain_location,
            'otc_pain_time': pain_time,
            'otc_cocomt_symptom': symptom_str,
            'otc_sleep': sleep,
            'otc_cause': cause
        }])

        X_proc = preprocessor.transform(input_df)
        probs = model.predict_proba(X_proc)[0]
        classes = model.classes_
        top_idx = probs.argsort()[-3:][::-1]

        st.subheader("Top 3 OTC Recommendations")
        for i in top_idx:
            st.write(f"- **{classes[i]}**")
