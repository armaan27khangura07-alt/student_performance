
import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os

# --------------------------------
# 1. LOAD DATASET AND MODEL
# --------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "student_performance_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "my_model.pkl")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError("Dataset CSV file is missing.")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Trained model my_model.pkl is missing.")

df = pd.read_csv(DATASET_PATH)
model = joblib.load(MODEL_PATH)

FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "Attendance_Rate",
    "Prior_GPA",
    "LMS_Logins",
    "Study_Hours_Per_Week",
    "Assignment_Submission_Rate",
    "Engagement_Score"
]

# Check dataset columns
missing = [col for col in FEATURE_COLUMNS if col not in df.columns]

if missing:
    raise ValueError(f"Missing dataset columns: {missing}")

# --------------------------------
# 2. PREDICTION FUNCTION
# --------------------------------

def predict(age, gender, attendance, gpa,
            logins, study_hours, assignment_rate, engagement):

    input_data = pd.DataFrame([[
        age,
        gender,
        attendance,
        gpa,
        logins,
        study_hours,
        assignment_rate,
        engagement
    ]], columns=FEATURE_COLUMNS)

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    confidence = np.max(probabilities) * 100

    return str(prediction), f"{confidence:.2f}%"

# --------------------------------
# 3. DASHBOARD DESIGN
# --------------------------------

css = """
.gradio-container {
    max-width: 1150px !important;
    margin: auto !important;
}

#header {
    background: linear-gradient(120deg, #14213d, #007f86);
    padding: 28px;
    border-radius: 18px;
    text-align: center;
    color: white;
    margin-bottom: 18px;
}

#header h1 {
    color: white;
    font-size: 32px;
}

#header p {
    color: #e0f2f1;
    font-size: 15px;
}

.card {
    border: 1px solid #dce5ed !important;
    border-radius: 15px !important;
    padding: 15px !important;
}

#predict {
    background: #087e8b !important;
    color: white !important;
    font-weight: bold !important;
    font-size: 17px !important;
    border-radius: 10px !important;
}
"""

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="blue"
    ),
    css=css,
    title="Student Performance Dashboard"
) as demo:

    gr.HTML("""
    <div id="header">
        <h1>🎓 Student Performance Prediction</h1>
        <p>Machine Learning Based Academic Performance Dashboard</p>
        <p>Analyze student information and predict academic performance</p>
    </div>
    """)

    # Dataset overview
    with gr.Row():

        with gr.Column():
            gr.Markdown("### 📊 Dataset Overview")
            gr.Markdown(
                f"**Total Records:** {len(df)}  \n"
                f"**Input Features:** {len(FEATURE_COLUMNS)}  \n"
                f"**Target:** Final Academic Performance"
            )

        with gr.Column():
            gr.Markdown("### 🤖 Model Information")
            gr.Markdown(
                "**Algorithm:** Gaussian Naive Bayes  \n"
                "**Prediction:** Academic Performance  \n"
                "**Model:** Trained Machine Learning Model"
            )

    gr.Markdown("---")
    gr.Markdown("## 👨‍🎓 Enter Student Details")

    with gr.Row():

        with gr.Column(elem_classes="card"):

            gr.Markdown("### Personal Information")

            age = gr.Number(
                label="Age",
                value=float(df["Age"].median())
            )

            gender = gr.Radio(
                choices=[
                    ("Female", 0),
                    ("Male", 1)
                ],
                label="Gender",
                value=0
            )

            gr.Markdown("### Academic Background")

            gpa = gr.Slider(
                float(df["Prior_GPA"].min()),
                float(df["Prior_GPA"].max()),
                value=float(df["Prior_GPA"].median()),
                label="Prior GPA"
            )

            attendance = gr.Slider(
                float(df["Attendance_Rate"].min()),
                float(df["Attendance_Rate"].max()),
                value=float(df["Attendance_Rate"].median()),
                label="Attendance Rate (%)"
            )

        with gr.Column(elem_classes="card"):

            gr.Markdown("### Learning & Engagement")

            logins = gr.Number(
                label="LMS Logins",
                value=float(df["LMS_Logins"].median())
            )

            study_hours = gr.Slider(
                float(df["Study_Hours_Per_Week"].min()),
                float(df["Study_Hours_Per_Week"].max()),
                value=float(df["Study_Hours_Per_Week"].median()),
                label="Study Hours Per Week"
            )

            assignment_rate = gr.Slider(
                float(df["Assignment_Submission_Rate"].min()),
                float(df["Assignment_Submission_Rate"].max()),
                value=float(df["Assignment_Submission_Rate"].median()),
                label="Assignment Submission Rate (%)"
            )

            engagement = gr.Slider(
                float(df["Engagement_Score"].min()),
                float(df["Engagement_Score"].max()),
                value=float(df["Engagement_Score"].median()),
                label="Engagement Score"
            )

    gr.Markdown("---")

    predict_btn = gr.Button(
        "🔍 Predict Academic Performance",
        variant="primary",
        elem_id="predict"
    )

    gr.Markdown("## 📈 Prediction Result")

    result = gr.Textbox(
        label="Predicted Category",
        interactive=False
    )

    confidence = gr.Textbox(
        label="Prediction Probability",
        interactive=False
    )

    predict_btn.click(
        fn=predict,
        inputs=[
            age, gender, attendance, gpa,
            logins, study_hours,
            assignment_rate, engagement
        ],
        outputs=[result, confidence]
    )

    gr.Markdown("""
    ---
    **Student Performance ML Project | Gaussian Naive Bayes**

    Disclaimer: This is an educational prediction model.
    Its output is not a definitive assessment of a student's ability.
    """)

# --------------------------------
# 4. LAUNCH APPLICATION
# --------------------------------

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)
