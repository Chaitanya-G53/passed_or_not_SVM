import os
import pickle
import math

import numpy as np
from flask import Flask, render_template, request

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "passed_or_not.pkl")

app = Flask(__name__)

# --------------------------------------------------------------------------------
# LOAD MODEL ONCE AT STARTUP
# --------------------------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Could not find 'passed_or_not.pkl' at {MODEL_PATH}. "
        "Make sure it is committed to the same repo folder as app.py."
    )

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

FEATURES = list(getattr(model, "feature_names_in_", [
    "gender", "age", "study_hours_per_week", "attendance_rate", "parent_education",
    "internet_access", "extracurricular", "previous_score", "final_score"
]))
CLASSES = list(getattr(model, "classes_", ["No", "Yes"]))

PARENT_EDU_LABELS = {
    0: "No Formal Education",
    1: "High School",
    2: "Bachelor's Degree",
    3: "Master's or Higher",
}


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def gauge_dasharray(pct, radius=70):
    pct = max(0.0, min(1.0, pct))
    circumference = 2 * math.pi * radius
    dash = circumference * pct
    return round(dash, 1), round(circumference, 1)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    form_values = {
        "gender": "Female", "age": 17, "study_hours": 10, "attendance": 80.0,
        "parent_education": 2, "internet_access": "Yes", "extracurricular": "Yes",
        "previous_score": 65, "final_score": 50,
    }

    if request.method == "POST":
        form_values = {
            "gender": request.form.get("gender", "Female"),
            "age": int(request.form.get("age", 17)),
            "study_hours": float(request.form.get("study_hours", 10)),
            "attendance": float(request.form.get("attendance", 80.0)),
            "parent_education": int(request.form.get("parent_education", 2)),
            "internet_access": request.form.get("internet_access", "Yes"),
            "extracurricular": request.form.get("extracurricular", "Yes"),
            "previous_score": float(request.form.get("previous_score", 65)),
            "final_score": float(request.form.get("final_score", 50)),
        }

        raw = {
            "gender": 1 if form_values["gender"] == "Male" else 0,
            "age": form_values["age"],
            "study_hours_per_week": form_values["study_hours"],
            "attendance_rate": form_values["attendance"],
            "parent_education": form_values["parent_education"],
            "internet_access": 1 if form_values["internet_access"] == "Yes" else 0,
            "extracurricular": 1 if form_values["extracurricular"] == "Yes" else 0,
            "previous_score": form_values["previous_score"],
            "final_score": form_values["final_score"],
        }
        row = np.array([[raw[f] for f in FEATURES]])

        pred_label = model.predict(row)[0]
        pred_pass = (pred_label == "Yes")

        # SVC was trained with probability=False, so no predict_proba is available.
        # Use the decision function (distance from the separating hyperplane) and
        # squash it with a sigmoid to get an interpretable, monotonic confidence score.
        decision_score = float(model.decision_function(row)[0])
        confidence = sigmoid(decision_score)
        if not pred_pass:
            confidence = 1 - confidence
        # confidence here always represents "confidence in the predicted class"
        pass_confidence = sigmoid(decision_score)  # confidence specifically toward "Yes"

        dash, circumference = gauge_dasharray(pass_confidence)
        gauge_color = "#2ecc71" if pred_pass else "#eb5757"

        # ---- Rule-based study guidance -------------------------------------
        tips = []
        if form_values["attendance"] < 75:
            tips.append(("warn", "📅 **Low attendance** — aim for at least 75–80% attendance; consistent class presence strongly correlates with passing."))
        else:
            tips.append(("good", "📅 Attendance is solid — keep it up."))

        if form_values["study_hours"] < 8:
            tips.append(("warn", "📚 **Low weekly study time** — try to build up to at least 8–10 focused study hours per week."))
        else:
            tips.append(("good", "📚 Study hours look reasonable for steady progress."))

        if form_values["previous_score"] < 50:
            tips.append(("warn", "📝 **Previous score is low** — revisit foundational topics from earlier coursework before moving ahead."))
        else:
            tips.append(("good", "📝 Previous academic performance is a positive indicator."))

        if form_values["internet_access"] == "No":
            tips.append(("warn", "🌐 **No internet access** — this can limit access to study resources; consider offline materials or library access."))

        if form_values["extracurricular"] == "No":
            tips.append(("good", "🎯 No extracurriculars reported — more time may be available for focused study."))

        if pred_pass:
            tips.append(("good", "🎓 **Overall model outlook: likely to pass** — maintain current habits and monitor progress regularly."))
        else:
            tips.append(("warn", "🎓 **Overall model outlook: at risk of not passing** — consider a structured study plan, tutoring support, and regular check-ins with instructors."))

        result = {
            "pred_label": pred_label,
            "pred_pass": pred_pass,
            "confidence_pct": round(pass_confidence * 100, 1),
            "gauge_color": gauge_color,
            "dash": dash,
            "circumference": circumference,
            "tips": tips,
        }

    return render_template(
        "index.html",
        form_values=form_values,
        result=result,
        parent_edu_labels=PARENT_EDU_LABELS,
    )


@app.route("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
