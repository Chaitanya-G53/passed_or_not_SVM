import os
import joblib
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load Model
MODEL_PATH = "passed_or_not.pkl"
model = None

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

# HTML/CSS UI Layout
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Pass / Fail Prediction System</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            --card-bg: rgba(255, 255, 255, 0.05);
            --card-border: rgba(255, 255, 255, 0.12);
            --primary-accent: #6366f1;
            --primary-hover: #4f46e5;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --input-bg: rgba(15, 23, 42, 0.6);
            --success-color: #10b981;
            --danger-color: #ef4444;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 850px;
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--card-border);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }

        .header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(to right, #a5b4fc, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.25rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .form-group label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .form-control {
            background: var(--input-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 0.75rem 1rem;
            color: var(--text-main);
            font-size: 0.95rem;
            outline: none;
            transition: all 0.2s ease;
        }

        .form-control:focus {
            border-color: var(--primary-accent);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25);
        }

        select.form-control option {
            background-color: #0f172a;
            color: #f8fafc;
        }

        .submit-btn {
            grid-column: 1 / -1;
            margin-top: 1rem;
            background: linear-gradient(135deg, var(--primary-accent), var(--primary-hover));
            color: white;
            border: none;
            border-radius: 12px;
            padding: 1rem;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px -5px rgba(99, 102, 241, 0.4);
        }

        .result-box {
            margin-top: 2rem;
            padding: 1.5rem;
            border-radius: 16px;
            text-align: center;
            display: none;
            animation: fadeIn 0.4s ease-in-out forwards;
        }

        .result-box.pass {
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.4);
            color: #34d399;
        }

        .result-box.fail {
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.4);
            color: #f87171;
        }

        .result-title {
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>

    <div class="container">
        <div class="header">
            <h1>Student Performance Analytics</h1>
            <p>Enter the evaluation metrics to predict student outcome</p>
        </div>

        <form id="predictionForm" class="form-grid">
            <div class="form-group">
                <label for="gender">Gender</label>
                <select id="gender" name="gender" class="form-control" required>
                    <option value="1">Male</option>
                    <option value="0">Female</option>
                </select>
            </div>

            <div class="form-group">
                <label for="age">Age</label>
                <input type="number" id="age" name="age" class="form-control" placeholder="e.g. 18" min="10" max="100" required>
            </div>

            <div class="form-group">
                <label for="study_hours">Study Hours / Week</label>
                <input type="number" step="0.1" id="study_hours" name="study_hours" class="form-control" placeholder="e.g. 20" required>
            </div>

            <div class="form-group">
                <label for="attendance_rate">Attendance Rate (%)</label>
                <input type="number" step="0.1" id="attendance_rate" name="attendance_rate" class="form-control" placeholder="e.g. 85.5" min="0" max="100" required>
            </div>

            <div class="form-group">
                <label for="parent_education">Parent Education Level</label>
                <select id="parent_education" name="parent_education" class="form-control" required>
                    <option value="0">High School</option>
                    <option value="1">Bachelor's</option>
                    <option value="2">Master's</option>
                    <option value="3">Doctorate</option>
                </select>
            </div>

            <div class="form-group">
                <label for="internet_access">Internet Access</label>
                <select id="internet_access" name="internet_access" class="form-control" required>
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                </select>
            </div>

            <div class="form-group">
                <label for="extracurricular">Extracurricular Activities</label>
                <select id="extracurricular" name="extracurricular" class="form-control" required>
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                </select>
            </div>

            <div class="form-group">
                <label for="previous_score">Previous Score</label>
                <input type="number" step="0.1" id="previous_score" name="previous_score" class="form-control" placeholder="e.g. 75.0" min="0" max="100" required>
            </div>

            <div class="form-group">
                <label for="final_score">Final Assessment Score</label>
                <input type="number" step="0.1" id="final_score" name="final_score" class="form-control" placeholder="e.g. 82.0" min="0" max="100" required>
            </div>

            <button type="submit" class="submit-btn">Evaluate Outcome</button>
        </form>

        <div id="resultBox" class="result-box">
            <div id="resultTitle" class="result-title"></div>
            <p id="resultDesc"></p>
        </div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const data = Object.fromEntries(formData.entries());

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                const result = await response.json();
                const resultBox = document.getElementById('resultBox');
                const resultTitle = document.getElementById('resultTitle');
                const resultDesc = document.getElementById('resultDesc');

                resultBox.style.display = 'block';

                if (result.status === 'success') {
                    if (result.prediction === 'Yes') {
                        resultBox.className = 'result-box pass';
                        resultTitle.textContent = '🎉 PASSED';
                        resultDesc.textContent = 'The model predicts the student is likely to pass.';
                    } else {
                        resultBox.className = 'result-box fail';
                        resultTitle.textContent = '⚠️ NOT PASSED';
                        resultDesc.textContent = 'The model predicts the student needs additional support.';
                    }
                } else {
                    resultBox.className = 'result-box fail';
                    resultTitle.textContent = 'Error';
                    resultDesc.textContent = result.message || 'Failed to process prediction.';
                }
            } catch (err) {
                alert('An error occurred while connecting to the server.');
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded correctly.'}), 500

    try:
        data = request.get_json()

        # Extract features matching model order:
        # gender, age, study_hours_per_week, attendance_rate, parent_education, internet_access, extracurricular, previous_score, final_score
        features = [
            float(data.get('gender', 0)),
            float(data.get('age', 0)),
            float(data.get('study_hours', 0)),
            float(data.get('attendance_rate', 0)),
            float(data.get('parent_education', 0)),
            float(data.get('internet_access', 0)),
            float(data.get('extracurricular', 0)),
            float(data.get('previous_score', 0)),
            float(data.get('final_score', 0))
        ]

        prediction = model.predict([features])[0]
        return jsonify({'status': 'success', 'prediction': str(prediction)})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
