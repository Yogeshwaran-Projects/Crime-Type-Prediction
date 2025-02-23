from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'crime_rate_model.pkl')
ENCODER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'label_encoder.pkl')

if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError("Model or encoder file not found! Train the model first.")

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

crime_severity = {
    'Theft': 'Low',
    'Burglary': 'Medium',
    'Assault': 'High',
    'Homicide': 'Severe',
    'Robbery': 'High',
    'Vandalism': 'Low',
    'Fraud': 'Medium',
    'Arson': 'Severe',
    'Drug Offense': 'Medium',
    'Kidnapping': 'Severe',
    'Sexual Assault': 'Severe',
    'Vehicle Theft': 'Medium',
    'Other': 'Unknown'  
}

severity_explanations = {
    'Low': 'Crimes in this category generally cause minimal harm and involve non-violent activities.',
    'Medium': 'These crimes may cause financial loss, distress, or potential harm to individuals.',
    'High': 'Involves violent or dangerous behavior that poses serious risks to individuals.',
    'Severe': 'These crimes can result in loss of life, extreme violence, or major public safety threats.',
    'Unknown': 'This crime type does not have a predefined severity classification.'
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        area = int(data['area'])
        weapon_used = int(data['weapon_used'])
        victim_age = int(data['victim_age'])
        victim_sex = data['victim_sex']

        gender_mapping = {'M': 0, 'F': 1, 'X': 2, 'H': 3, '-': 4}
        victim_sex_mapped = gender_mapping.get(victim_sex, 4)

        input_data = pd.DataFrame({
            'AREA': [area],
            'Weapon Used Cd': [weapon_used],
            'Vict Age': [victim_age],
            'Vict Sex': [victim_sex_mapped]
        })

        predicted_label = model.predict(input_data)[0]
        predicted_crime = label_encoder.inverse_transform([predicted_label])[0]

        crime_keywords = {
            'assault': 'Assault',
            'homicide': 'Homicide',
            'robbery': 'Robbery',
            'theft': 'Theft',
            'burglary': 'Burglary',
            'fraud': 'Fraud',
            'arson': 'Arson',
            'vandalism': 'Vandalism',
            'drug': 'Drug Offense',
            'kidnapping': 'Kidnapping',
            'sexual': 'Sexual Assault',
            'vehicle': 'Vehicle Theft'
        }

        predicted_crime_normalized = "Other"
        for keyword, category in crime_keywords.items():
            if keyword in predicted_crime.lower():
                predicted_crime_normalized = category
                break

        severity = crime_severity.get(predicted_crime_normalized, 'Unknown')
        severity_reason = severity_explanations.get(severity, 'No information available.')

        return jsonify({
            'predictedCrime': predicted_crime,
            'severityLevel': severity,
            'severityReason': severity_reason
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
