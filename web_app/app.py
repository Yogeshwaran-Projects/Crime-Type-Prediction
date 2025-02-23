import os
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import tensorflow as tf

app = Flask(__name__)

# Load trained model and encoder
model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), '../models/crime_model.h5'))
label_encoder = joblib.load(os.path.join(os.path.dirname(__file__), '../models/label_encoder.pkl'))
scaler = joblib.load(os.path.join(os.path.dirname(__file__), '../models/scaler.pkl'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        area = int(request.form['area'])
        weapon_used = int(request.form['weapon_used'])
        victim_age = int(request.form['victim_age'])
        victim_sex = request.form['victim_sex']

        gender_mapping = {'M': 0, 'F': 1, 'X': 2, 'H': 3, '-': 4}
        victim_sex_mapped = gender_mapping.get(victim_sex, 4)

        input_data = pd.DataFrame([[area, weapon_used, victim_age, victim_sex_mapped]], 
                                  columns=['AREA', 'Weapon Used Cd', 'Vict Age', 'Vict Sex'])
        
        # Standardize input
        input_data = scaler.transform(input_data)

        prediction = model.predict(input_data)
        crime_type = label_encoder.inverse_transform([prediction.argmax()])[0]

        return render_template('index.html', prediction_text=f'Predicted Crime Type: {crime_type}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

# REST API for JSON response
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json
    try:
        input_data = pd.DataFrame([[
            data['area'], data['weapon_used'], data['victim_age'], data['victim_sex']
        ]], columns=['AREA', 'Weapon Used Cd', 'Vict Age', 'Vict Sex'])

        input_data = scaler.transform(input_data)
        prediction = model.predict(input_data)
        crime_type = label_encoder.inverse_transform([prediction.argmax()])[0]

        return jsonify({"crime_type": crime_type})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
