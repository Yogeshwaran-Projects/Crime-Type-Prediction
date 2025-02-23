import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Load dataset
df = pd.read_csv("data/crime_data.csv")

# Selecting relevant features
features = ['AREA', 'Weapon Used Cd', 'Vict Age', 'Vict Sex']
target = 'Crm Cd Desc'  # Crime type

# Drop missing values
df = df.dropna(subset=features + [target])

# Encode categorical variables
label_encoder = LabelEncoder()
df[target] = label_encoder.fit_transform(df[target])  # Encode crime type
df['Vict Sex'] = df['Vict Sex'].map({'M': 0, 'F': 1, 'X': 2, 'H': 3, '-': 4})

# Standardize numerical values
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Split dataset
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Deep Learning Model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save model & encoder
model.save("models/crime_model.h5")
joblib.dump(label_encoder, "models/label_encoder.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Model training complete and saved!")
