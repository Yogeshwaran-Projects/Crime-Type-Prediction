CREATE TABLE crime_predictions (
    id SERIAL PRIMARY KEY,
    area INT NOT NULL,
    weapon_used INT NOT NULL,
    victim_age INT NOT NULL,
    victim_sex VARCHAR(5) NOT NULL,
    predicted_crime VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
