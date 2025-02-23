import pandas as pd
from prophet import Prophet

# Load dataset
df = pd.read_csv("data/crime_data.csv")

# Prepare data
df['DATE OCC'] = pd.to_datetime(df['DATE OCC'])
df = df.groupby('DATE OCC').size().reset_index(name='crime_count')

# Prophet model
df.rename(columns={'DATE OCC': 'ds', 'crime_count': 'y'}, inplace=True)
model = Prophet()
model.fit(df)

# Predict future crimes
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# Save forecast
forecast.to_csv("data/crime_forecast.csv", index=False)
print("Forecast saved!")
