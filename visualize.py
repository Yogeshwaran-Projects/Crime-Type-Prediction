import pandas as pd
import folium

df = pd.read_csv("data/crime_data.csv")

crime_map = folium.Map(location=[34.0522, -118.2437], zoom_start=10)
for _, row in df.iterrows():
    folium.Marker([row['LAT'], row['LON']], popup=row['Crm Cd Desc'], icon=folium.Icon(color="red")).add_to(crime_map)

crime_map.save("web_app/static/crime_map.html")
print("Crime heatmap generated successfully!")
