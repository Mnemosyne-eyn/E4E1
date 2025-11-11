# environmental_analyzer.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import os

# --- Step 1: Simulate 7 Days of Hourly Environmental Data ---
hours = pd.date_range("2025-01-01", periods=24*7, freq="H")

data = {
    "Temperature (°C)": 20 + 5*np.sin(np.linspace(0, 3*np.pi, len(hours))) + np.random.randn(len(hours)),
    "Humidity (%)": 60 + 10*np.random.randn(len(hours)),
    "Light (lux)": np.abs(500*np.sin(np.linspace(0, 6*np.pi, len(hours)))) + np.random.randn(len(hours))*20
}

df = pd.DataFrame(data, index=hours)

# Add artificial anomalies (optional, for testing)
df.iloc[50, 0] += 15   # spike in temperature
df.iloc[120, 1] -= 30  # drop in humidity
df.iloc[140, 2] += 1000  # spike in light

# --- Step 2: Detect Anomalies with Isolation Forest ---
features = df[["Temperature (°C)", "Humidity (%)", "Light (lux)"]]
model = IsolationForest(contamination=0.1, random_state=42)
df["anomaly"] = model.fit_predict(features)

# --- Step 3: Summary Statistics ---
summary = {
    "Average Temperature (°C)": df["Temperature (°C)"].mean(),
    "Average Humidity (%)": df["Humidity (%)"].mean(),
    "Average Light (lux)": df["Light (lux)"].mean(),
    "Total Anomalies": (df["anomaly"] == -1).sum(),
}
summary_df = pd.DataFrame([summary])
print("\n📊 Environmental Summary Report:")
print(summary_df)

# --- Step 4: Plot and Visualize Anomalies ---
plt.figure(figsize=(10, 4))
plt.plot(df.index, df["Temperature (°C)"], label="Temperature", color="orange")
plt.scatter(
    df.index[df["anomaly"] == -1],
    df[df["anomaly"] == -1]["Temperature (°C)"],
    color="blue",
    label="Anomalies"
)
plt.legend()
plt.title("Temperature with Detected Anomalies")
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.tight_layout()
plt.show()

# --- Step 5: Save Outputs ---
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

df.to_csv(os.path.join(output_dir, "environmental_anomalies.csv"))
summary_df.to_csv(os.path.join(output_dir, "summary_report.csv"), index=False)

print("\n✅ Data and summary saved in the 'outputs' folder:")
print(" - environmental_anomalies.csv")
print(" - summary_report.csv")

