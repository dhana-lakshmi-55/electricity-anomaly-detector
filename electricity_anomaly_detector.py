import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Read CSV file
data = pd.read_csv("electricity_usage.csv")

# Create model
model = IsolationForest(contamination=0.2, random_state=42)

# Detect anomalies
data["Anomaly"] = model.fit_predict(data[["Units"]])

print(data)

# Display only anomalies
print("\nAbnormal Electricity Usage:")
print(data[data["Anomaly"] == -1])

# Plot graph
colors = ["red" if x == -1 else "blue" for x in data["Anomaly"]]

plt.figure(figsize=(8,5))
plt.scatter(data["Day"], data["Units"], c=colors, s=100)

plt.xlabel("Day")
plt.ylabel("Electricity Units")
plt.title("Electricity Usage Anomaly Detector")

plt.grid(True)
plt.show()
