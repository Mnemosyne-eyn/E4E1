# Environmental Data Analyzer

## Overview
The **Environmental Data Analyzer** is a Python-based project that simulates environmental sensor data (temperature, humidity, and light) over a week and detects anomalies using machine learning. The project demonstrates skills in:

- Data simulation and preprocessing
- Time-series data visualization
- Anomaly detection using **Isolation Forest**
- Python libraries: `pandas`, `numpy`, `matplotlib`, `scikit-learn`
- Automated output generation and reporting

This project is a great example of hands-on data analysis and mini-project development for research or engineering applications.

---

## Features

- **Simulates environmental data:** Generates hourly temperature, humidity, and light data for 7 days.
- **Detects anomalies:** Identifies unusual readings using Isolation Forest.
- **Visualizations:** Plots sensor trends with anomalies highlighted.
- **Automatic output:** Saves CSV files:
  - `environmental_anomalies.csv` → complete dataset with anomalies
  - `summary_report.csv` → averages and total anomalies
- **Easy to extend:** Can be connected to real sensor data in future (e.g., Raspberry Pi or Arduino).

---

## Installation

1. Clone this repository:

```bash
git clone https://github.com/<your-username>/EnvironmentalAnalyzer.git
cd EnvironmentalAnalyzer

