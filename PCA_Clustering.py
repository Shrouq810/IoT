"""
Title:Hydrologic–Hydrodynamic Flood Inundation Modeling Enhanced by IoT Water-Level Measurements in a Data-Scarce Basin
Author: Shrouq abuismail, github: ShrouqAbuismail
Date:20/04/2026
Description: A Machine-learning based tool for the paper titled "Hydrologic–Hydrodynamic Flood Inundation Modeling Enhanced by IoT Water-Level Measurements in a Data-Scarce Basin" 
"""

"""
Title:Hydrologic–Hydrodynamic Flood Inundation Modeling Enhanced by IoT Water-Level Measurements in a Data-Scarce Basin
Author: Shrouq abuismail, github: ShrouqAbuismail
Date:20/04/2026
Description: PCA and Clustering analysis code for the paper titled "Hydrologic–Hydrodynamic Flood Inundation Modeling Enhanced by IoT Water-Level Measurements in a Data-Scarce Basin" 
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ============================
# SETTINGS
# ============================
FEATURES = ["RMSE", "Bias", "PeakErr", "PeakTime"]
SCENARIO_COL = "Combo"

METRIC_COLS_BY_EVENT = {
    f"T{i}": {
        "RMSE":     f"RMSE_T{i}",
        "Bias":     f"Bias_T{i}",
        "PeakErr":  f"PeakBias_T{i}",
        "PeakTime": f"PeakTime_T{i}",
    }
    for i in range(1, 16)
}

# ============================
# LOAD + PREP DATA
# ============================
def load_event_data(excel_path, sheet_name):
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    event_rows = []
    for idx, row in df.iterrows():
        scenario = row[SCENARIO_COL]

        for ev, cols in METRIC_COLS_BY_EVENT.items():
            event_rows.append({
                "Scenario": scenario,
                "Event": ev,
                "RMSE": row[cols["RMSE"]],
                "Bias": row[cols["Bias"]],
                "PeakErr": row[cols["PeakErr"]],
                "PeakTime": row[cols["PeakTime"]],
            })

    event_df = pd.DataFrame(event_rows)
    event_df = event_df.dropna(subset=FEATURES)

    return event_df

# ============================
# SCALING + PCA
# ============================
def run_pca(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    return X_pca, pca

# ============================
# CLUSTERING
# ============================
def run_kmeans(X, k=3):
    model = KMeans(n_clusters=k, random_state=42, n_init=50)
    labels = model.fit_predict(X)
    centers = model.cluster_centers_
    return labels, centers

# ============================
# DISTANCE CALCULATION
# ============================
def compute_distances(X_points, centers):
    distances = []
    for p in X_points:
        row = [np.sqrt(np.sum((p - c) ** 2)) for c in centers]
        distances.append(row)
    return np.array(distances)

# ============================
# SCENARIO-LEVEL DISTANCE (YOUR KEY STEP)
# ============================
def compute_scenario_distances(event_df, X_pca, centers):
    event_df["PC1"] = X_pca[:, 0]
    event_df["PC2"] = X_pca[:, 1]

    # median location per scenario
    centroids = (
        event_df.groupby("Scenario")[["PC1", "PC2"]]
        .median()
        .reset_index()
    )

    rows = []
    for _, r in centroids.iterrows():
        scenario = r["Scenario"]
        point = np.array([r["PC1"], r["PC2"]])

        for i, c in enumerate(centers):
            dist = np.sqrt(np.sum((point - c) ** 2))
            rows.append({
                "Scenario": scenario,
                "Cluster": f"C{i+1}",
                "Distance": dist
            })

    return pd.DataFrame(rows)

# ============================
# MAIN PIPELINE
# ============================
def run_pipeline(excel_path, sheet_name, k=3):
    # Load data
    event_df = load_event_data(excel_path, sheet_name)

    # Features
    X = event_df[FEATURES].values

    # PCA
    X_pca, pca = run_pca(X)

    # Clustering
    labels, centers = run_kmeans(X_pca, k=k)
    event_df["Cluster"] = labels

    # Distances
    dist_df = compute_scenario_distances(event_df, X_pca, centers)

    return event_df, dist_df, pca

# ============================
# RUN
# ============================
if __name__ == "__main__":
    excel_path = "your_file.xlsx"
    sheet_name = "scaled2"

    event_df, dist_df, pca = run_pipeline(excel_path, sheet_name)

    # Save results
    event_df.to_csv("event_clusters.csv", index=False)
    dist_df.to_csv("scenario_distances.csv", index=False)

    print("Done.")