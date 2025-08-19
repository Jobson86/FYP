import sys
import web3
import subprocess
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import streamlit as st
import folium
import webbrowser
import pandas as pd
import hashlib
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from streamlit_folium import folium_static

# --- Generate Synthetic Dataset ---
def generate_synthetic_dataset(num_records=100000, split_ratio=0.8):
    np.random.seed(42)
    
    vehicle_ids = np.random.randint(1000, 9999, num_records)
    gps_latitudes = np.random.uniform(35.0, 40.0, num_records)
    gps_longitudes = np.random.uniform(-125.0, -120.0, num_records)
    packet_loss = np.random.uniform(0, 1, num_records)
    csi_values = np.random.uniform(-100, 0, num_records)  # Simulated Channel State Information
    timestamps = [datetime.utcnow() - timedelta(seconds=np.random.randint(0, 3600)) for _ in range(num_records)]
    
    attack_labels = []
    attacked_vehicles = []
    for i in range(num_records):
        if np.random.rand() < 0.1:
            attack_labels.append("Sybil Attack")
            vehicle_ids[i] = vehicle_ids[i - 1]  # Duplicate vehicle ID
            attacked_vehicles.append((vehicle_ids[i], attack_labels[i], timestamps[i]))
        elif np.random.rand() < 0.1:
            attack_labels.append("Black Hole Attack")
            packet_loss[i] = 1.0  # 100% packet loss
            attacked_vehicles.append((vehicle_ids[i], attack_labels[i], timestamps[i]))
        elif np.random.rand() < 0.1:
            attack_labels.append("GPS Spoofing")
            gps_latitudes[i] += np.random.uniform(0.5, 1.0)  # Sudden GPS jump
            gps_longitudes[i] += np.random.uniform(0.5, 1.0)
            attacked_vehicles.append((vehicle_ids[i], attack_labels[i], timestamps[i]))
        else:
            attack_labels.append("Normal")
    df_train = pd.DataFrame({
        'Vehicle ID': vehicle_ids[:int(split_ratio * num_records)],
        'GPS Latitude': gps_latitudes[:int(split_ratio * num_records)],
        'GPS Longitude': gps_longitudes[:int(split_ratio * num_records)],
        'Packet Loss': packet_loss[:int(split_ratio * num_records)],
        'CSI Value': csi_values[:int(split_ratio * num_records)],
        'Timestamp': timestamps[:int(split_ratio * num_records)],
        'Attack Type': attack_labels[:int(split_ratio * num_records)]  # Training set has labels
    })
    df_test = pd.DataFrame({
        'Vehicle ID': vehicle_ids[int(split_ratio * num_records):],
        'GPS Latitude': gps_latitudes[int(split_ratio * num_records):],
        'GPS Longitude': gps_longitudes[int(split_ratio * num_records):],
        'Packet Loss': packet_loss[int(split_ratio * num_records):],
        'CSI Value': csi_values[int(split_ratio * num_records):],
        'Timestamp': timestamps[int(split_ratio * num_records):],
    }) 
    
    df_train.to_csv("train_vanet_attacks.csv", index=False)
    df_test.to_csv("test_vanet_attacks.csv", index=False)
    
    return df_train, df_test

# --- Folium Map Visualization ---
def display_map(df):
    m = folium.Map(location=[37.5, -122.0], zoom_start=6)
    attacked_df = df[df['Attack Type'] != "Normal"]  # Filter only attacked vehicles
    for _, row in attacked_df.iterrows():
        folium.Marker(
            location=[row['GPS Latitude'], row['GPS Longitude']],
            popup=f"Vehicle ID: {row['Vehicle ID']}\nAttack: {row['Attack Type']}",
            icon=folium.Icon(color="red")
        ).add_to(m)
    return m


# --- Proof of Work for GPS Spoofing Detection ---
def proof_of_work(nonce, difficulty=4):
    while True:
        hash_attempt = hashlib.sha256(str(nonce).encode()).hexdigest()
        if hash_attempt[:difficulty] == "0" * difficulty:
            return nonce, hash_attempt
        nonce += 1

# --- Autoencoder for Black Hole Attack Detection ---
def train_autoencoder(df_train):
    X_train = df_train[['Packet Loss']].values
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = keras.Sequential([
        keras.layers.Dense(8, activation='relu', input_shape=(1,)),
        keras.layers.Dense(4, activation='relu'),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train_scaled, X_train_scaled, epochs=10, batch_size=32)
    return model, scaler


# --- Attack Detection Function ---
def detect_attacks(df_test, autoencoder, scaler, df_train):
    df_test['Predicted Attack'] = "Normal"

    # --- Sybil Attack Detection (K-Means Clustering) ---
    kmeans = KMeans(n_clusters=2, random_state=42)
    df_test['Cluster'] = kmeans.fit_predict(df_test[['Vehicle ID', 'CSI Value']])
    
    # If multiple vehicle IDs fall into the same cluster, it's likely a Sybil attack
    for cluster in df_test['Cluster'].unique():
        cluster_data = df_test[df_test['Cluster'] == cluster]
        if cluster_data['Vehicle ID'].nunique() < len(cluster_data):
            df_test.loc[df_test['Cluster'] == cluster, 'Predicted Attack'] = "Sybil Attack"

    # --- Black Hole Attack Detection (Autoencoder) ---
    X_test = df_test[['Packet Loss']].values
    X_test_scaled = scaler.transform(X_test)

    reconstruction_error = np.abs(autoencoder.predict(X_test_scaled) - X_test_scaled)
    df_test.loc[reconstruction_error.flatten() > 0.1, 'Predicted Attack'] = "Black Hole Attack"

    # --- GPS Spoofing Detection (Proof-of-Work) ---
    for i in range(1, len(df_test)):
        lat_diff = abs(df_test.iloc[i]['GPS Latitude'] - df_test.iloc[i - 1]['GPS Latitude'])
        lon_diff = abs(df_test.iloc[i]['GPS Longitude'] - df_test.iloc[i - 1]['GPS Longitude'])
        
        if lat_diff > 0.5 or lon_diff > 0.5:
            df_test.loc[i, 'Predicted Attack'] = "GPS Spoofing"

    # --- Evaluation: Compare Predictions with True Labels ---
    df_true = pd.read_csv("test_vanet_attacks.csv")  # Load actual labels for comparison
    if 'Attack Type' in df_true.columns:
        df_test['True Attack'] = df_true['Attack Type']

        y_true = df_test['True Attack']
        y_pred = df_test['Predicted Attack']
        
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, zero_division=0)
        conf_matrix = confusion_matrix(y_true, y_pred)

        # --- Display Results in Streamlit ---
        st.subheader("📊 Evaluation Metrics")
        st.write(f"**Accuracy:** {accuracy:.4f}")
        st.text(report)

        # --- Confusion Matrix ---
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=set(y_true), yticklabels=set(y_true))
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        st.pyplot(fig)

    st.subheader("🔍 Attack Detection Results")
    st.dataframe(df_test[['Vehicle ID', 'GPS Latitude', 'GPS Longitude', 'Predicted Attack']])
    st.success("Attack detection completed!")

# --- Main Streamlit UI ---
def main():
    st.set_page_config(page_title="VANET Attack Detection", layout="wide")
    st.title("🚗 VANET Attack Detection System")
    
    df_train, df_test = generate_synthetic_dataset()
    
    st.subheader("📂 Training Dataset")
    st.dataframe(df_train.head())
    st.download_button("Download Training Data", df_train.to_csv(index=False), file_name="train_vanet_attacks.csv")
    
    st.subheader("📂 Test Dataset")
    st.dataframe(df_test.head())
    st.download_button("Download Test Data", df_test.to_csv(index=False), file_name="test_vanet_attacks.csv")
    
    autoencoder, scaler = None, None
    if st.button("Train Model"):
        with st.spinner("Training models..."):
            autoencoder,scaler = train_autoencoder(df_train)
            st.session_state.autoencoder = autoencoder
            st.session_state.scaler = scaler  # Store scaler in session state
        st.success("Training completed successfully!")
        st.success("Training completed successfully!")
    
    uploaded_file = st.file_uploader("Upload Test Dataset", type=["csv"])
    if uploaded_file is not None:
        df_test = pd.read_csv(uploaded_file)
    
    if st.button("Detect Attacks"):
        with st.spinner("Detecting attacks..."):
            detect_attacks(df_test, st.session_state.autoencoder, st.session_state.scaler,df_train)
    if st.button("Display Map"):
        st.subheader("🌍 Vehicle Locations and Attacks Map")
        folium_static(display_map(df_test))
if __name__ == "__main__":
    main()
