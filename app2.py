import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import base64

# Function to set background and font color
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            color: white;
        }}
        /* Input labels and other text */
        label, .css-10trblm, .st-bf, .st-cz, .st-emotion-cache-1c7y2kd, .st-emotion-cache-1kyxreq {{
            color: white
            !important;
        }}
        /* Sidebar text */
        .css-1d391kg, .css-1lcbmhc {{
            color:blue !important;
        }}
        /* Header, subheader, title */
        h1 {{
            color: white !important;
        }}
        h2{{
            color:white !important;
        }}
        h6{{
            color:white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Load models and scaler
rf_model = joblib.load('random_forest_model1.pkl')
dt_model = joblib.load('decision_tree_model1.pkl')
gan_generator = load_model('gan_generator1.h5')
scaler = joblib.load('scaler1.pkl')

# Label mapping
label_mapping = {0: 'normal', 1: 'dos', 2: 'r2l', 3: 'probe', 4: 'u2r'}

# Login check
def login(username, password):
    return username == 'Admin' and password == '123'

# Main app
def main():
    st.title("Network Intrusion Detection System")
    add_bg_from_local(r"/Users/krishnavamsi/Downloads/Intrusion detection 2/CODE new/FRONTEND/WallpaperDog-20598720.jpg")  # Replace with your image file

    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if not st.session_state['logged_in']:
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if login(username, password):
                st.session_state['logged_in'] = True
                st.success("Login successful")
            else:
                st.error("Invalid username or password")
    else:
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Choose a page", ["ML Prediction", "GAN Prediction"])

        if page == "ML Prediction":
            st.header("ML Prediction")
            model_choice = st.selectbox("Select a model", ["Random Forest", "Decision Tree"])
            
            inputs = {}
            columns = ['duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
                       'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
                       'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
                       'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
                       'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                       'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                       'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                       'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']
            
            col1, col2 = st.columns(2)
            for i, col in enumerate(columns):
                with col1 if i % 2 == 0 else col2:
                    inputs[col] = st.number_input(f"{col}", step=0.01, format="%.5f")

            if st.button("Predict"):
                input_data = pd.DataFrame([inputs])
                scaled_input = scaler.transform(input_data)

                if model_choice == "Random Forest":
                    prediction = rf_model.predict(scaled_input)[0]
                else:
                    prediction = dt_model.predict(scaled_input)[0]

                prediction_label = label_mapping.get(prediction, "Unknown")
                
                # Custom styled prediction result (sky blue text)
                st.markdown(
                    f"<h4 style='color:#00BFFF;'>Prediction using {model_choice}: {prediction_label}</h4>",
                    unsafe_allow_html=True
                )

        elif page == "GAN Prediction":
            st.header("GAN Prediction")
            noise_dimension = 32
            num_samples = st.number_input("Number of samples to generate", min_value=1, max_value=100, value=10)

            if st.button("Generate"):
                noise = np.random.normal(0, 1, (num_samples, noise_dimension))
                generated_samples = gan_generator.predict(noise)

                st.write("Generated Samples:")
                st.dataframe(generated_samples)

                scaled_generated_samples = scaler.transform(generated_samples)
                rf_predictions_generated = rf_model.predict(scaled_generated_samples)
                dt_predictions_generated = dt_model.predict(scaled_generated_samples)

                rf_predictions_mapped = [label_mapping.get(pred, "Unknown") for pred in rf_predictions_generated]
                dt_predictions_mapped = [label_mapping.get(pred, "Unknown") for pred in dt_predictions_generated]

                st.write("Random Forest Predictions on Generated Data:")
                st.write(rf_predictions_mapped)
                st.write("Decision Tree Predictions on Generated Data:")
                st.write(dt_predictions_mapped)

if __name__ == "__main__":
    main()
