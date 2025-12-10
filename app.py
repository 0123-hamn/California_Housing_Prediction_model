import streamlit as st
import pandas as pd
import joblib
import os

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def predict_values(input_df):
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    transformed = pipeline.transform(input_df)
    preds = model.predict(transformed)
    input_df["median_house_value"] = preds

    return input_df




st.title("🏡 California Housing Prediction App")
st.write("Upload a CSV file to predict median house value.")


if not os.path.exists(MODEL_FILE) or not os.path.exists(PIPELINE_FILE):
    st.error("❌ Model or pipeline file missing! Make sure model.pkl and pipeline.pkl exist.")
else:
    uploaded_file = st.file_uploader("Upload input.csv", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### 📄 Uploaded Data Preview", df.head())

        if st.button("Predict"):
            with st.spinner("Running prediction..."):
                output_df = predict_values(df)

            st.success("Prediction Complete!")
            st.write("### 🔍 Output Preview", output_df.head())

            csv = output_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="⬇️ Download Predictions CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )
