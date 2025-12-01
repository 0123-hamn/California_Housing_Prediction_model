import streamlit as st
import pandas as pd
import numpy as np
import os
# import joblib

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"


def build_eda_pipeline(num_attribs, cat_attribs):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])

    return full_pipeline



def train_model():

    if os.path.exists(MODEL_FILE):
        return "Model already exists. Delete model.pkl to retrain."

    housing = pd.read_csv("housing.csv")

    housing['income_cat'] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, _ in split.split(housing, housing['income_cat']):
        housing = housing.loc[train_index].drop("income_cat", axis=1)

    labels = housing["median_house_value"].copy()
    features = housing.drop("median_house_value", axis=1)

    # Fill missing
    features["total_bedrooms"] = features["total_bedrooms"].fillna(
        features["total_bedrooms"].median()
    )

    num_attribs = features.drop("ocean_proximity", axis=1).columns.tolist()
    cat_attribs = ["ocean_proximity"]

    pipeline = build_eda_pipeline(num_attribs, cat_attribs)
    prepared_data = pipeline.fit_transform(features)

    model = RandomForestRegressor(random_state=42)
    model.fit(prepared_data, labels)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

    return "Model trained successfully!"



def predict_values(input_df):
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    transformed = pipeline.transform(input_df)
    preds = model.predict(transformed)
    input_df["median_house_value"] = preds

    return input_df



st.title("🏡 California Housing Prediction App")
st.write("Train model and run inference using Streamlit.")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", ["Train Model", "Predict (Inference)"])

# TRAIN PAGE
if page == "Train Model":
    st.header("🔧 Train the Model")

    if st.button("Train Now"):
        with st.spinner("Training model..."):
            message = train_model()
        st.success(message)

    st.info("• Place 'housing.csv' in the same folder as this app.\n"
            "• Model will be saved as model.pkl and pipeline.pkl")



elif page == "Predict (Inference)":
    st.header("📊 Predict Median House Values")

    uploaded_file = st.file_uploader("Upload input.csv", type=["csv"])

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data", input_df.head())

        if st.button("Predict"):
            if not os.path.exists(MODEL_FILE):
                st.error("Model not found! Please train the model first.")
            else:
                with st.spinner("Running prediction..."):
                    output_df = predict_values(input_df)

                st.success("Prediction complete!")
                st.write("### Output Preview", output_df.head())

                # Download button
                csv = output_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Predictions CSV",
                    data=csv,
                    file_name="output.csv",
                    mime="text/csv"
                )

