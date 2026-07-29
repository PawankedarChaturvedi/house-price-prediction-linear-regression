import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score
import numpy as np
import os
import kagglehub
# %load_ext cudf.pandas
from kagglehub import KaggleDatasetAdapter
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import joblib
import streamlit as st
import os

# --------------------------------------------------
# Load Model & Pipeline
# --------------------------------------------------

MODEL_FILE = os.path.join(os.getcwd(), "Models", "model.pkl")
PIPELINE_FILE = os.path.join(os.getcwd(), "Models", "pipeline.pkl")

model = joblib.load(MODEL_FILE)
pipeline = joblib.load(PIPELINE_FILE)

# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Prediction App")

tab1, tab2 = st.tabs(["Single Prediction", "Bulk Prediction"])

# ==================================================
# SINGLE PREDICTION
# ==================================================

with tab1:

    st.subheader("Enter  Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        city = st.selectbox(
            "City",
            ["Mumbai", "Pune", "Delhi", "Bangalore", "Chennai"]
        )

        property_type = st.selectbox(
            "Property Type",
            [
                "Studio",
                "Apartment",
                "Row House",
                "Independent House",
                "Villa",
                "Penthouse"
            ]
        )

        bhk = st.number_input(
            "BHK",
            min_value=1,
            max_value=10,
            value=2
        )

        bathrooms = st.number_input(
            "Bathrooms",
            min_value=1,
            max_value=10,
            value=2
        )

    with col2:

        balconies = st.number_input(
            "Balconies",
            min_value=0,
            max_value=10,
            value=1
        )

        built_up_area = st.number_input(
            "Built-up Area",
            min_value=100.0,
            value=1000.0
        )

        carpet_area = st.number_input(
            "Carpet Area",
            min_value=100.0,
            value=850.0
        )

        floor_number = st.number_input(
            "Floor Number",
            min_value=0,
            value=3
        )

    with col3:

        total_floors = st.number_input(
            "Total Floors",
            min_value=1,
            value=10
        )

        floor_category = st.selectbox(
            "Floor Category",
            [
                "Ground",
                "Low (1-4)",
                "Mid (5-9)",
                "High (10-19)",
                "Top (20+)"
            ]
        )

        facing = st.selectbox(
            "Facing",
            [
                "East",
                "West",
                "North",
                "South"
            ]
        )

        furnishing_status = st.selectbox(
            "Furnishing Status",
            [
                "Unfurnished",
                "Semi-Furnished",
                "Fully Furnished"
            ]
        )

    st.divider()

    col4, col5, col6 = st.columns(3)

    with col4:

        property_age = st.number_input(
            "Property Age",
            min_value=0,
            value=5
        )

        parking_spaces = st.number_input(
            "Parking Spaces",
            min_value=0,
            value=1
        )

        swimming_pool = st.selectbox(
            "Swimming Pool",
            [0, 1]
        )

    with col5:

        lift_available = st.selectbox(
            "Lift Available",
            [0, 1]
        )

        distance_to_city_center_km = st.number_input(
            "Distance to City Center (KM)",
            min_value=0.0,
            value=5.0
        )

        transaction_type = st.selectbox(
            "Transaction Type",
            [
                "Resale",
                "New"
            ]
        )

    with col6:

        locality_tier = st.selectbox(
            "Locality Tier",
            [
                "Affordable",
                "Standard",
                "Premium"
            ]
        )

        price_category = st.selectbox(
            "Price Category",
            [
                "Low",
                "High"
            ]
        )

    if st.button("Predict House Price"):

        input_df = pd.DataFrame([{
            "city": city,
            "locality_tier": locality_tier,
            "property_type": property_type,
            "bhk": bhk,
            "bathrooms": bathrooms,
            "balconies": balconies,
            "built_up_area": built_up_area,
            "carpet_area": carpet_area,
            "floor_number": floor_number,
            "total_floors": total_floors,
            "floor_category": floor_category,
            "facing": facing,
            "furnishing_status": furnishing_status,
            "property_age": property_age,
            "parking_spaces": parking_spaces,
            "swimming_pool": swimming_pool,
            "lift_available": lift_available,
            "distance_to_city_center_km": distance_to_city_center_km,
            "transaction_type": transaction_type,
            "price_category": price_category
        }])

        transformed = pipeline.transform(input_df)

        prediction = model.predict(transformed)[0]

        st.success(
            f"Predicted House Price: ₹ {prediction:,.2f} Lakhs"
        )


# ==================================================
# BULK PREDICTION
# ==================================================

with tab2:

    st.subheader("Upload File")

    uploaded_file = st.file_uploader(
        "Upload CSV / Excel",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_file:

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        else:
            df = pd.read_excel(uploaded_file)

        st.write("Preview")

        st.dataframe(df.head())

        expected_columns = pipeline.feature_names_in_

        missing_cols = set(expected_columns) - set(df.columns)

        if missing_cols:

            st.error(
                f"Missing Columns: {missing_cols}"
            )

        else:

            transformed = pipeline.transform(df)

            predictions = model.predict(transformed)

            df["Predicted House Price"] = predictions

            st.success("Prediction Completed")

            st.dataframe(df.head())

            csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Results",
                data=csv,
                file_name="Predicted_House_Prices.csv",
                mime="text/csv"
            )