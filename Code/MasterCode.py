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

MODEL_FILE = os.path.join(os.getcwd(), "Models\\model.pkl")
PIPELINE_FILE  = os.path.join(os.getcwd(), "Models\\pipeline.pkl")
INPUT_FILES = os.path.join(os.getcwd(), "Input Files")
input_Files = [file for file in os.listdir(INPUT_FILES) if os.path.isfile(os.path.join(INPUT_FILES, file)) and not file.startswith("~")][0]
OUTPUT_DIR = os.path.join(os.getcwd(), "Output")

def readFileType(file, skiprows=0):
    # Check if the file is an Excel file (.xlsx or .xls)
    if file.lower().endswith(".xlsx") or file.lower().endswith(".xls"):
        # Read the Excel file into a DataFrame, skipping the specified number of rows
        df = pd.read_excel(file, skiprows=skiprows)
        return df

    # Check if the file is a CSV file
    elif file.lower().endswith(".csv"):
        # Read the CSV file into a DataFrame, skipping the specified number of rows
        df = pd.read_csv(file, skiprows=skiprows)
        return df
    else:
        raise ValueError(f"Unsupported File Type : {file}")

def build_pipeline(num_attributes, cat_ordinal_attributes, cat_one_hot_attributes):
    num_pipeline = Pipeline([
        ("imputers", SimpleImputer(strategy="mean"))
    ])

    cat_ordinal_pipeline = Pipeline([
        ("imputers", SimpleImputer(strategy="most_frequent")),

        ("ordinal_encoder", OrdinalEncoder(
            categories=[
                ['Studio', 'Apartment', 'Row House', 'Independent House', 'Villa', 'Penthouse'],  # property_type
                ['Ground', 'Low (1-4)', 'Mid (5-9)', 'High (10-19)', 'Top (20+)'],               # floor_category
                ['Unfurnished', 'Semi-Furnished', 'Fully Furnished'],                             # furnishing_status
                ['Resale', 'New'],                                                                # transaction_type
                ['Affordable', 'Standard', 'Premium'],                                          # locality_tier
                ["Low", "High"]
            ]
        ))
    ])

    cat_one_hot_pipeline = Pipeline([
        ("imputers", SimpleImputer(strategy="most_frequent")),

        ("one_hot_encoder", OneHotEncoder(drop='first',handle_unknown='ignore', sparse_output=False))
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attributes ),
        ("cat_ordinal", cat_ordinal_pipeline, cat_ordinal_attributes),
        ("cat_one_hot", cat_one_hot_pipeline, cat_one_hot_attributes )
    ])

    return full_pipeline


if not (os.path.exists(MODEL_FILE) and os.path.exists(PIPELINE_FILE)):
    file_name_in_dataset = "housing_price_dataset.csv"
    house_Data = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS,"ameyac11/housing-price-prediction-dataset",file_name_in_dataset)
    house_Data = house_Data[["city", "locality_tier", "property_type", "bhk", "bathrooms",
        "balconies", "built_up_area", "carpet_area", "floor_number", "total_floors",
        "floor_category", "facing", "furnishing_status", "property_age", "parking_spaces",
        "swimming_pool", "lift_available", "distance_to_city_center_km", "transaction_type", "price_in_lakhs", "price_category"]]


    house_Data["house_price_category"] = pd.cut(house_Data["price_in_lakhs"],
                                                bins = [0, 150.0, 300.0, 450.0, 600.0, np.inf],
                                                labels=[1, 2, 3, 4, 5] )
    
    split_model = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, _ in split_model.split(house_Data, house_Data["house_price_category"]):
        housing = house_Data.iloc[train_index].drop(columns = "house_price_category")

    housing_features = housing.drop(columns = "price_in_lakhs")
    housing_labels = housing["price_in_lakhs"].copy()

    num_attributes = housing_features.select_dtypes(include=["int64", "float64"]).columns.tolist()
    
    cat_ordinal_attributes = [
        'property_type',
        'floor_category',
        'furnishing_status',
        'transaction_type',
        'locality_tier',
        "price_category"
    ]
    cat_one_hot_attributes = ["city", "facing"]
    pipeline = build_pipeline(num_attributes, cat_ordinal_attributes, cat_one_hot_attributes)
    house_Filtered_Data = pipeline.fit_transform(housing_features)
    
    model = RandomForestRegressor(random_state=42)
    model.fit(house_Filtered_Data, housing_labels)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

    print("Model Train and Saved Succesfully")

else:
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = readFileType(os.path.join(INPUT_FILES, input_Files))
    expected_columns = pipeline.feature_names_in_

    missing_cols = set(expected_columns) - set(input_data.columns)

    if missing_cols:
        raise ValueError(
            f"Missing columns: {missing_cols}"
        )
    transformed_Input = pipeline.transform(input_data)
    predicted_value = model.predict(transformed_Input)
    input_data["Housing Price Predicted Value"] = predicted_value

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join( OUTPUT_DIR, "Predicted Housing Value.csv" )
    input_data.to_csv(output_file, index=False)  
    print("Inference complete. Results saved to output folder")
