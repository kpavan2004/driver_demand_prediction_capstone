import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
import numpy as np
import typing as t
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from delivery_time_model import __version__ as _version
from delivery_time_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


##  Pre-Pipeline Preparation

# Extract year and month from the date column and create two another columns

def add_new_features(dataframe: pd.DataFrame):

    df = dataframe.copy()
    print("start of add_new_features function")
    print(df.iloc[0].to_dict())
    # convert 'dteday' column to Datetime datatype
    df["Order_Date_tmp"] = pd.to_datetime(df["Order_Date"], format='%d-%m-%Y')
    
    # Add new features
    df["day_of_week"] = df["Order_Date_tmp"].dt.dayofweek # monday = 0 and sunday = 6
    df["is_weekend"] = df["day_of_week"].apply(lambda x:1 if x in [5,6] else 0) # 5 = saturday and 6 = sunday
    df["quarter"] = df["Order_Date_tmp"].dt.quarter
    df['yr'] = df["Order_Date_tmp"].dt.year
    df['mnth'] = df["Order_Date_tmp"].dt.month_name()
    df.drop(['Order_Date_tmp'], axis=1, inplace=True)
    print("From add_new_features functions")
    # df["Order_Date"] = dataframe['Order_Date']
    print(df.iloc[0].to_dict())
    return df
def calculate_picked_time(row):
    # Add a day offset if 'Time_Order_picked' is less than 'Time_Orderd'
    if row['Time_Order_picked'] < row['Time_Orderd']:
        return row['Order_Date'] + pd.DateOffset(days=1) + row['Time_Order_picked']
    else:
        return row['Order_Date'] + row['Time_Order_picked']
    
def calculate_time_diff(dataframe: pd.DataFrame):
    
    print("Inside calculate_time_diff function")
    
    df = dataframe.copy()
    print(df.iloc[0].to_dict())
    # Ensure 'Order_Date' is in datetime format
    df['Order_Date'].head()
    df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='%d-%m-%Y') # 
    
    # Convert 'Time_Orderd' and 'Time_Order_picked' to timedelta
    df['Time_Orderd'] = pd.to_timedelta(df['Time_Orderd'])
    df['Time_Order_picked'] = pd.to_timedelta(df['Time_Order_picked'])
    
    # Apply the time offset logic correctly
    # df['Time_Order_picked_formatted'] = df['Order_Date'] + np.where(df['Time_Order_picked'] < df['Time_Orderd'], pd.DateOffset(days=1), pd.DateOffset(days=0)) + df['Time_Order_picked']
    # Apply the function to each row
    df['Time_Order_picked_formatted'] = df.apply(calculate_picked_time, axis=1)

    df['Time_Ordered_formatted'] = df['Order_Date'] + df['Time_Orderd']
    
    # Ensure both columns are datetime before performing subtraction
    df['Time_Order_picked_formatted'] = pd.to_datetime(df['Time_Order_picked_formatted'])
    df['Time_Ordered_formatted'] = pd.to_datetime(df['Time_Ordered_formatted'])
    
    # Calculate the order preparation time in minutes
    df['order_prepare_time'] = (df['Time_Order_picked_formatted'] - df['Time_Ordered_formatted']).dt.total_seconds() / 60
    
    # Handle null values by filling with the median
    df['order_prepare_time'] = df['order_prepare_time'].fillna(df['order_prepare_time'].median())
    
    # Drop all the time & date related columns
    df.drop(['Time_Ordered_formatted', 'Time_Order_picked_formatted'], axis=1, inplace=True)
    df["Order_Date"] = dataframe['Order_Date']
    df["Time_Orderd"] = dataframe['Time_Orderd']
    df['Time_Order_picked'] = dataframe['Time_Order_picked']
    
    return df

# using haversine method 
def haversine_distance(loc_list):
    # earth's radius in km
    R = 6371.0

    # convert lat and lon from deg to radians 
    lat1,lon1,lat2,lon2 = map(np.radians,loc_list)
    # diff between lat and lon
    d_lat = lat2 - lat1 
    d_lon = lon2 - lon1 
    # applying haversine formula 
    a = np.sin(d_lat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R * c
    return round(d,2)

def rename_label(dataframe: pd.DataFrame):
    df = dataframe.copy()
    return df

def data_munging(dataframe: pd.DataFrame):
    df = dataframe.copy()
    
    # Add new features city derived 
    print("Inside data_munging")
    print(df.iloc[0].to_dict())
    
    # Handle Nan text values
    # data_frame = data_frame.apply(lambda x: x.str.lower().replace("nan",np.nan,regex = True) if x.dtype == "object" else x)
    
    print("NUmber of null values in columns before tranformation",df.isna().sum())
    
    # df = df.apply(lambda x: x.str.lower().replace("nan",np.nan,regex = True) if x.dtype == "object" else x)
    df["Weatherconditions"] = df["Weatherconditions"].replace("nan", np.nan, regex=True).replace("NaN", np.nan, regex=True)
    df["Road_traffic_density"] = df["Road_traffic_density"].replace("nan", np.nan, regex=True).replace("NaN", np.nan, regex=True)
    df["Festival"] = df["Festival"].replace("nan", np.nan, regex=True).replace("NaN", np.nan, regex=True)
    df["City_area"] = df["City_area"].replace("nan", np.nan, regex=True).replace("NaN", np.nan, regex=True)
    df["Delivery_person_Age"] = df["Delivery_person_Age"].replace("nan", np.nan, regex=True).replace("NaN", np.nan, regex=True)
    df["Delivery_person_Ratings"] = df["Delivery_person_Ratings"].replace("nan", np.nan, regex=True).replace("NaN", np.nan, regex=True)
    
    # cols = ["Weatherconditions","Road_traffic_density","City_area","Festival","Delivery_person_Age","Delivery_person_Ratings"]
    # for col in cols:
    #     df[col].replace("nan", np.nan, regex=True).replace("NaN", np.nan, regex=True)
        
    # Need to check below later
    # df = df.apply(lambda x: x.replace("nan",np.nan,regex = True) if x.dtype == "object" else x)
    
    print("NUmber of null values in columns after converting string NaN/nan to np.nan tranformation",df.isna().sum())
    
    # For numeric columns use median as imputation for np.nan
    df["Delivery_person_Age"] = df["Delivery_person_Age"].fillna(df["Delivery_person_Age"].astype("float").median())
    median_rating = df["Delivery_person_Ratings"].astype("float").median()
    df["Delivery_person_Ratings"] = df["Delivery_person_Ratings"].fillna(median_rating)
    df["Delivery_person_Ratings"] = df["Delivery_person_Ratings"].replace(6, median_rating)
    
    # For categorical columns use mode as imputation for np.nan
    cols = ["Weatherconditions","Road_traffic_density","City_area","Festival"]
    for col in cols:
        df[col] = df[col].fillna(df[col].mode()[0])    
    
    #Derive new column city from Delivery_person_ID
    df["City"] = df["Delivery_person_ID"].str.split("RES").str[0].astype(str)    
    
    # Removing condition word from the data
    df["Weatherconditions"] = df["Weatherconditions"].str.split(" ").str[-1]
    # print(df["Weatherconditions"].value_counts())
    
    
    print("NUmber of null values in columns after tranformation",df.isna().sum())
    
    #Drop rows where restauratn lat and long are zero
    
    
    print(df.iloc[0].to_dict())
    return df
        
def pre_pipeline_trans(*, data_frame: pd.DataFrame) -> pd.DataFrame:
    
    print('columns to the input of pre_pipeline_trans function')
    print(data_frame.iloc[0].to_dict())
    
    #Derive order_prepare_time
    data_frame = calculate_time_diff(dataframe=data_frame)    
    
    data_frame = add_new_features(dataframe=data_frame)
    
    #To fix error for predict.py TypeError: ufunc 'radians' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe'' 
    data_frame['Restaurant_latitude'] = pd.to_numeric(data_frame['Restaurant_latitude'], errors='coerce')
    data_frame['Restaurant_longitude'] = pd.to_numeric(data_frame['Restaurant_longitude'], errors='coerce')
    data_frame['Delivery_location_latitude'] = pd.to_numeric(data_frame['Delivery_location_latitude'], errors='coerce')
    data_frame['Delivery_location_longitude'] = pd.to_numeric(data_frame['Delivery_location_longitude'], errors='coerce')
    
    print("Before Haversine")
    #Calcuate the distance between Restaurant and Delivery location
    # loc_cols = ["Restaurant_latitude","Restaurant_longitude","Delivery_location_latitude","Delivery_location_longitude"]
    # distance = []
    # for i in range(len(data_frame[loc_cols[0]])):
    #     location_list = [data_frame[loc_cols[j]][i] for j in range(len(loc_cols))]
    #     distance.append(haversine_distance(location_list))
    # data_frame["Distance"] = distance
    
    loc_cols = ["Restaurant_latitude", "Restaurant_longitude", "Delivery_location_latitude", "Delivery_location_longitude"]
    distance = []

    # Iterate over each row in the DataFrame
    for _, row in data_frame.iterrows():
        # Extract values for the location columns
        location_list = [row[col] for col in loc_cols]
    
        # Calculate haversine distance
        distance.append(haversine_distance(location_list))

    # Add the distances as a new column
    data_frame["Distance"] = distance

    print("After Haversine")
    print(data_frame.iloc[0].to_dict())
    
    # Issue while doing predction as it expects str
    data_frame[['Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude']] = data_frame[['Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude']].astype(str)
    
    # Strip spaces from the object type  columns
    data_frame = data_frame.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        
    #data_frame = get_year_and_month(dataframe = data_frame, date_var = config.ml_config.date_var)
    if "City_area" not in data_frame.columns:
        data_frame.rename(columns={"City": "City_area"}, inplace=True)
    
    #Pre Pipeline data munging
    data_frame = data_munging(dataframe = data_frame)
        
    print("*********************")
    print(data_frame.iloc[0].to_dict())
    print("*********************")
    # data_frame = get_city(dataframe = data_frame)
    
	# Strip spaces from the object type  columns as there is space for NaN
    data_frame = data_frame.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    #Replace Nan with float nan

       
    print(data_frame.columns)
	
	# Remove (min) from the target column data and convert to float datatype
    # data_frame['Time_taken(min)'] = data_frame['Time_taken(min)'].str.replace(r'\(min\)', '', regex=True).astype(float)  # Ensure target variable is float
    # data_frame['Time_taken'] = data_frame['Time_taken'].str.replace(r'\(min\)', '', regex=True).astype(float)  # Ensure target variable is float
    
    #Derive city from delivery person id
    # data_frame["City"] = data_frame["Delivery_person_ID"].str.split("RES").str[0].astype(str)
    
    # Drop unnecessary fields
    for field in config.ml_config.unused_fields:
        if field in data_frame.columns:
            data_frame.drop(labels = field, axis=1, inplace=True)    

    return data_frame
    
def pre_pipeline_preparation(*, data_frame: pd.DataFrame) -> pd.DataFrame:

    # Strip spaces from the object type  columns
    data_frame = data_frame.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    #Derive order_prepare_time
    data_frame = calculate_time_diff(dataframe=data_frame)    
    
    data_frame = add_new_features(dataframe=data_frame)
       
    # data_frame = rename_label(dataframe = data_frame )
    
    #Rename city and Time_taken columns
    data_frame.rename(columns = {"City":"City_area","Time_taken(min)":"Time_taken"},inplace=True)
    
    #Pre Pipeline data munging
    data_frame = data_munging(dataframe = data_frame)
    
    print("Before Haversine")
    #Calcuate the distance between Restaurant and Delivery location
    loc_cols = ["Restaurant_latitude","Restaurant_longitude","Delivery_location_latitude","Delivery_location_longitude"]
    
    #To fix error for predict.py TypeError: ufunc 'radians' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe'' 
    data_frame['Restaurant_latitude'] = pd.to_numeric(data_frame['Restaurant_latitude'], errors='coerce')
    data_frame['Restaurant_longitude'] = pd.to_numeric(data_frame['Restaurant_longitude'], errors='coerce')
    data_frame['Delivery_location_latitude'] = pd.to_numeric(data_frame['Delivery_location_latitude'], errors='coerce')
    data_frame['Delivery_location_longitude'] = pd.to_numeric(data_frame['Delivery_location_longitude'], errors='coerce')
    
    distance = []
    for i in range(len(data_frame[loc_cols[0]])):
        location_list = [data_frame[loc_cols[j]][i] for j in range(len(loc_cols))]
        distance.append(haversine_distance(location_list))
    data_frame["Distance"] = distance
    print("After Haversine")
    print(data_frame.iloc[0].to_dict())
    
    # Issue while doing predction as it expects float
    data_frame[['Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude']] = data_frame[['Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude']].astype(str)

    
    print(data_frame.columns)
	
	# Remove (min) from the target column data and convert to float datatype
    # data_frame['Time_taken(min)'] = data_frame['Time_taken(min)'].str.replace(r'\(min\)', '', regex=True).astype(float)  # Ensure target variable is float
    data_frame['Time_taken'] = data_frame['Time_taken'].str.replace(r'\(min\)', '', regex=True).astype(float)  # Ensure target variable is float
    
    # Strip spaces from the object type  columns
    data_frame = data_frame.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    #Derive city from delivery person id
    # data_frame["City"] = data_frame["Delivery_person_ID"].str.split("RES").str[0].astype(str)
    
    # Drop unnecessary fields
    for field in config.ml_config.unused_fields:
        if field in data_frame.columns:
            data_frame.drop(labels = field, axis=1, inplace=True)    

    return data_frame

def pre_pipeline_preparation_test(*, data_frame: pd.DataFrame) -> pd.DataFrame:

    print("columns in pre pipeline prearation test:*********")
    print(data_frame.columns)
    # Strip spaces from the object type  columns
    data_frame = data_frame.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    #Derive order_prepare_time
    data_frame = calculate_time_diff(dataframe=data_frame)    
    
    data_frame = add_new_features(dataframe=data_frame)
    
    # data_frame = rename_label(dataframe = data_frame )
    
    # Strip spaces from the object type
    # data_frame = data_frame.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    #Rename city and Time_taken columns
    data_frame.rename(columns = {"City":"City_area","Time_taken(min)":"Time_taken"},inplace=True)
    
    print("Before Haversine")
    #Calcuate the distance between Restaurant and Delivery location
    loc_cols = ["Restaurant_latitude","Restaurant_longitude","Delivery_location_latitude","Delivery_location_longitude"]
    
    #To fix error for predict.py TypeError: ufunc 'radians' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe'' 
    data_frame['Restaurant_latitude'] = pd.to_numeric(data_frame['Restaurant_latitude'], errors='coerce')
    data_frame['Restaurant_longitude'] = pd.to_numeric(data_frame['Restaurant_longitude'], errors='coerce')
    data_frame['Delivery_location_latitude'] = pd.to_numeric(data_frame['Delivery_location_latitude'], errors='coerce')
    data_frame['Delivery_location_longitude'] = pd.to_numeric(data_frame['Delivery_location_longitude'], errors='coerce')

    
    distance = []
    for i in range(len(data_frame[loc_cols[0]])):
        location_list = [data_frame[loc_cols[j]][i] for j in range(len(loc_cols))]
        distance.append(haversine_distance(location_list))
    data_frame["Distance"] = distance
    print("After Haversine")
    print(data_frame.iloc[0].to_dict())
    
    print(data_frame.columns)
    
    #Handle 
    data_frame = data_munging(dataframe = data_frame)
	
	# Remove (min) from the target column data and convert to float datatype
    # data_frame['Time_taken(min)'] = data_frame['Time_taken(min)'].str.replace(r'\(min\)', '', regex=True).astype(float)  # Ensure target variable is float
    data_frame['Time_taken'] = data_frame['Time_taken'].str.replace(r'\(min\)', '', regex=True).astype(float)  # Ensure target variable is float
    
    # Strip spaces from the object type
    data_frame = data_frame.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    #Derive city from delivery person id
    # data_frame["City"] = data_frame["Delivery_person_ID"].str.split("RES").str[0].astype(str)
    print("columns so far")
    print(data_frame.columns)
    # Create a list of fields to drop, excluding "nodrop"
    # fields_to_drop = [field for field in config.ml_config.unused_fields if field != "Delivery_person_ID"]
    # print("Fields to drop")
    # print(fields_to_drop)

    # Drop the fields from the data frame
    # data_frame.drop(labels=fields_to_drop, axis=1, inplace=True)

    return data_frame


def _load_raw_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    return dataframe

def load_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    transformed = pre_pipeline_preparation(data_frame = dataframe)
    return transformed

def load_dataset_test(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    transformed = pre_pipeline_trans(data_frame = dataframe)
    return transformed

def load_dataset_test1(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    transformed = pre_pipeline_preparation_test(data_frame = dataframe)
    print(transformed.columns)
    return transformed

def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous saved models. 
    This ensures that when the package is published, there is only one trained model that 
    can be called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one mapping between the package version and 
    the model version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
