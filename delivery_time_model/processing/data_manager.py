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
import boto3
from io import StringIO
import os

##  Pre-Pipeline Preparation


def read_s3_csv(bucket_name, folder_prefix):
	
    # Create an S3 client
    s3 = boto3.client(
            's3',
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            region_name='ap-south-1'
        )
     
    # List objects in the specified folder (prefix)
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_prefix)
    
    # Get the list of CSV files in the folder
    csv_files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.csv')]

    # Initialize an empty list to store DataFrames
    dfs = []

    # Loop through each CSV file and read it into a DataFrame
    for file_key in csv_files:
        # Fetch the file from S3
        response = s3.get_object(Bucket=bucket_name, Key=file_key)

        # Read the file content
        csv_content = response['Body'].read().decode('utf-8')

        # Convert to a pandas DataFrame
        df = pd.read_csv(StringIO(csv_content))

        # Append the DataFrame to the list
        dfs.append(df)
        
    # Concatenate all DataFrames into one
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df
	
# Extract year and month from the date column and create two another columns

def add_new_features(dataframe: pd.DataFrame):

    df = dataframe.copy()
    # print("start of add_new_features function")
    # print(df.iloc[0].to_dict())
    # convert 'dteday' column to Datetime datatype
    df["Order_Date_tmp"] = pd.to_datetime(df["Order_Date"], format='%d-%m-%Y')
    
    # Add new features
    df["day_of_week"] = df["Order_Date_tmp"].dt.dayofweek # monday = 0 and sunday = 6
    df["is_weekend"] = df["day_of_week"].apply(lambda x:1 if x in [5,6] else 0) # 5 = saturday and 6 = sunday
    df["quarter"] = df["Order_Date_tmp"].dt.quarter
    df['yr'] = df["Order_Date_tmp"].dt.year
    df['mnth'] = df["Order_Date_tmp"].dt.month_name()
    df.drop(['Order_Date_tmp'], axis=1, inplace=True)
    # print("From add_new_features functions")
    # df["Order_Date"] = dataframe['Order_Date']
    # print(df.iloc[0].to_dict())
    return df
def calculate_picked_time(row):
    # Add a day offset if 'Time_Order_picked' is less than 'Time_Orderd'
    if row['Time_Order_picked'] < row['Time_Orderd']:
        return row['Order_Date'] + pd.DateOffset(days=1) + row['Time_Order_picked']
    else:
        return row['Order_Date'] + row['Time_Order_picked']
    
def calculate_time_diff(dataframe: pd.DataFrame):
    
    # print("Inside calculate_time_diff function")
    
    df = dataframe.copy()
    # print(df.iloc[0].to_dict())
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
       
    # Handle Nan text values
    df["Weatherconditions"] = df["Weatherconditions"].replace("nan", np.nan, regex=True).replace("NaN", np.nan, regex=True)
    df["Road_traffic_density"] = df["Road_traffic_density"].replace("nan", np.nan, regex=True).replace("NaN", np.nan, regex=True)
    df["Festival"] = df["Festival"].replace("nan", np.nan, regex=True).replace("NaN", np.nan, regex=True)
    df["City_area"] = df["City_area"].replace("nan", np.nan, regex=True).replace("NaN", np.nan, regex=True)
    df["Delivery_person_Age"] = df["Delivery_person_Age"].replace("nan", np.nan, regex=True).replace("NaN", np.nan, regex=True)
    df["Delivery_person_Ratings"] = df["Delivery_person_Ratings"].replace("nan", np.nan, regex=True).replace("NaN", np.nan, regex=True)

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

    return df
        
def pre_pipeline_trans(*, data_frame: pd.DataFrame) -> pd.DataFrame:
       
    #Derive order_prepare_time
    data_frame = calculate_time_diff(dataframe=data_frame)    
    
    data_frame = add_new_features(dataframe=data_frame)
    
    #To fix error for predict.py TypeError: ufunc 'radians' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe'' 
    data_frame['Restaurant_latitude'] = pd.to_numeric(data_frame['Restaurant_latitude'], errors='coerce')
    data_frame['Restaurant_longitude'] = pd.to_numeric(data_frame['Restaurant_longitude'], errors='coerce')
    data_frame['Delivery_location_latitude'] = pd.to_numeric(data_frame['Delivery_location_latitude'], errors='coerce')
    data_frame['Delivery_location_longitude'] = pd.to_numeric(data_frame['Delivery_location_longitude'], errors='coerce')
    
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
   
    # Issue while doing predction as it expects str
    data_frame[['Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude']] = data_frame[['Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude']].astype(str)
    
    # Strip spaces from the object type  columns
    data_frame = data_frame.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        
    #data_frame = get_year_and_month(dataframe = data_frame, date_var = config.ml_config.date_var)
    if "City_area" not in data_frame.columns:
        data_frame.rename(columns={"City": "City_area"}, inplace=True)
    
    #Pre Pipeline data munging
    data_frame = data_munging(dataframe = data_frame)

	# Strip spaces from the object type  columns as there is space for NaN
    data_frame = data_frame.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

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
    
    #Rename city and Time_taken columns
    data_frame.rename(columns = {"City":"City_area","Time_taken(min)":"Time_taken"},inplace=True)
    
    #Pre Pipeline data munging
    data_frame = data_munging(dataframe = data_frame)
    
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
    
    # Issue while doing predction as it expects float
    data_frame[['Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude']] = data_frame[['Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude']].astype(str)

    data_frame['Time_taken'] = data_frame['Time_taken'].str.replace(r'\(min\)', '', regex=True).astype(float)  # Ensure target variable is float
    
    # Strip spaces from the object type  columns
    data_frame = data_frame.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    # Drop unnecessary fields
    for field in config.ml_config.unused_fields:
        if field in data_frame.columns:
            data_frame.drop(labels = field, axis=1, inplace=True)    

    return data_frame

def pre_pipeline_preparation_test(*, data_frame: pd.DataFrame) -> pd.DataFrame:

    # Strip spaces from the object type  columns
    data_frame = data_frame.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    #Derive order_prepare_time
    data_frame = calculate_time_diff(dataframe=data_frame)    
    
    data_frame = add_new_features(dataframe=data_frame)
    
    #Rename city and Time_taken columns
    data_frame.rename(columns = {"City":"City_area","Time_taken(min)":"Time_taken"},inplace=True)
    
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

    #Handle 
    data_frame = data_munging(dataframe = data_frame)
	
	# Remove (min) from the target column data and convert to float datatype
    data_frame['Time_taken'] = data_frame['Time_taken'].str.replace(r'\(min\)', '', regex=True).astype(float)  # Ensure target variable is float
    
    # Strip spaces from the object type
    data_frame = data_frame.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    print("End of pipeline preprocess function")
    return data_frame

def read_data_file():
    import os
    import dvc.api
    import pandas as pd

    repo = 'https://' + os.environ['GH_USERNAME'] + ':' + os.environ['GH_ACCESS_TOKEN'] + '@github.com/kpavan2004/dvc-driver-demand-capstone'
    data_revision = os.environ['DATA_VERSION']
    remote_config = {
        'access_key_id': os.environ["AWS_ACCESS_KEY_ID"],
        'secret_access_key': os.environ["AWS_SECRET_ACCESS_KEY"],
    }
    try:
        with dvc.api.open('data/train.csv', repo=repo, rev=data_revision, remote_config=remote_config) as file:
            df = pd.read_csv(file)
        return df
    except Exception as e:
        print(f"Error occurred while reading dvc training data: {e}")
        raise

def _load_raw_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    return dataframe

def load_dataset(*, file_name: str) -> pd.DataFrame:
    train_df = read_data_file()  # pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    bucket_name = "pk-capstone-bucket-01"
    object_key = "new_data/"
    new_df =  read_s3_csv(bucket_name,object_key)
    if new_df.empty :
        combined_df = train_df
    else:
        combined_df = pd.concat([train_df, new_df], ignore_index=True)
    transformed = pre_pipeline_preparation(data_frame = combined_df)
    return transformed

def load_dataset_test(*, file_name: str) -> pd.DataFrame:
    dataframe = read_data_file() # pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    transformed = pre_pipeline_trans(data_frame = dataframe)
    return transformed

def load_dataset_test1(*, file_name: str) -> pd.DataFrame:
    train_df = read_data_file()  #pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    bucket_name = "pk-capstone-bucket-01"
    object_key = "new_data/"
    new_df =  read_s3_csv(bucket_name,object_key)
    if new_df.empty :
        combined_df = train_df
    else:
        combined_df = pd.concat([train_df, new_df], ignore_index=True)
    transformed = pre_pipeline_preparation_test(data_frame = combined_df)
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
