# Path setup, and access the config.yml file, datasets folder & trained models
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel
from strictyaml import YAML, load

import delivery_time_model

# Project Directories
PACKAGE_ROOT = Path(delivery_time_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
print(CONFIG_FILE_PATH)

DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"


class AppConfig(BaseModel):
    """
    Application-level config.
    """

    package_name: str
    training_data_file: str
    testing_data_file: str
    pipeline_name: str
    pipeline_save_file: str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """

    target: str
    features: List[str]
    unused_fields: List[str]
    
    Delivery_person_Age_var: str
    Delivery_person_Ratings_var: str
    Restaurant_latitude_var: str
    Restaurant_longitude_var: str
    Delivery_location_latitude_var: str
    Delivery_location_longitude_var: str
    Weatherconditions_var: str
    Road_traffic_density_var: str
    Vehicle_condition_var: str
    Type_of_order_var: str
    Type_of_vehicle_var: str
    multiple_deliveries_var: str
    Festival_var: str
    City_area_var: str
    City_var: str
    day_of_week_var : str
    is_weekend_var : str
    quarter_var : str
    yr_var: str
    mnth_var: str
    Distance_var: str
    order_prepare_time_var : str
        
    weather_mappings: Dict[str, int]
    traff_den_mappings: Dict[str, int]
    order_type_mappings: Dict[str, int]
    vehicle_mappings: Dict[str, int]
    festival_mappings: Dict[str, int]
    city_area_mappings: Dict[str, int]
    # city_mappings: Dict[str, int]
    # yr_mappings: Dict[str, int]
    mnth_mappings: Dict[str, int]
    
    test_size:float
    random_state: int
    n_estimators: int
    max_depth: int


class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    ml_config: ModelConfig


def find_config_file() -> Path:
    """Locate the configuration file."""
    
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
        
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config = AppConfig(**parsed_config.data),
        ml_config = ModelConfig(**parsed_config.data),
    )

    return _config


config = create_and_validate_config()