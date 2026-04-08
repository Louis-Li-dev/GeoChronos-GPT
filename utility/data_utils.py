
import pandas as pd
from geo_tools.geo_api.coordinate_lookup import *
from geo_tools.geo_api.coordinate_transform import *
from sklearn.model_selection import train_test_split
from utility.data_split import *
from utility.eval import *
from utility.functional import *
import numpy as np
from dotenv import load_dotenv
load_dotenv()

class DataProcessor:
    def __init__(self, country_cache_path='../cache/country_cache.json', 
                 lookup_cache_path='../cache/lookup_cache', grid_size=0.3):
        self.locator = CountryLocator(cache_path=country_cache_path, 
                                    lookup_cache_path=lookup_cache_path)
        self.coordinator = CoordinateGrid(grid_size)
        self.discretizer = CoordDiscretizer(output='matrix')

    def load_and_clean_data(self, file_path='../data/checkins_data.csv'):
        """Load and clean initial dataset"""
        df = pd.read_csv(file_path)
        notna_index = df[['Latitude', "Longitude", 'Uid']].dropna().index
        clean_df = df.loc[notna_index]
        return clean_df[(clean_df['Latitude'] > -90) & (clean_df['Latitude'] < 90)]

    def filter_country_data(self, country="Japan", df=None):
        """Filter data for Japan and add country information"""
        df["Country"] = self.locator.lookup(df, lat_col='Latitude', lon_col='Longitude')
        japan_df = df[df['Country'] == country].copy()
        return japan_df[japan_df['Country'] != "unknown"]
    
    # Function to assign regions based on coordinates in a DataFrame
    def get_region_coordinates(self, df, lat_col='Latitude', long_col='Longitude'):
        japan_regions = {
            "Hokkaido": {"lat_min": 41.35, "lat_max": 45.55, "lon_min": 139.75, "lon_max": 145.82},  # Northern island
            "Tohoku": {"lat_min": 36.75, "lat_max": 41.35, "lon_min": 139.75, "lon_max": 141.95},     # NE Honshu
            "Kanto": {"lat_min": 34.95, "lat_max": 36.75, "lon_min": 138.85, "lon_max": 141.05},      # Tokyo area
            "Chubu": {"lat_min": 34.55, "lat_max": 37.75, "lon_min": 136.65, "lon_max": 139.15},      # Central Honshu
            "Kinki (Kansai)": {"lat_min": 33.85, "lat_max": 35.65, "lon_min": 134.75, "lon_max": 136.25},  # Osaka area
            "Chugoku": {"lat_min": 33.85, "lat_max": 35.55, "lon_min": 131.75, "lon_max": 134.45},    # Western Honshu
            "Shikoku": {"lat_min": 32.75, "lat_max": 34.35, "lon_min": 132.55, "lon_max": 134.65},    # Shikoku island
            "Kyushu": {"lat_min": 31.05, "lat_max": 33.95, "lon_min": 129.85, "lon_max": 131.95},     # Southern island
            "Okinawa": {"lat_min": 24.05, "lat_max": 28.05, "lon_min": 123.75, "lon_max": 131.35}     # Far south islands
        }

        # Convert to DataFrame for easier handling
        regions_df = pd.DataFrame.from_dict(japan_regions, orient='index')
        regions_df.index.name = 'Region'
        # Ensure the input DataFrame has the specified lat/long columns
        if lat_col not in df.columns or long_col not in df.columns:
            raise ValueError(f"DataFrame must contain '{lat_col}' and '{long_col}' columns")
        
        # Create a copy of the input DataFrame
        result_df = df.copy()
        
        # Initialize the Region column as "Unlisted"
        result_df['Region'] = 'Unlisted'
        
        # Loop through each region and assign based on ranges
        for region, bounds in regions_df.iterrows():
            mask = (
                (result_df[lat_col] >= bounds['lat_min']) & 
                (result_df[lat_col] <= bounds['lat_max']) & 
                (result_df[long_col] >= bounds['lon_min']) & 
                (result_df[long_col] <= bounds['lon_max'])
            )
            result_df.loc[mask, 'Region'] = region
        
        # Reorder columns to have Region first

        return result_df




def transform_data(coordinator, df):
    """Transform coordinates and aggregate daily data"""
    df = df.copy()
    df = coordinator.fit(df, lat_col='Latitude', lon_col='Longitude')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    daily_first = df.sort_values('Timestamp').groupby(['Uid', 'Date'], as_index=False).first()
    return daily_first.drop_duplicates(subset=['Uid', 'cell_x', 'cell_y'], keep='first')

from sklearn.model_selection import train_test_split
import numpy as np

def prepare_ml_data(transformed_df, test_size=0.2, random_state=42, sigma=5, return_test=False, return_discretizer=False, discretizer=None):
    """Prepare data for machine learning with optional testing data return."""
    # Discretize coordinates
    if discretizer is not None:
        dis = discretizer
        uid_val = dis.transform(
            df=transformed_df,
            uid_col='Uid',
            date_col='Date',
            lon_col='cell_x',
            lat_col='cell_y'
        )
    else:
        dis = CoordDiscretizer(output='matrix')
        uid_val = dis.fit_transform(
            df=transformed_df,
            uid_col='Uid',
            date_col='Date',
            lon_col='cell_x',
            lat_col='cell_y'
        )

    # Extract all user‐matrices
    uid_vals = list(uid_val.values())

    if return_test:
        # Only split if we're going to return test data
        train_d, test_d = train_test_split(
            uid_vals,
            test_size=test_size,
            random_state=random_state
        )
    else:
        # No test split: everything is training
        train_d = uid_vals
        test_d = None

    # Helper to split into (x, y, original) tuples
    train_x, train_y, train_o = split_data_by_value(train_d, keep_original=True)
    if sigma != 0:
        g_train_x = apply_gaussian(train_x, sigma=sigma).reshape(len(train_x), -1)
        g_train_y = apply_gaussian(train_y, sigma=sigma).reshape(len(train_y), -1)
    else:
        g_train_x = np.array(train_x).reshape(len(train_x), -1)
        g_train_y = np.array(train_y).reshape(len(train_y), -1)

    if return_test:
        test_x, test_y, test_o = split_data_by_value(test_d, keep_original=True)
        if sigma != 0:
            g_test_x = apply_gaussian(test_x, sigma=sigma).reshape(len(test_x), -1)
            g_test_y = apply_gaussian(test_y, sigma=sigma).reshape(len(test_y), -1)
        else:
            g_test_x = np.array(test_x).reshape(len(test_x), -1)
            g_test_y = np.array(test_y).reshape(len(test_y), -1)
        if return_discretizer:
            return g_train_x, g_train_y, g_test_x, g_test_y, train_o, test_o, dis
        else:
            return g_train_x, g_train_y, g_test_x, g_test_y, train_o, test_o

    # Only returning training data
    if return_discretizer:
        return g_train_x, g_train_y, train_o, dis
    else:
        return g_train_x, g_train_y, train_o


# Function to assign regions based on coordinates in a DataFrame
def get_region_coordinates(df, lat_col='Latitude', long_col='Longitude'):
    japan_regions = {
        "Hokkaido": {"lat_min": 41.35, "lat_max": 45.55, "lon_min": 139.75, "lon_max": 145.82},  # Northern island
        "Tohoku": {"lat_min": 36.75, "lat_max": 41.35, "lon_min": 139.75, "lon_max": 141.95},     # NE Honshu
        "Kanto": {"lat_min": 34.95, "lat_max": 36.75, "lon_min": 138.85, "lon_max": 141.05},      # Tokyo area
        "Chubu": {"lat_min": 34.55, "lat_max": 37.75, "lon_min": 136.65, "lon_max": 139.15},      # Central Honshu
        "Kinki (Kansai)": {"lat_min": 33.85, "lat_max": 35.65, "lon_min": 134.75, "lon_max": 136.25},  # Osaka area
        "Chugoku": {"lat_min": 33.85, "lat_max": 35.55, "lon_min": 131.75, "lon_max": 134.45},    # Western Honshu
        "Shikoku": {"lat_min": 32.75, "lat_max": 34.35, "lon_min": 132.55, "lon_max": 134.65},    # Shikoku island
        "Kyushu": {"lat_min": 31.05, "lat_max": 33.95, "lon_min": 129.85, "lon_max": 131.95},     # Southern island
        "Okinawa": {"lat_min": 24.05, "lat_max": 28.05, "lon_min": 123.75, "lon_max": 131.35}     # Far south islands
    }

    # Convert to DataFrame for easier handling
    regions_df = pd.DataFrame.from_dict(japan_regions, orient='index')
    regions_df.index.name = 'Region'
    # Ensure the input DataFrame has the specified lat/long columns
    if lat_col not in df.columns or long_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{lat_col}' and '{long_col}' columns")
    
    # Create a copy of the input DataFrame
    result_df = df.copy()
    
    # Initialize the Region column as "Unlisted"
    result_df['Region'] = 'Unlisted'
    
    # Loop through each region and assign based on ranges
    for region, bounds in regions_df.iterrows():
        mask = (
            (result_df[lat_col] >= bounds['lat_min']) & 
            (result_df[lat_col] <= bounds['lat_max']) & 
            (result_df[long_col] >= bounds['lon_min']) & 
            (result_df[long_col] <= bounds['lon_max'])
        )
        result_df.loc[mask, 'Region'] = region
    
    # Reorder columns to have Region first
    return result_df['Region']
