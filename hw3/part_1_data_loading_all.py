import os
import dask.dataframe as dd
from dask import delayed
import pandas as pd
import numpy as np
from dask_jobqueue import SLURMCluster
from dask.distributed import Client

import logging
logging.basicConfig(level=logging.DEBUG)

print("LOADED  MODULES")

# Mapping dictionaries and dtypes remain the same.
PAYMENT_TYPES_STR_TO_INT = {
    'flex': 0,
    'flex fare trip': 0,
    'fare': 0,
    'credit': 1,
    'credit card': 1,
    'cash': 2,
    'no charge': 3,
    'dispute': 4,
    'unknown': 5,
    'voided trip': 6,
    'voided': 6,
}

DTYPES_ALL = {
    'airport_fee': 'float64',
    'congestion_surcharge': 'float64',
    'improvement_surcharge': 'float64',
    'extra': 'float64',
    'dollocationid': 'int64',
    'pulocationid': 'int64',
    'total_amount': 'float64',
    'tolls_amount': 'float64',
    'tip_amount': 'float64',
    'mta_tax': 'float64',
    'surcharge': 'float64',
    'fare_amount': 'float64',
    'tools_amount': 'float64',
    'payment_type': 'int64',
    'payment_type_string': 'category',
    'dropoff_latitude': 'float64',
    'dolocationid': 'int64',
    'pulocationid': 'int64',
    'dropoff_longitude': 'float64',
    'store_and_fwd_flag': 'category',
    'pickup_latitude': 'float64',
    'pickup_longitude': 'float64',
    'trip_distance': 'float64',
    'passenger_count': 'int64',
    'tpep_dropoff_datetime': 'datetime64[ns]',
    'tpep_pickup_datetime': 'datetime64[ns]',
    'vendorid': 'int64',
    'vendor_name': 'category',
    'ratecodeid': 'int64',
}

COLUMN_MAPPING = {
    "VendorID": "vendorid",
    "vendor_id": "vendorid",
    "vendorid": "vendorid",
    "vendor_name": "vendor_name",
    "pickup_datetime": "tpep_pickup_datetime",
    "tpep_pickup_datetime": "tpep_pickup_datetime",
    "trip_pickup_datetime": "tpep_pickup_datetime",
    "dropoff_datetime": "tpep_dropoff_datetime",
    "tpep_dropoff_datetime": "tpep_dropoff_datetime",
    "trip_dropoff_datetime": "tpep_dropoff_datetime",
    "start_lat": "pickup_latitude",
    "pickup_latitude": "pickup_latitude",
    "start_lon": "pickup_longitude",
    "pickup_longitude": "pickup_longitude",
    "end_lat": "dropoff_latitude",
    "dropoff_latitude": "dropoff_latitude",
    "end_lon": "dropoff_longitude",
    "dropoff_longitude": "dropoff_longitude",
    'fare_amt': "fare_amount",
    "fare_amount": "fare_amount",
    "ratecodeid": "ratecodeid",
    "rate_code": "ratecodeid",
    "store_and_forward": "store_and_fwd_flag",
    "store_and_fwd_flag": "store_and_fwd_flag",
    "surcharge": "surcharge",
    "tip_amt": "tip_amount",
    "tip_amount": "tip_amount",
    "tolls_amt": "tolls_amount",
    "tools_amount": "tools_amount",
    "total_amt": "total_amount",
    "total_amount": "total_amount",
    "passengerCount": "passenger_count",
    "Trip_Distance": "trip_distance",
    "tripDistance": "trip_distance",
    "payment_type": "payment_type",
    "paymentType": "payment_type",
    "Fare_amount": "fare_amount",
    "fareAmount": "fare_amount",
    "Extra": "extra",
    "Tip_amount": "tip_amount",
    "tipAmount": "tip_amount",
    "tollsAmount": "tolls_amount",
    "Total_amount": "total_amount",
    "Improvement_surcharge": "improvement_surcharge",
    "Congestion_surcharge": "congestion_surcharge",
    "store_and_forward_flag": "store_and_fwd_flag",
    'payment_type_string': "payment_type_string",
    'pulocationid': "pulocationid",
    'dolocationid': "dolocationid",


}

ALL_COLUMNS = set(COLUMN_MAPPING.values())

def standardize_columns(df, year):
    # Adjust flags and payment types for certain years.
    print("-" * 100)
    print("COLUMNS: ", df.columns)
    print("-" * 100)
    
    # cast columns to lowercase
    df.columns = df.columns.str.lower()
    # Rename columns based on mapping.
    df = df.rename(columns={col: COLUMN_MAPPING.get(col, col) for col in df.columns})
    if year in [2009, 2010]:
        df['store_and_fwd_flag'] = df['store_and_fwd_flag'].replace({1: 'YES', 0: 'NO'})
        
    
    if year in [2009, 2010]:
        df['payment_type_string'] = df['payment_type'].str.lower().astype('category')
        df['payment_type'] = df['payment_type'].replace(PAYMENT_TYPES_STR_TO_INT)
    df['payment_type'] = df['payment_type'].astype('int64')
    
    
    for col in ALL_COLUMNS:
        if col not in df.columns:
            dtype = DTYPES_ALL.get(col, 'float64')
            df[col] = pd.Series([np.nan] * len(df), dtype=dtype)
        

    
    for col in df.columns:
        if col in DTYPES_ALL:
            df[col] = df[col].astype(DTYPES_ALL[col])
        else:
            print(f"Warning: Column '{col}' not found in DTYPES_ALL. Skipping dtype conversion.")

        
    print("ALL COLUMNS: ", df.columns)
    # Check for missing columns and add them as null columns.
    #  
    
    if 'year' not in df.columns and 'tpep_pickup_datetime' in df.columns:
        df['year'] = df['tpep_pickup_datetime'].dt.year
        
    return df

def combine_multiple_years(start_year, end_year, base_path='.'):
    ddf_list = []
    for year in range(start_year, end_year + 1):
        file_pattern = os.path.join(base_path, f"yellow_tripdata_{year}*.parquet")
        ddf = dd.read_parquet(file_pattern, engine='pyarrow')
        
        # Using meta to optimize scheduling.
        meta = ddf._meta.copy()
        meta = standardize_columns(meta, year)
        
        ddf = ddf.map_partitions(standardize_columns, year=year, meta=meta)

        ddf = ddf[list(sorted(ddf.columns))]
        ddf_list.append(ddf)
    
    combined_ddf = dd.concat(ddf_list, interleave_partitions=True)
    return combined_ddf

if __name__ == "__main__":

    # --- Create the SLURM Cluster ---
    cluster = SLURMCluster(
        queue='all',
        processes=1,
        cores=8,            # 8 cores per job
        memory='256GB',      # ~4GB per core
        walltime='00:20:00',
        scheduler_options={'dashboard_address': ':8087'},
        death_timeout=600,  # seconds
        job_name='dask_job',
        job_extra=['--output=slurm-%j.out',
                #    '--reservation=fri'
                   ],  # Save job logs
        env_extra=['export LANG="en_US.utf8"', 
                   'export LC_ALL="en_US.utf8"']
    )

    cluster.adapt(minimum=1, maximum=10)      
    
    client = Client(cluster)
    
    
    print("Dashboard link:", cluster.dashboard_link)
    
    # Set input and output paths.
    INPUT_DATA_PATH = '/d/hpc/projects/FRI/bigdata/data/Taxi'
    OUTPUT_DATA_PATH = '/d/hpc/projects/FRI/bigdata/students/in7357'
    START_YEAR = 2011
    END_YEAR = 2012 
    
   
    file_pattern = os.path.join(INPUT_DATA_PATH, f"yellow_tripdata_{2009}*.parquet")
    ddf = dd.read_parquet(file_pattern, engine='pyarrow')
    output_file = os.path.join(OUTPUT_DATA_PATH, f'yellow_tripdata_combined_{START_YEAR}_{END_YEAR}.parquet')
    print("HEAD: ", ddf.head().compute())
    ddf.to_parquet(output_file, engine='pyarrow', write_index=False)
    # combined_ddf = combine_multiple_years(START_YEAR, END_YEAR, base_path=INPUT_DATA_PATH)
    
    
    # combined_ddf = combined_ddf.persist()
    # combined_ddf = combined_ddf.clear_divisions()  # avoids some known bugs
    # combined_ddf = combined_ddf.categorize(columns=["payment_type_string", "store_and_fwd_flag", "vendor_name"])
    # combined_ddf = combined_ddf[['vendorid', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']]
    # print("COLUMNS: ", combined_ddf.columns)
    # _ = combined_ddf.head()  # Forces computation and may show error earlier
    # print("HEAD: ", combined_ddf.head().compute())
    
    # output_file = os.path.join(OUTPUT_DATA_PATH, f'yellow_tripdata_combined_{START_YEAR}_{END_YEAR}.parquet')
    # combined_ddf.to_parquet(output_file, engine='pyarrow', compression='gzip')
    

    client.close()
