import os
import numpy as np

##################  VARIABLES  ##################
DATA_SIZE = "1k" # ["1k", "200k", "all"]
CHUNK_SIZE = 200
GCP_PROJECT = "<your project id>" # TO COMPLETE
GCP_PROJECT_WAGON = "wagon-public-datasets"
BQ_DATASET = "taxifare"
BQ_REGION = "EU"
MODEL_TARGET = "local"
##################  CONSTANTS  #####################
#LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "data")
#LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "training_outputs")
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), "data")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), "training_outputs")

COLUMN_NAMES = ['checking_account', 'duration', 'credit_history', 'purpose', 'credit_amount', 'savings_account', 'employment_duration', 'installment_rate', 'personal_status', 'other_debtors', 'residence_duration', 'property', 'age', 'other_installment_plans', 'housing', 'existing_credits', 'job', 'dependents', 'telephone', 'foreign_worker', 'credit_risk']



DTYPES_RAW = {
    "fare_amount": "float32",
    "pickup_datetime": "datetime64[ns, UTC]",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16"
}


{'checking_account': ,
, 'duration', 'credit_history', 'purpose', 'credit_amount', 'savings_account', 'employment_duration', 'installment_rate', 'personal_status', 'other_debtors', 'residence_duration', 'property', 'age', 'other_installment_plans', 'housing', 'existing_credits', 'job', 'dependents', 'telephone', 'foreign_worker', 'credit_risk']


DTYPES_PROCESSED = np.float32
