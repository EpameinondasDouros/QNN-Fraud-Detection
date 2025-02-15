# Import from imports.py
from .libs.imports import *
from io import StringIO
# Import scripts from models/
from .libs.models import evaluation_functions, quantum_layers
# Import scripts from data/
from .libs.data import data_loading, data_process_utils,finance_data
from loguru import logger
import os

def load_data(finance_df):
    print("Loading data")

    # Check if dataset is non-empty
    if finance_df is not None and not finance_df.empty:
        logger.info("Using provided dataset for processing")
        X_test, y_test= data_loading.load_preprocess_data(finance_df)
        testing=False
        logger.info("Data loaded successfully!")
    else:
        logger.info("Dataset is empty. Loading pre-existing files instead.")     
        
        print("Loading data from .npy and .csv files")
    
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        X_test = np.load(os.path.join(parent_dir, "src/libs/data/X.npy"))
        y_test = pd.read_csv(os.path.join(parent_dir, "src/libs/data/y.csv"))

        logger.info("Data loaded successfully!")
        testing=True

    return X_test, y_test,testing


def create_model(qlayer_long,X_train):

    print("Creating model")
    
    quantum_model_long = Sequential([
        Dense(8, activation=tf.nn.relu, input_shape=(X_train.shape[1],)),
        Dense(4, activation=tf.nn.relu),
        qlayer_long,
        Dense(2, activation=tf.nn.softmax)
    ])
    # parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    current_dir = os.path.dirname(os.path.abspath(__file__))

    print(current_dir)

    # Combine the current directory with the file path name
    h5_file_path = os.path.join(current_dir, "libs/models/quantum_model_weights.h5")  # Replace with the actual file name
    quantum_model_long.load_weights(h5_file_path)

    print("✅ Model weights loaded successfully!")

    return quantum_model_long
            


def run_pipeline(finance_df, run_type,device_name):
    
    print('Running main function')

    X_test, y_test, testing = load_data(finance_df)

    qlayer_long=quantum_layers.create_qlayer_long(n_qubits=3,runtype=run_type,device_name="ibm_brisbane")

    quantum_model_long=create_model(qlayer_long,X_test)

    print('Evaluating Model')

    if testing:
        results = evaluation_functions.evaluate_model(quantum_model_long, X_test, y_test)
    else:
        results = evaluation_functions.predict_fraud(quantum_model_long, X_test, y_test)
        for i, result in enumerate(results):
            print(f"Prediction {i}: {result}")

    return results,testing