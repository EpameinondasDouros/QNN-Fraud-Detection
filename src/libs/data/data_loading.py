from ..imports import pd, np, tf, plt, sns, qml, Sequential, Dense, Dropout
from . import data_process_utils as dpu
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from loguru import logger
import joblib
import os
import pickle


def load_preprocess_data(finance_df):
    logger.info("Data Processing and Cleaning")


    parent_dir = os.path.dirname(os.path.abspath(__file__))
    print(parent_dir)
    scaler = joblib.load(os.path.join(parent_dir, "scaler_exp_8.joblib")) 

    df_cleaned = finance_df.loc[finance_df['type'].isin(['CASH_OUT', 'TRANSFER'])].copy()
    df_cleaned.drop(columns=['nameOrig','nameDest','isFlaggedFraud'],inplace=True)
    df_cleaned['type'] = finance_df['type'].map({'CASH_OUT': 0, 'TRANSFER': 1})
    
    final_df=df_cleaned

    y_test = final_df['isFraud'].astype(int)
    X_test = final_df.drop(columns=['isFraud'])

    print(X_test.head())
    X_test = scaler.transform(X_test)

    logger.info("Data Loaded and Scaler Saved")

    logger.info("Data Loaded")

    return X_test, y_test