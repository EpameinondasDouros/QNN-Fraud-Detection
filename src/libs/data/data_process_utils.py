from imblearn.over_sampling import SMOTE
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd

def count_fraud_percentage(df, fraud_col='isFraud'):
    """
    Calculate the count and percentage of fraud cases in a DataFrame.

    """

    fraud_count = df[fraud_col].sum()

    total_count = len(df)
    fraud_percentage = (fraud_count / total_count) * 100

    return fraud_count, fraud_percentage

def balance_with_smote(df, target_col='Class', smote_ratio=0.2, random_state=42):
    """Apply SMOTE to increase the minority class size by a specified ratio."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Apply SMOTE with a limited ratio
    smote = SMOTE(sampling_strategy=smote_ratio, random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
    balanced_df[target_col] = y_resampled
    return balanced_df


def scale_features(df, target_col='targets'):
    """
    Scales all features except the target column using MinMaxScaler.
    """
    scaler = MinMaxScaler()
    features = df.drop(columns=[target_col])
    df[features.columns] = scaler.fit_transform(features)
    return df

def split_data(df, target_col='targets', test_size=0.3, random_state=42):
    """
    Splits the dataset into train, validation, and test sets.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Initial train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

def convert_to_numpy(*args):
    """
    Converts pandas DataFrames or Series to NumPy arrays.
    """
    return [arg.to_numpy() if isinstance(arg, (pd.DataFrame, pd.Series)) else arg for arg in args]