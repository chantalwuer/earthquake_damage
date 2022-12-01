import pandas as pd
import os

from sklearn.model_selection import train_test_split

my_name = os.environ.get("MY_NAME")


def train_test_val(test_size=0.3, random_state=42):
    """
    This function defines X and y,
    then splits the data into a training, test and validation set.
    """
    X_processed = pd.read_csv(f'/Users/{my_name}/code/chantalwuer/earthquake_damage/processed_data/X_processed.csv')
    y_processed = pd.read_csv(f'/Users/{my_name}/code/chantalwuer/earthquake_damage/processed_data/y_processed.csv')

    # X = df.drop(['damage_grade', 'building_id'], axis=1)
    # y = df['damage_grade']

    # Split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=test_size, random_state=random_state)

    # Split the data into test and val
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=random_state)

    return X_train, X_test, X_val, y_train, y_test, y_val
