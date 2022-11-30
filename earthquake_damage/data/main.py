import pandas as pd
import os

from earthquake_damage.data.merge_dataset import drop_and_merge
from sklearn.model_selection import train_test_split

my_name = os.environ.get("MY_NAME")


b_structure = pd.read_csv(f'/Users/{my_name}/code/chantalwuer/earthquake_damage/raw_data/building/csv_building_structure.csv')
b_owner_use = pd.read_csv(f'/Users/{my_name}/code/chantalwuer/earthquake_damage/raw_data/building/csv_building_ownership_and_use.csv')


merged_df = drop_and_merge(b_structure, b_owner_use)


def train_test_val(df, test_size=0.3, random_state=42):
    """
    This function defines X and y,
    then splits the data into a training, test and validation set.
    """
    X = df.drop(['damage_grade', 'building_id'], axis=1)
    y = df['damage_grade']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Split the data into train and test
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5,random_state)

    return X_train, X_test, X_val, y_train, y_test, y_val
