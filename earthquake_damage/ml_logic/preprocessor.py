import os
import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn import set_config; set_config(display='diagram')

from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer

my_name = os.environ.get('MY_NAME')


def cus_imputation(df: pd.DataFrame=True, filename=False) -> pd.DataFrame:
    '''
    This function is used to impute the missing values in the whole dataset before training the model.
    It is using the Simple Imputer to fill the missing values with the mean of the column.
    '''
    if filename:
        path = f'/Users/{my_name}/code/chantalwuer/earthquake_damage/processed_data/{filename}.csv'
        df = pd.read_csv(path)
        missing = df.isna().sum().sum()

    print("\nImputation...")

    my_imputer = SimpleImputer(strategy='most_frequent')
    df_imputed = pd.DataFrame(my_imputer.fit_transform(df), columns = df.columns).astype(df.dtypes.to_dict())
    # X_imputed.columns = X.columns

    print("\n✅  There are", missing, "vaules missing in the dataset.")

    print("\n✅ df_imputed, with shape", df_imputed.shape)

    # Save data
    path = f'/Users/{my_name}/code/chantalwuer/earthquake_damage/processed_data/df_imputed.csv'
    df_imputed.to_csv(path, index=False)

    print(f"✅ df_imputed saved to {path}")

    return None


def preprocess_features(X_train=None, X_test=None, X_val=None) -> pd.DataFrame:

    ''' This function is used to preprocess the data before training the model.
        It is scaling the numerical features with Robust Scaler
        and One Hot Encoding the categorical features.'''

    path = f'/Users/{my_name}/code/chantalwuer/earthquake_damage/processed_data/df_imputed.csv'
    df_imputed = pd.read_csv(path)

    X = df_imputed.drop(['building_id', 'damage_grade'], axis=1)

    # Make Preprocessing Pipeline
    num_transformer = make_pipeline(RobustScaler())
    num_col = make_column_selector(dtype_include=['int64', 'float64'])

    cat_transformer = make_pipeline(OneHotEncoder(handle_unknown='ignore'))
    cat_col = make_column_selector(dtype_include=['object'])

    preprocessor = make_column_transformer((num_transformer, num_col),
                                            (cat_transformer, cat_col),
                                            remainder='passthrough')

    print("\nPreprocess features...")

    X_processed = preprocessor.fit_transform(X)

    print("\n✅ X_processed, with shape", X_processed.shape)

    # Save data
    X_processed = pd.DataFrame(X_processed)
    X_processed.columns = preprocessor.get_feature_names_out()

    path = f'/Users/{my_name}/code/chantalwuer/earthquake_damage/processed_data/X_processed.csv'
    X_processed.to_csv(path, index=False)

    print(f"✅ X_processed saved to {path}")

    return None

def preprocess_targets(y_train=None, y_test=None, y_val=None) -> np.ndarray:
    '''
    This fucntion is used to preprocess the target data before traning the model.
    If the labels are strings, it is converting them to integers.
    Otherwise, it is returning the labels as they are.
    '''
    path = f'/Users/{my_name}/code/chantalwuer/earthquake_damage/processed_data/df_imputed.csv'
    df_imputed = pd.read_csv(path)

    y = df_imputed['damage_grade']

    print("\nPreprocess target...")

    if y.dtypes == 'object':
        le = LabelEncoder()
        y_processed = le.fit_transform(y)
    else:
        y_processed = y


    print("\n✅ y processed, with shape", y_processed.shape)

    #Save data
    y_processed = pd.DataFrame(y_processed, columns = ['damage_grade'])
    path = f'/Users/{my_name}/code/chantalwuer/earthquake_damage/processed_data/y_processed.csv'
    y_processed.to_csv(path, index=False)
    print(f"✅ y_processed saved to {path}")

    return None


# def save_data(X : pd.DataFrame, y: pd.DataFrame):
#     X = pd.DataFrame(X)
#     y = pd.DataFrame(y)
#     my_name = os.environ.get('MY_NAME')
#     path_X = f'/Users/{my_name}/.lewagon/project_weeks/data/processed/ X_processed.csv'
#     path_y = f'/Users/{my_name}/.lewagon/project_weeks/data/processed/ y_processed.csv'
#     X.to_csv(path_X, index=False)
#     y.to_csv(path_y, index=False)


# def test_preprocess_features(df):
#     '''
#     This is a test of the feature preprocessing function above
#     As a backup
#     '''
#     num_columns = [name for col, name in zip(df, df.columns) if df[col].dtypes =='int64' or df[col].dtypes == 'float64']
#     print('There are', len(num_columns),'columns with numeric values')

#     text_columns = [name for col, name in zip(df, df.columns) if df[col].dtypes =='object']
#     print('There are', len(text_columns) ,'columns with string values')

#     cat_transformer = OneHotEncoder(handle_unknown = 'ignore')
#     rb_scaler = RobustScaler()

#     preprocessor = ColumnTransformer([
#         ('rb_scaler', rb_scaler, num_columns),
#         ('cat_transformer', cat_transformer, text_columns)], remainder = 'passthrough')

#     df_pre = preprocessor.fit_transform(df)
#     df_pre = pd.DataFrame(df_pre, columns = preprocessor.get_feature_names_out())
#     #df_pre.columns = preprocessor.get_feature_names_out()

#     print("\n✅ df_processed, with shape", df_pre.shape)

#     return df_pre
