from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler

import pandas as pd
import numpy as np
import os

from earthquake_damage.ml_logic.params import LOCAL_DATA_PATH

def preprocess_features(X: pd.DataFrame) -> np.ndarray:

    def create_sklearn_preprocessor() -> ColumnTransformer:
        """
        Create a scikit-learn preprocessor
        that transforms a cleaned dataset
        into a preprocessed one of different
        """

        num_transformer = make_pipeline(StandardScaler())
        num_col = make_column_selector(dtype_include=['int64'])

        cat_transformer = make_pipeline(OneHotEncoder(handle_unknown='ignore'))
        cat_col = make_column_selector(dtype_include=['object'])

        preproc_basic = make_column_transformer((num_transformer, num_col),
                                                (cat_transformer, cat_col),
                                                remainder='passthrough')


        return preproc_basic

    print("\nPreprocess features...")

    preprocessor = create_sklearn_preprocessor()

    X_processed = preprocessor.fit_transform(X)

    print("\n✅ X_processed, with shape", X_processed.shape)


    # X_processed.to_csv('/Users/chantalwuerschinger/code/chantalwuer/earthquake_damage/raw_data/challenge_data/submissions/submission_baseline.csv', index=False)

    path = os.path.join(
        os.path.expanduser(LOCAL_DATA_PATH),
        "raw" if "raw" in path else "processed",
        f"{path}.csv")

    print(Fore.BLUE + f"\nSave data to {path}:" + Style.RESET_ALL)

    X_processed.to_csv(path,
                mode="w" if is_first else "a",
                header=is_first,
                index=False)

    return X_processed
