from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler

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

    return X_processed
