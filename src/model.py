from src.overhead_functions import timer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
import typing
import pandas as pd


def pipeline_preprocessing(df: pd.DataFrame, model) -> Pipeline:
    """This function will be used to preprocess features
    """
    numeric_features = df.select_dtypes(include=['number']).columns
    categorical_features = df.select_dtypes(
        include=['object', 'category']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    # Create pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])
    return pipeline


@timer
def evaluate_model(model, X, X_train, X_test, y_train, y_test):
    print(f"Training model: {model}")
    pipeline = pipeline_preprocessing(X, model)
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    print(f'Test score: {score}')
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"R2: {r2:.2f}")
    print(f"MSE: {mse:,.0f}")
    print(f"RMSE: {rmse:,.0f}")


def main(df: pd.DataFrame, y_variable: str, chosen_features: typing.List[str]):
    """This function will be used to train the model
    """
    print(f"Chosen features: {chosen_features}")
    # Split the data into train and test

    X = df[chosen_features]
    y = df[y_variable]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    models = [LinearRegression(),
              Ridge(),
              Lasso(),
              GradientBoostingRegressor(verbose=1)]

    for model in models:
        evaluate_model(model, X, X_train, X_test, y_train, y_test)
