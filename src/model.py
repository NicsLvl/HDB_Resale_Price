from src.overhead_functions import timer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
import typing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def linear_preprocess(X):
    """This function will be used to preprocess features for linear model
    """
    numeric_features = X.select_dtypes(
        include=['number']).columns
    categorical_features = X.select_dtypes(
        include=['object', 'category']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ])

    X = preprocessor.fit_transform(X)
    return X


@timer
def fit_model(model, X_train, X_test, y_train, y_test):
    print(f"Training model: {model}")
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f'Test score: {score:.2f}')
    y_pred = model.predict(X_test)
    return y_test, y_pred


@timer
def train_model(df: pd.DataFrame, y_variable: str, chosen_features: typing.List[str], model):
    """This function will be used to train the model
    """
    print(f"Chosen features: {chosen_features}")
    X = df[chosen_features]
    y = df[y_variable]

    log_features = ['resale_price']
    # Split the data first to prevent the data leakage and then preprocess
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    X_train_tf = linear_preprocess(X_train)
    X_test_tf = linear_preprocess(X_test)
    y_test, y_pred = fit_model(model, X_train_tf, X_test_tf,
                               y_train, y_test)
    return y_test, y_pred


@timer
def evaluate_model(y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"R2: {r2:.2f}")
    print(f"MSE: {mse:,.2f}")
    print(f"RMSE: {rmse:,.2f}")

    residuals = y_test - y_pred

    # Residual Plots
    plt.figure()
    sns.scatterplot(x=y_pred, y=residuals)
    plt.xlabel("Predicted House Prices")
    plt.ylabel("Residuals")
    plt.show()

    z_scores = (residuals - residuals.mean()) / residuals.std()
    outlier_threshold = 3
    outlier_indices = np.abs(stats.zscore(residuals)) > outlier_threshold
    plt.scatter(y_pred[outlier_indices], residuals[outlier_indices],
                color='red', s=50, label='Potential Outliers')
    plt.legend()
    plt.show()

    plt.figure()
    sns.histplot(residuals, kde=True)
    plt.title("Distribution of Residuals")
    plt.show()

    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("Q-Q Plot of Residuals")
    plt.show()
