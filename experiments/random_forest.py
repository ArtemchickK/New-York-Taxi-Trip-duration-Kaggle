import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import os

if __name__ == "__main__":
    current_directory = os.getcwd()
    current_path = current_directory.split(sep="\\")
    project_path = "/".join(current_path[:-1])
    data_path = project_path + "/src/PreprocessedData/data_preprocessed.csv"
    y_path = project_path + "/src/PreprocessedData/y_preprocessed.csv"

    data = pd.read_csv(data_path)
    y = pd.read_csv(y_path)

    data.drop(columns=[data.columns[0]], inplace=True)
    y.drop(columns=[y.columns[0]], inplace=True)

    numeric_features = list(data.select_dtypes(include=['int64', 'float64']).columns)
    categorical_features = list(data.select_dtypes(include=['object']).columns)

    x_train, x_test, y_train, y_test = train_test_split(data, y, train_size=0.7, random_state=10)

    ct = ColumnTransformer([('ohe', OneHotEncoder(), categorical_features), ('ss', StandardScaler(), numeric_features)])
    pipe = Pipeline([('ct', ct), ('model', RandomForestRegressor(n_estimators=10))])
    pipe.fit(x_train, y_train)

    y_pred = pipe.predict(x_test)

    time_test = np.exp(y_test) - 1
    time_pred = np.exp(y_pred) - 1

    mae_in_minutes = mean_absolute_error(time_test, time_pred) / 60
    rmse_in_minutes = mean_squared_error(time_test, time_pred, squared=False) / 60

    print("mae in minutes %.1f" % mae_in_minutes)
    print("rmse in minutes %.1f" % rmse_in_minutes)

    joblib.dump(pipe, "Models/random_forest.pkl")


