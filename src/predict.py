import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

if __name__ == "__main__":

    model = joblib.load('Model/xgb.model')
    x_test = pd.read_csv('TestData/x_test.csv')
    y_test = pd.read_csv('TestData/y_test.csv')

    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)

    print("mae = %.4f" % mae)
    print("mse = %.4f" % mse)
    print("rmse = %.4f" % rmse)

    time_test = np.exp(y_test) - 1
    time_pred = np.exp(y_pred) - 1

    mae_in_seconds = mean_absolute_error(time_test, time_pred)
    rmse_in_seconds = mean_squared_error(time_test, time_pred, squared=False)

    mae_in_minutes = mae_in_seconds / 60
    rmse_in_minutes = rmse_in_seconds / 60

    print("mae in minutes = %.1f" % mae_in_minutes)
    print("rmse in minutes = %.1f" % rmse_in_minutes)

    file = open("metrics.txt", "w")
    file.write("mae in minutes = " + str(mae_in_minutes) + "\n")
    file.write("rmse in minutes = " + str(rmse_in_minutes) + "\n")
    file.close()

