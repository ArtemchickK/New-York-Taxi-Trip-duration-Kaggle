import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import joblib
import os

if __name__ == "__main__":
    current_directory = os.getcwd()
    current_path = current_directory.split(sep="\\")
    project_path = "/".join(current_path[:-1])
    data_path = project_path + "/src/PreprocessedData/data_preprocessed.csv"
    y_path = project_path + "/src/PreprocessedData/y_preprocessed.csv"

    path = project_path + data_path
    data = pd.read_csv(data_path)
    y = pd.read_csv(y_path)

    data.drop(columns=[data.columns[0]], inplace=True)
    y.drop(columns=[y.columns[0]], inplace=True)

    numeric_features = list(data.select_dtypes(include=['int64', 'float64']).columns)
    categorical_features = list(data.select_dtypes(include=['object']).columns)

    x_train, x_test, y_train, y_test = train_test_split(data, y, train_size=0.7, random_state=10)
    column_transformer = make_column_transformer((OneHotEncoder(), categorical_features),
                                                 (StandardScaler(), numeric_features))
    column_transformer.fit_transform(x_train)
    column_transformer.fit_transform(x_test)

    const_mae = mean_absolute_error(y, np.median(y) * np.ones(y.shape))
    const_mse = mean_squared_error(y, np.median(y) * np.ones(y.shape))
    const_rmse = mean_squared_error(y, np.median(y) * np.ones(y.shape), squared=False)

    print("Estimation on preprocessed data:")
    print("mae = %.4f" % const_mae)
    print("mse = %.4f" % const_mse)
    print("rmse = %.4f" % const_rmse)

    for model in [LinearRegression(), Lasso(), Ridge()]:
        pipe = make_pipeline(column_transformer, model)
        pipe.fit(x_train, y_train)
        y_pred = pipe.predict(x_test)
        print(model)
        print('mse = %.4f' % mean_squared_error(y_test, y_pred))
        print('rmse = %.4f' % mean_squared_error(y_test, y_pred, squared=False))
        print('mae = %.4f' % mean_absolute_error(y_test, y_pred))

    ct = ColumnTransformer([('ohe', OneHotEncoder(), categorical_features), ('ss', StandardScaler(), numeric_features)],
                           remainder='passthrough')
    ct.fit(x_train)

    alphas = np.logspace(-2, 4, 20)
    params = {
        'model__alpha': alphas
    }

    pipe = Pipeline([('ct', ct), ('model', Ridge())])
    gs = GridSearchCV(pipe, params, cv=5, scoring='neg_mean_squared_error')
    gs.fit(x_train, y_train)

    best_alpha = gs.best_params_['model__alpha']
    best_score = gs.best_score_
    print("Best alpha %.4f" % best_alpha)
    print("Best score %.4f" % -best_score)

    y_pred = gs.predict(x_test)
    print("mse after hyperparameter tuning ", mean_squared_error(y_test, y_pred))
    print("mae after hyperparameter tuning ", mean_absolute_error(y_test, y_pred))
    print("rmse after hyperparameter tuning ", mean_squared_error(y_test, y_pred, squared=False))

    joblib.dump(pipe, "Models/linear_models.pkl")
