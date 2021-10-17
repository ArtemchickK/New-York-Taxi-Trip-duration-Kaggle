import joblib
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def get_numeric_categorical_features(df):
    numeric_features = list(df.select_dtypes(include=['int64', 'float64']).columns)
    categorical_features = list(df.select_dtypes(include=['object']).columns)
    return numeric_features, categorical_features


if __name__ == "__main__":
    data = pd.read_csv('PreprocessedData/data_preprocessed.csv')
    y = pd.read_csv('PreprocessedData/y_preprocessed.csv')

    data.drop(columns=[data.columns[0]], inplace=True)
    y.drop(columns=[y.columns[0]], inplace=True)
    print(data.shape, y.shape)

    numeric, categorical = get_numeric_categorical_features(data)
    x_train, x_test, y_train, y_test = train_test_split(data, y, train_size=0.7, random_state=10)

    ct = ColumnTransformer([('ohe', OneHotEncoder(), categorical), ('ss', StandardScaler(), numeric)],
                           remainder='passthrough')
    ct.fit_transform(x_train)

    model = xgb.XGBRegressor(n_estimators=120)
    pipe = Pipeline([('ct', ct), ('model', model)])
    pipe.fit(x_train, y_train, model__eval_metric='mae')

    joblib.dump(pipe, 'Model/xgb.model')
    x_test.to_csv('TestData/x_test.csv', index=False)
    y_test.to_csv('TestData/y_test.csv', index=False)
