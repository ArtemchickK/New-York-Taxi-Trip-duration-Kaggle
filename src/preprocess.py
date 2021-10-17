import pandas as pd
import numpy as np
import calendar


def first_preprocessing(df):
    df = df.copy()
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    y = df['trip_duration'].apply(lambda x: np.log1p(x)).copy()
    df.drop(columns=['trip_duration', 'dropoff_datetime', 'id'], inplace=True)
    return df, y


def distance(df):
    df = df.copy()
    lat1 = df['pickup_latitude'].values
    lng1 = df['pickup_longitude'].values
    lat2 = df['dropoff_latitude'].values
    lng2 = df['dropoff_longitude'].values
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    df['distance'] = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    df['distance'] = df['distance'].apply(lambda x: np.log1p(x))
    df.drop(columns=['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude'], inplace=True)
    return df


def get_day_of_week(data):
    df = data.copy()
    tmp = df['pickup_datetime'].dt.date
    df['pickup_day_of_the_week'] = tmp.apply(lambda x: calendar.day_name[x.weekday()])
    return df


def get_day_of_month(data):
    df = data.copy()
    df['pickup_day'] = df['pickup_datetime'].apply(lambda x: x.day)
    return df


def get_month(data):
    df = data.copy()
    df['pickup_month'] = data['pickup_datetime'].apply(lambda x: x.month)
    return df


def get_hour_of_day(data):
    df = data.copy()
    df['pickup_hour'] = df['pickup_datetime'].apply(lambda x: x.hour)
    return df


def haversine_distance(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h


def preprocessing():

    # Удаляем слишком длительные поездки
    data = pd.read_csv('Data/train.csv')
    data = data.drop(data[data['trip_duration'] > 20000].index)

    # Удаляем записи дни, когда был режим ЧС
    data['date'] = pd.to_datetime(data['pickup_datetime']).dt.date
    data = data.drop(data[data['date'] == pd.to_datetime('2016-01-23')].index)
    data = data.drop(data[data['date'] == pd.to_datetime('2016-01-24')].index)
    data = data.drop(columns=['date'])

    # Удаляем поездки с невозможным соотношением расстояния и длительности поездки
    data['distance_haversine'] = haversine_distance(data['pickup_latitude'].values, data['pickup_longitude'].values,
                                                    data['dropoff_latitude'].values, data['dropoff_longitude'].values)
    data['log_distance_haversine'] = data['distance_haversine'].apply(lambda x: np.log1p(x))
    data = data.drop(columns=['distance_haversine'])
    data['log_trip_duration'] = data['trip_duration'].apply(lambda x: np.log1p(x))
    data = data.drop(data[(data['log_trip_duration'] < 7.5) & (data['log_distance_haversine'] > 5)].index)
    data = data.drop(columns=['log_trip_duration', 'log_distance_haversine'])

    # Выделяем новые признаки
    data, y = first_preprocessing(data)
    data = distance(data)
    data = get_day_of_week(data)
    data = get_day_of_month(data)
    data = get_hour_of_day(data)
    data = get_month(data)
    data = data.drop(columns=['pickup_datetime'])

    return data, y


if __name__ == "__main__":
    data, y = preprocessing()
    data.to_csv('PreprocessedData/data_preprocessed.csv')
    y.to_csv('PreprocessedData/y_preprocessed.csv')
