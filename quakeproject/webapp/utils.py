import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta

def prepare_earthquake_data_and_model(days_out_to_predict=7, max_depth=3, eta=0.1):
    # get latest data from USGS servers
    df = pd.read_csv('https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.csv')
    df = df.sort_values('time', ascending=True)
    df['date'] = df['time'].str[0:10]

    df = df[['date', 'latitude', 'longitude', 'depth', 'mag', 'place']]
    temp_df = df['place'].str.split(', ', expand=True)
    df['place'] = temp_df[1]
    df_coords = df.groupby(['place'], as_index=False)[['latitude', 'longitude']].mean()
    df = pd.merge(df[['date', 'depth', 'mag', 'place']], df_coords, on='place')

    eq_data, df_live = [], []

    for symbol in set(df['place']):
        temp_df = df[df['place'] == symbol].copy()
        for win in [22, 15, 7]:
            temp_df[f'depth_avg_{win}'] = temp_df['depth'].rolling(window=win).mean()
            temp_df[f'mag_avg_{win}'] = temp_df['mag'].rolling(window=win).mean()

        temp_df['mag_outcome'] = temp_df['mag_avg_7'].shift(-days_out_to_predict)
        df_live.append(temp_df.tail(days_out_to_predict))
        eq_data.append(temp_df)

    df = pd.concat(eq_data)
    df = df.dropna(subset=['depth_avg_22', 'mag_avg_22', 'mag_outcome'])
    df['mag_outcome'] = (df['mag_outcome'] > 2.5).astype(int)

    df_live = pd.concat(df_live)
    df_live = df_live.dropna(subset=['mag_avg_22'])

    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    combined_places = pd.concat([df['place'], df_live['place']])
    le.fit(combined_places)
    df['place'] = le.transform(df['place'])
    df_live['place'] = le.transform(df_live['place'])

    features = [f for f in df.columns if f not in ['date', 'mag_outcome', 'latitude', 'longitude']]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df[features], df['mag_outcome'], test_size=0.3, random_state=42)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    param = {
        'objective': 'binary:logistic',
        'booster': 'gbtree',
        'eval_metric': 'auc',
        'max_depth': max_depth,
        'eta': eta,
    }

    model = xgb.train(param, dtrain, num_boost_round=1000)

    dlive = xgb.DMatrix(df_live[features])
    preds = model.predict(dlive)

    df_live = df_live[['date', 'place', 'latitude', 'longitude']]
    df_live = df_live.assign(preds=pd.Series(preds).values)
    df_live = df_live.groupby(['date', 'place'], as_index=False).mean()

    df_live['date'] = pd.to_datetime(df_live['date']) + pd.to_timedelta(days_out_to_predict, unit='d')

    return df_live


def get_earth_quake_estimates(desired_date, df_live):
    live_set_tmp = df_live[df_live['date'] == desired_date]
    LatLngString = ''

    if not live_set_tmp.empty:
        for lat, lon, pred in zip(live_set_tmp['latitude'], live_set_tmp['longitude'], live_set_tmp['preds']):
            if pred > 0.3:
                LatLngString += f"new google.maps.LatLng({lat},{lon}),"

    return LatLngString
