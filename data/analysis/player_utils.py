import re
import ast
import win32api

import pandas as pd

def calculate(df: pd.DataFrame, calculation: str, target_column='result', window=4, percentage=True):
    if percentage:
        percent = 100
    else:
        percent = 1

    df = df.copy()

    if 'uid' not in df.columns:
        df.loc[:, 'uid'] = range(1, len(df) + 1)

    if calculation == 'rolling_average':
        df['target_calculation'] = df[target_column].rolling(window=window).mean() * percent

    elif calculation == 'rolling_average_no_overlap':
        # df = df.dropna(subset=[target_column])
        df['group'] = (df['uid'] - 1)  // window
        df['target_calculation'] = df[target_column].groupby(df['group']).transform('mean') * percent
        df = df.drop_duplicates(subset=['group'], keep='last')

    elif calculation == 'rolling_sum_no_overlap':
        # df = df.dropna(subset=[target_column])
        df['group'] = (df['uid'] - 1)  // window
        df['target_calculation'] = df[target_column].groupby(df['group']).transform('sum') * percent
        df = df.drop_duplicates(subset=['group'], keep='last')

    elif calculation == 'zscore':
        data = df[target_column].values
        data_mean = data.mean()
        data_std = data.std()
        df['target_calculation'] = (data - data_mean)/data_std

    elif calculation == 'zscore_rolling_average_no_overlap':
        data = df[target_column].values
        data_mean = data.mean()
        data_std = data.std()
        zscore = (data - data_mean)/data_std

        df['group'] = (df['uid'] - 1)  // window
        df['target_calculation'] = zscore.groupby(df['group']).transform('mean') * percent
        df = df.drop_duplicates(subset=['group'], keep='last')

    # save to file for debugging
    # df.to_csv('calculation.csv')

    return df

def quote_keys(dict_string: str):
    quoted_string = re.sub(r'(\w+):', r'"\1":', dict_string)
    return quoted_string

def as_dict(string):
    return ast.literal_eval(quote_keys(string))

def get_monitor_refresh_rate():
    device = win32api.EnumDisplayDevices()
    settings = win32api.EnumDisplaySettings(device.DeviceName, -1)
    return int(getattr(settings,'DisplayFrequency'))