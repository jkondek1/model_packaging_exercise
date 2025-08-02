import pandas as pd
import numpy as np

COLS_TO_DROP = [
    'id',
    'name',
    'host_name',
    'host_id',
    'features',
    'amenities',
    'safety_rules',
    'hourse_rules',
    'img_links',
    'country',
]

def create_host_counts(data: pd.DataFrame) -> pd.DataFrame:
    df_host_counts = data.groupby('host_id')['name'].nunique().rename('host_count')
    data['host_count'] = df_host_counts.loc[data['host_id']].values
    return data

def drop_cols(data: pd.DataFrame, cols: list = COLS_TO_DROP) -> pd.DataFrame:
    data = data.drop(columns=cols)
    return data

def modify_other_stuff(data: pd.DataFrame) -> pd.DataFrame:
    data['address_0'] = data['address'].str.split(',').str[0]
    df_rbb_address_0_counter = data['address_0'].value_counts().loc[data['address_0']]
    data['address_0'] = np.where(df_rbb_address_0_counter > 50, data['address_0'], 'other')
    data = data.drop(columns=['address', 'checkout', 'checkin'])
    df_rbb_rating = np.where(
        data['rating'] == 'New',
        np.NaN,
        data['rating']
    )
    df_rbb_rating_mean = pd.Series(df_rbb_rating.astype(float)).dropna().mean()
    df_rbb_rating = np.where(
        pd.Series(df_rbb_rating).isna(),
        df_rbb_rating_mean,
        data['rating']
    ).astype(float)
    data['rating'] = df_rbb_rating
    return data
