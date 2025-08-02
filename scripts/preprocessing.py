
df_host_counts = df_rbb.groupby('host_id')['name'].nunique().rename(
    'host_count'
)

df_rbb['host_count'] = df_host_counts.loc[df_rbb['host_id']].values

cols_to_drop = [
    'id',
    'name',
    # 'rating',
    # 'reviews',
    'host_name',
    'host_id',
    # 'address',
    'features',
    'amenities',
    'safety_rules',
    'hourse_rules',
    'img_links',
    # 'price',
    'country',
    # 'bathrooms',
    # 'beds',
    # 'guests',
    # 'toiles',
    # 'bedrooms',
    # 'studios',
    # 'checkin',
    # 'checkout'
]

df_rbb = df_rbb.drop(columns=cols_to_drop)

df_rbb['address_0'] = df_rbb['address'].str.split(',').str[0]
df_rbb_address_0_counter = df_rbb['address_0'].value_counts().loc[df_rbb['address_0']]
df_rbb['address_0'] = np.where(df_rbb_address_0_counter > 50, df_rbb['address_0'], 'other')
df_rbb = df_rbb.drop(columns=['address', 'checkout', 'checkin'])
df_rbb_rating = np.where(
    df_rbb['rating'] == 'New',
    np.NaN,
    df_rbb['rating']
)
df_rbb_rating_mean = pd.Series(df_rbb_rating.astype(float)).dropna().mean()
df_rbb_rating = np.where(
    pd.Series(df_rbb_rating).isna(),
    df_rbb_rating_mean,
    df_rbb['rating']
).astype(float)
df_rbb['rating'] = df_rbb_rating
