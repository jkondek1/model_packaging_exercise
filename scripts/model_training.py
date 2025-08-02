from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df_rbb['reviews'] = df_rbb['reviews'].str.replace(',', '').astype(int)
encoder = OneHotEncoder(sparse_output=False)
df_rbb_address_0_ohe = encoder.fit_transform(df_rbb[['address_0']])
df_rbb_address_0_ohe = pd.DataFrame(
    df_rbb_address_0_ohe,
    columns=encoder.get_feature_names_out()
)
df_rbb = pd.concat([df_rbb, df_rbb_address_0_ohe], axis=1)

df_rbb
#%%
X_train, X_test, y_train, y_test = train_test_split(
    df_rbb.drop(columns=['price']),
    df_rbb['price'],
    test_size=0.2,
    random_state=42
)
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
