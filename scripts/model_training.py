import pandas as pd
import pickle
import sklearn
from sklearn.model_selection import train_test_split

def modify_pre_training(data: pd.DataFrame) -> pd.DataFrame:
    data['reviews'] = data['reviews'].str.replace(',', '').astype(int)
    return data

def encode(data: pd.DataFrame, encoder: sklearn.preprocessing.OneHotEncoder) -> pd.DataFrame:
    data_address_0_ohe = encoder.fit_transform(data[['address_0']])
    data_address_0_ohe = pd.DataFrame(
        data_address_0_ohe,
        columns=encoder.get_feature_names_out()
    )
    data = pd.concat([data, data_address_0_ohe], axis=1)
    return data



def create_train_test_split(data: pd.DataFrame) -> tuple:
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(columns=['price']),
        data['price'],
        test_size=0.2,
        random_state=42
    )
    return X_train, X_test, y_train, y_test

def fit_model(train, label, model):
    model.fit(train, label)
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    return model

#model_lr = LinearRegression()
#model_lr.fit(X_train, y_train)
