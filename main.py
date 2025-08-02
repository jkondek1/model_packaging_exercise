from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

from scripts.data_loading import load_data
from scripts.model_training import modify_pre_training, encode, create_train_test_split, fit_model
from scripts.preprocessing import run_preprocessing

if __name__ == '__main__':
    # mali by sme pridat argument parser a zadat path z command line
    # pre jednoduchost fixneme
    data = load_data('/Users/A107809368/projects/playground/my_ml_project/data/airbnb.csv.zip')
    data = run_preprocessing(data)
    encoder = OneHotEncoder(sparse_output=False)
    model_lr = LinearRegression()
    data = modify_pre_training(data)
    data = encode(data, encoder)
    x_train, x_test, y_train, y_test = create_train_test_split(data)
    fit_model(x_train, y_train, model_lr)
