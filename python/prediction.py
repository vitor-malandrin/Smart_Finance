import pandas as pd
from data_processing import prepare_data
from models import *

def test_all_models(train_x, test_x, train_y, test_y):
    linear_regression_prediction = linear_regression_test(train_x, test_x, train_y, test_y)
    random_forest_prediction = random_forest_test(train_x, test_x, train_y, test_y)
    support_vector_prediction = support_vector_test(train_x, test_x, train_y, test_y)
    gradient_boost_prediction = gradient_boost_test(train_x, test_x, train_y, test_y)
    ridge_prediction = ridge_test(train_x, test_x, train_y, test_y)
    lasso_prediction = lasso_test(train_x, test_x, train_y, test_y)

    return linear_regression_prediction, random_forest_prediction, support_vector_prediction, \
        gradient_boost_prediction, ridge_prediction, lasso_prediction

crypto_symbols = [
    'ADA',
    'AXS',
    'BNB',
    'BTC',
    'BUSD',
    'C98',
    'CHZ',
    'DOGE',
    'DOT',
    'ENJ',
    'ETH',
    'FIS',
    'LINK',
    'LTC',
    'MATIC',
    'SHIB',
    'SOL',
    'USDT',
    'WIN',
    'XRP',
]

def predict_for_all_symbols():
    for symbol in crypto_symbols:
        train_x, test_x, train_y, test_y = prepare_data(symbol)

        print(f'Testing for: {symbol}BRL')
        predictions = test_all_models(train_x, test_x, train_y, test_y)
        print()

        linear_regression_prediction = predictions[0]
        random_forest_prediction = predictions[1]
        support_vector_prediction = predictions[2]
        gradient_boost_prediction = predictions[3]
        ridge_prediction = predictions[4]
        lasso_prediction = predictions[5]

        aux_dataframe = pd.DataFrame()
        aux_dataframe['Open'] = test_x['Open']
        aux_dataframe['High'] = test_x['High']
        aux_dataframe['Low'] = test_x['Low']
        aux_dataframe['test_y (Close)'] = test_y

        aux_dataframe['Linear Regression'] = linear_regression_prediction
        aux_dataframe['Random Forest'] = random_forest_prediction
        aux_dataframe['Support Vector'] = support_vector_prediction
        aux_dataframe['Gradient Boost'] = gradient_boost_prediction
        aux_dataframe['Ridge'] = ridge_prediction
        aux_dataframe['Lasso'] = lasso_prediction

        print(aux_dataframe)
        print('-' * 80)
