import pandas as pd
from data_processing import prepare_data
from models_tests import *
from tuned_models_tests import *
from save_models import *

def test_all_models(train_x, test_x, train_y, test_y):
    linear_regression_prediction = linear_regression_test(train_x, test_x, train_y, test_y)
    random_forest_prediction = random_forest_test(train_x, test_x, train_y, test_y)
    support_vector_prediction = support_vector_test(train_x, test_x, train_y, test_y)
    gradient_boost_prediction = gradient_boost_test(train_x, test_x, train_y, test_y)
    ridge_prediction = ridge_test(train_x, test_x, train_y, test_y)
    lasso_prediction = lasso_test(train_x, test_x, train_y, test_y)

    tuned_linear_regression_prediction = tuned_linear_regression_test(train_x, test_x, train_y, test_y)
    tuned_random_forest_prediction = tuned_random_forest_test(train_x, test_x, train_y, test_y)
    tuned_support_vector_prediction = tuned_support_vector_test(train_x, test_x, train_y, test_y)
    tuned_gradient_boost_prediction = tuned_gradient_boost_test(train_x, test_x, train_y, test_y)
    tuned_ridge_prediction = tuned_ridge_test(train_x, test_x, train_y, test_y)
    tuned_lasso_prediction = tuned_lasso_test(train_x, test_x, train_y, test_y)

    return linear_regression_prediction, random_forest_prediction, support_vector_prediction, \
        gradient_boost_prediction, ridge_prediction, lasso_prediction, tuned_linear_regression_prediction, \
        tuned_random_forest_prediction, tuned_support_vector_prediction, tuned_gradient_boost_prediction, \
        tuned_ridge_prediction, tuned_lasso_prediction

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
    model_folder = 'trained_models'
    create_directory(model_folder)

    for symbol in crypto_symbols:
        train_x, test_x, train_y, test_y = prepare_data(symbol)

        print(f'Testing for: {symbol}BRL')
        predictions = test_all_models(train_x, test_x, train_y, test_y)
        print()

        models = {
            'linear_regression': predictions[0],
            'random_forest': predictions[1],
            'support_vector': predictions[2],
            'gradient_boost': predictions[3],
            'ridge': predictions[4],
            'lasso': predictions[5],

            'tuned_linear_regression': predictions[6],
            'tuned_random_forest': predictions[7],
            'tuned_support_vector': predictions[8],
            'tuned_gradient_boost': predictions[9],
            'tuned_ridge': predictions[10],
            'tuned_lasso': predictions[11],
        }

        symbol_model_folder = os.path.join(model_folder, symbol)
        create_directory(symbol_model_folder)

        for model_name, model in models.items():
            model_file_name = os.path.join(symbol_model_folder, f'{symbol}_{model_name}.pkl')
            save_model(model, model_file_name)

        aux_dataframe = pd.DataFrame()
        aux_dataframe['Open'] = test_x['Open']
        aux_dataframe['High'] = test_x['High']
        aux_dataframe['Low'] = test_x['Low']
        aux_dataframe['test_y (Close)'] = test_y

        aux_dataframe['linear_regression'] = models['linear_regression']
        aux_dataframe['tuned_linear_regression'] = models['tuned_linear_regression']

        aux_dataframe['random_forest'] = models['random_forest']
        aux_dataframe['tuned_random_forest'] = models['tuned_random_forest']

        aux_dataframe['support_vector'] = models['support_vector']
        aux_dataframe['tuned_support_vector'] = models['tuned_support_vector']

        aux_dataframe['gradient_boost'] = models['gradient_boost']
        aux_dataframe['tuned_gradient_boost'] = models['tuned_gradient_boost']

        aux_dataframe['ridge'] = models['ridge']
        aux_dataframe['tuned_ridge'] = models['tuned_ridge']

        aux_dataframe['lasso'] = models['lasso']
        aux_dataframe['tuned_lasso'] = models['tuned_lasso']

        print(aux_dataframe)
        print('-' * 60)
