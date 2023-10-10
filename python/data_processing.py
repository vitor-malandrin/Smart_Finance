import pandas as pd
from sklearn.model_selection import train_test_split

def read_csv(symbol):
    url = f'https://www.cryptodatadownload.com/cdd/Binance_{symbol}BRL_d.csv'
    dataframe = pd.read_csv(url, skiprows=1)
     # Se a coluna não for num faz substituição
    if not pd.api.types.is_numeric_dtype(dataframe[f'Volume {symbol}']):
        dataframe[f'Volume {symbol}'] = dataframe[f'Volume {symbol}'].str.replace('.', '').astype(float)
    if not pd.api.types.is_numeric_dtype(dataframe['Volume BRL']):
        dataframe['Volume BRL'] = dataframe['Volume BRL'].str.replace('.', '').astype(float)
    dataframe = dataframe.drop(['Date', 'Symbol', 'Unix'], axis=1)
    return dataframe

def create_lagged_features(dataframe, lags=5):
    lagged_dataframes = [dataframe.shift(i) for i in range(lags + 1)]
    for i, lagged_dataframe in enumerate(lagged_dataframes):
        dataframe = pd.concat([dataframe, lagged_dataframe[['Open', 'High', 'Low', 'Close']].rename(columns=lambda x: f'{x}_lag_{i}')], axis=1)
    return dataframe.dropna()

def prepare_data(symbol):
    dataframe = read_csv(symbol)

    # Incluindo valores de OHLC dos últimos "n" dias
    dataframe = create_lagged_features(dataframe, 5)
    x = dataframe.drop('Close', axis=1)
    y = dataframe['Close']

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25)
    return train_x, test_x, train_y, test_y
 