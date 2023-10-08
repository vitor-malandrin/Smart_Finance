import pandas as pd
from sklearn.model_selection import train_test_split

def read_csv(symbol):
    url = f'https://www.cryptodatadownload.com/cdd/Binance_{symbol}BRL_d.csv'
    dataframe = pd.read_csv(url, skiprows=1)
    return dataframe

def prepare_data(symbol):
    dataframe = read_csv(symbol)

    x = dataframe[['Open', 'High', 'Low']]
    y = dataframe['Close']

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25)
    return train_x, test_x, train_y, test_y
