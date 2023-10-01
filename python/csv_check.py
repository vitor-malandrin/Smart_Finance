import os
import pandas as pd

dir = 'crypto data'

for file in os.listdir(dir):
    file_path = os.path.join(dir, file)

    dataframe = pd.read_csv(file_path)
    dataframe.info()
    print()
