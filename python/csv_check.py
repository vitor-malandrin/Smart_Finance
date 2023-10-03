import os
import pandas as pd

dir = 'crypto data'


for file in [f for f in os.listdir(dir) if f.endswith(".csv")]:
    file_path = os.path.join(dir, file)

    dataframe = pd.read_csv(file_path)
    dataframe.info()
    print()
