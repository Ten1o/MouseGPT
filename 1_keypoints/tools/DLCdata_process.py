# import pandas as pd

# start_row = 0
# df = pd.read_csv('1DLC_resnet101_upMay15shuffle1_120000.csv', skiprows=range(1, start_row))
# data_dict = df.set_index(df.columns[0]).to_dict(orient='index')


import csv
import numpy as np
import scipy.io as sio
data_dict = {}
start_row = 1

with open('1DLC_resnet101_upMay15shuffle1_120000.csv', 'r') as file:
    csv_reader = csv.reader(file)
    current_row = 0
    for row in csv_reader:
        current_row += 1
        if current_row >= start_row:
            key = row[0]
            value = row[1:]
            data_dict[key] = value

print(data_dict)
sio.savemat('dlc_result.mat', data_dict)





