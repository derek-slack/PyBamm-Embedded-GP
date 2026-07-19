import csv
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

directory = "/Users/derekslack/Pybamm-Embedded-GP-live/src/Data/Processed_Data"
for root,dirs,files in os.walk(directory):
    for file in files:
       if file.endswith(".csv"):
            file_os = os.path.join(directory, file)
            f = pd.read_csv(file_os)
            V = f['Volts'].to_numpy()
            I = f['Amps'].to_numpy()
            P = I*V

            time = np.linspace(0,len(V)-1,len(V))
            # plt.plot(time,V, label=file + 'Voltage')
            plt.plot(time,I/2.5,label=file + ' Power')
plt.legend()
plt.show()
