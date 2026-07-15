import pandas as pd
import matplotlib
matplotlib.use('Webagg')
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(legacy='1.25')

import os
import re

d_str = '/Users/derekslack/Pybamm-Embedded-GP-live/src/NASAData-raw'
directory = os.fsencode('/Users/derekslack/Pybamm-Embedded-GP-live/src/NASAData-raw')
DFs = []

for filename in os.listdir(d_str):
    if filename.endswith(".csv"):
        path = os.path.join(d_str, filename)

        cleaned_lines = []

        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i < 2:  # keep header
                    cleaned_lines.append(line)
                    continue

                parts = line.strip().split(',')
                if i == 2:
                    EXPECTED_NUM_COLUMNS = len(parts) - 1
                # CASE 1: normal row (no thousands split)
                if len(parts) == EXPECTED_NUM_COLUMNS:
                    cleaned_lines.append(line)
                    continue

                # CASE 2: broken row (extra column due to comma in first entry)
                # fix by merging leading numeric pieces
                while len(parts) > EXPECTED_NUM_COLUMNS:
                    if parts[0].isdigit() and parts[1].isdigit():
                        parts[0] += parts[1]
                        parts.pop(1)
                    else:
                        break

                cleaned_lines.append(','.join(parts) + '\n')

        from io import StringIO
        df = pd.read_csv(StringIO(''.join(cleaned_lines)), skiprows=2)
        df_discharge = df[df["State"] == "D"]
        DFs.append(df_discharge)

def convert_time(t_vec):
    first_char = 11
    t_converted= np.zeros(len(t_vec))
    for i, t_i in enumerate(t_vec):
        t_hours = float(t_i[-first_char:-first_char+2])*3600
        t_minutes = float(t_i[-first_char+3:-first_char+5])*60
        t_seconds = float(t_i[-first_char+6:-first_char+8])
        t_milliseconds = float(t_i[-3:])
        t_converted[i] = t_hours + t_minutes + t_seconds + t_milliseconds
    return t_converted


for df in DFs:
    Volt = df['Volts'].to_numpy()
    Amps = df['Amps'].to_numpy()
    # Amps[2464:] = -Amps[2464:]
    time = df['TestTime'].to_numpy()
    normtime = convert_time(time)
    t = normtime - normtime[0]
    # Temp = df['Temp 2'].to_numpy()
    plt.plot(t,Volt)
plt.show()



