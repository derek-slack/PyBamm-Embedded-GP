import numpy as np
import pybamm
import pybop
import pandas as pd
import matplotlib
matplotlib.use('Tkagg')
import matplotlib.pyplot as plt
import os
from io import StringIO

# --- 1. DATA LOADING & ALIGNMENT ---
d_str = '/Users/derekslack/Pybamm-Embedded-GP-live/src/NASAData-raw/OCV/'
DFs = []

for filename in os.listdir(d_str):
    if filename.endswith(".csv"):
        path = os.path.join(d_str, filename)
        cleaned_lines = []
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i < 2:
                    cleaned_lines.append(line)
                    continue
                parts = line.strip().split(',')
                if i == 2:
                    EXPECTED_NUM_COLUMNS = len(parts) - 1
                if len(parts) == EXPECTED_NUM_COLUMNS:
                    cleaned_lines.append(line)
                    continue
                while len(parts) > EXPECTED_NUM_COLUMNS:
                    if parts[0].isdigit() and parts[1].isdigit():
                        parts[0] += parts[1]
                        parts.pop(1)
                    else:
                        break
                cleaned_lines.append(','.join(parts) + '\n')

        df = pd.read_csv(StringIO(''.join(cleaned_lines)), skiprows=2)
        DFs.append(df[df["State"] == "D"])

def convert_time(t_vec):
    first_char = 11
    t_converted = np.zeros(len(t_vec))
    for i, t_i in enumerate(t_vec):
        t_hours = float(t_i[-first_char:-first_char+2])*3600
        t_minutes = float(t_i[-first_char+3:-first_char+5])*60
        t_seconds = float(t_i[-first_char+6:-first_char+8])
        t_milliseconds = float(t_i[-3:])
        t_converted[i] = t_hours + t_minutes + t_seconds + t_milliseconds
    return t_converted

# Master time vector for proper alignment
t_master = convert_time(DFs[0]['TestTime'].to_numpy())
t_master = t_master - t_master[0]

V_interp_list = []
I_interp_list = []

for df in DFs:
    time = convert_time(df['TestTime'].to_numpy())
    normtime = time - time[0]
    # Interpolate all datasets onto the master time vector to prevent index averaging errors
    V_interp = np.interp(t_master, normtime, df['Volts'].to_numpy())
    I_interp = np.interp(t_master, normtime, df['Amps'].to_numpy())
    V_interp_list.append(V_interp)
    I_interp_list.append(I_interp)

Volt = np.mean(V_interp_list, axis=0)
Amps = np.mean(I_interp_list, axis=0)
t = t_master

dataset = pybop.Dataset(
    {
        "Time [s]": t,
        "Current [A]": Amps,
        "Voltage [V]": Volt,
    }
)

# --- 2. MODEL & PARAMETER SETUP ---
parameter_values = pybamm.ParameterValues("Chen2020")
# Lock the capacity to the Samsung 25R spec
parameter_values["Nominal cell capacity [A.h]"] = 2.5

# For a slow pseudo-OCV discharge, fit the thermodynamic limits, not the geometry.
parameters = {
    "Initial concentration in negative electrode [mol.m-3]": pybop.Parameter(
        pybop.Gaussian(28000, 2000, truncated_at=[20000, 32000]),
    ),
    "Initial concentration in positive electrode [mol.m-3]": pybop.Parameter(
        pybop.Gaussian(15000, 2000, truncated_at=[10000, 25000]),
    ),
    "Negative electrode active material volume fraction": pybop.Parameter(
        pybop.Gaussian(0.75, 0.05, truncated_at=[0.6, 0.9]),
    ),
    "Positive electrode active material volume fraction": pybop.Parameter(
        pybop.Gaussian(0.65, 0.05, truncated_at=[0.5, 0.8]),
    )
}

parameter_values = pybamm.get_size_distribution_parameters(parameter_values)
parameter_values.set_initial_state(1)
parameter_values.update(parameters)

current_interpolant = pybamm.Interpolant(t, Amps, pybamm.t)
parameter_values["Current function [A]"] = current_interpolant

# --- 3. OPTIMIZATION ---
model = pybamm.lithium_ion.SPMe()

print(f"Running {model.name}")
simulator = pybop.pybamm.Simulator(
    model, parameter_values=parameter_values, protocol=dataset
)
cost = pybop.SumSquaredError(dataset)
problem = pybop.Problem(simulator, cost)

# Give the optimizer enough iterations and population size to explore a 4D space
options = pybop.PintsOptions(
    verbose=True,
    max_iterations=150,
    max_unchanged_iterations=15,
)
optim = pybop.XNES(problem, options=options)
optim.set_population_size(15)
result = optim.run()

print(f"| Model: {model.name} | Results: {result.x} |")
pybop.plot.problem(result.problem, inputs=result.best_inputs, title=model.name)