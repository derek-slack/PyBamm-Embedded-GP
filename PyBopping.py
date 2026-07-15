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

# - Positive electrode capacity
# - Negative electrode capacity
# - Initial stoichiometries x_0, y_0
# "Maximum concentration in positive electrode [mol.m-3]": 63104.0
# "Initial concentration in negative electrode [mol.m-3]": 29866.0,
# "Initial concentration in positive electrode [mol.m-3]": 17038.0
# [37623.28985312 54363.69593941 16231.60734993 17758.58944916]
parameters = {
    "Maximum concentration in positive electrode [mol.m-3]": pybop.Parameter(
        pybop.Gaussian(54363.69593941, 5e3),
    ),
    "Maximum concentration in negative electrode [mol.m-3]": pybop.Parameter(
        pybop.Gaussian(37623.28985312, 5e3),
    ),
    "Initial concentration in negative electrode [mol.m-3]": pybop.Parameter(
        pybop.Gaussian(16231.60734993, 5e3),
    ),
    "Initial concentration in positive electrode [mol.m-3]": pybop.Parameter(
        pybop.Gaussian(17758.58944916, 5e3),
    ),
    "Negative electrode thickness [m]": pybop.Parameter(
        pybop.Gaussian(1.28e-05, 0.5e-05, truncated_at=[7e-06, 6e-04]),
    ),
    "Positive electrode thickness [m]": pybop.Parameter(
        pybop.Gaussian(1.41e-05, 0.5e-05, truncated_at=[2e-06, 1e-04]),
    ),
    "Nominal cell capacity [A.h]": pybop.Parameter(
        pybop.Gaussian(2.4, 0.05, truncated_at=[2.2, 2.5]),
    )
}


parameter_values = pybamm.ParameterValues("Mohtat2020")
# [8.35373054e-05 3.78259750e-05 4.05762239e+04 6.35463501e+04
#  1.82098527e+04 1.56431275e+04]
# parameter_values["Negative electrode thickness [m]"] = 8.35373054e-05
# parameter_values["Positive electrode thickness [m]"]  = 3.78259750e-05
samsung_25r_updates = {
    # Cell Geometry
    "Electrode height [m]": 0.875,  # Unwound length from Kartini paper
    "Electrode width [m]": 0.057,  # Unwound width from Kartini paper
    "Cell volume [m3]": 1.65e-05,  # Volume of an 18650 cylinder

    # Electrode Thicknesses (Correcting the decimal error in the paper)
    # "Negative electrode thickness [m]": 1.28e-05,
    # "Positive electrode thickness [m]": 1.41e-05,
    "Separator thickness [m]": 1.5e-05,

    # Particle Radii (Crucial for fixing the GP relaxation artifact)
    "Positive particle radius [m]": 2.0e-07,  # 400nm diameter from SEM
    # Note: Assuming graphite anode radius is fairly standard if not in the paper
    "Negative particle radius [m]": 2.0e-06,
}

parameter_values.update(samsung_25r_updates, check_already_exists=False)
parameter_values = pybamm.get_size_distribution_parameters(parameter_values)
# parameter_values["Negative particle radius [m]"] = 7.77983119e-07
# parameter_values["Positive particle radius [m]"] = 5.96532330e-06

# Initial
# parameters: [1.26397504e-05 1.61884775e-05 2.43799087e+00 4.14049531e+04
#              5.26329984e+04 1.27325701e+04 1.86737934e+04]
# Optimised
# parameters: [8.88722531e-05 6.81458654e-05 2.32234349e+00 4.51571679e+04
#              6.37991454e+04 3.52763550e+04 1.46025037e+03]

samsung_25r_updates = {
    # Cell Geometry
    "Electrode height [m]": 0.875,  # Unwound length from Kartini paper
    "Electrode width [m]": 0.057,  # Unwound width from Kartini paper
    "Cell volume [m3]": 1.65e-05,  # Volume of an 18650 cylinder

    # Electrode Thicknesses (Correcting the decimal error in the paper)
    "Negative electrode thickness [m]": 8.88722531e-05,
    "Positive electrode thickness [m]": 6.81458654e-05,
    "Separator thickness [m]": 1.5e-05,

    # Particle Radii (Crucial for fixing the GP relaxation artifact)
    "Positive particle radius [m]": 2.0e-07,  # 400nm diameter from SEM
    # Note: Assuming graphite anode radius is fairly standard if not in the paper
    "Negative particle radius [m]": 2.0e-06,
    "Nominal cell capacity [A.h]": 2.32234349e+00,
    "Maximum concentration in positive electrode [mol.m-3]":6.37991454e+04,
    "Maximum concentration in negative electrode [mol.m-3]":4.51571679e+04,
    "Initial concentration in negative electrode [mol.m-3]":3.52763550e+04,
    "Initial concentration in positive electrode [mol.m-3]":1.46025037e+03
}

# parameter_values["Maximum concentration in positive electrode [mol.m-3]"] = 6.35463501e+04
# parameter_values["Maximum concentration in negative electrode [mol.m-3]"] = 4.05762239e+04
# parameter_values["Initial concentration in negative electrode [mol.m-3]"] = 16231.60734993
# parameter_values["Initial concentration in positive electrode [mol.m-3]"] = 17758.58944916
parameter_values = pybamm.get_size_distribution_parameters(parameter_values)
parameter_values.set_initial_state(1)

current_interpolant = pybamm.Interpolant(t, Amps, pybamm.t)
parameter_values["Current function [A]"] = current_interpolant
# parameter_values["Lower voltage cut-off [V]"]= 3.0

#
true_values = [parameter_values[p] for p in parameters.keys()]
parameter_values.update(parameters)

model = pybamm.lithium_ion.SPMe()
# solver = pybamm.IDAKLUSolver()
# sim = pybamm.Simulation(model, parameter_values=parameter_values, solver=solver)
# sol = sim.solve(t,t_interp=t)

print(f"Running {model.name}")
simulator = pybop.pybamm.Simulator(
    model, parameter_values=parameter_values, protocol=dataset
)
cost = pybop.SumSquaredError(dataset)
problem = pybop.Problem(simulator, cost)
options = pybop.PintsOptions(
    verbose=True,
    max_iterations=120,
    max_unchanged_iterations=20,
)
optim = pybop.XNES(problem, options=options)
optim.set_population_size(8)
result = optim.run()

print(f"| Model: {model.name} | Results: {result.x} |")
pybop.plot.problem(result.problem, inputs=result.best_inputs, title=model.name)
result.plot_surface(title=model.name)