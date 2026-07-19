import json
import pybamm
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import os

with open('results.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

Cathode_SOC = np.array(data['Cathode SOC full'])
Cathode_OCP = np.array(data['Cathode OCP full'])

Anode_SOC = np.array(data['Anode SOC full'])
Anode_OCP = np.array(data['Anode OCP full'])


def create_OCP(SOC, OCP):
    def OCP_function(sto):
        return pybamm.Interpolant(SOC,OCP,sto,interpolator="cubic")
    return OCP_function

cathode_func = create_OCP(Cathode_SOC, Cathode_OCP)
anode_func = create_OCP(Anode_SOC, Anode_OCP)
parameters = pybamm.ParameterValues("Mohtat2020")

e, f, g, h = data['Best Parameters']

# 2. Convert indices to exact stoichiometry fractions
x_0   = e / 1001.0   # Anode min lithiation
x_100 = f / 1001.0   # Anode max lithiation
y_0   = g / 1001.0   # Cathode max lithiation
y_100 = h / 1001.0   # Cathode min lithiation

# 3. Reference Max Concentrations from your baseline template
c_n_max = parameters["Maximum concentration in negative electrode [mol.m-3]"]
c_p_max = parameters["Maximum concentration in positive electrode [mol.m-3]"]

# 4. Update Initial Concentrations (State at 100% SOC)
parameters.update({
    "Initial concentration in negative electrode [mol.m-3]": x_100 * c_n_max,
    "Initial concentration in positive electrode [mol.m-3]": y_100 * c_p_max,
    "Positive electrode OCP [V]": cathode_func,
    "Negative electrode OCP [V]": anode_func,

})

directory = "/Users/derekslack/Pybamm-Embedded-GP-live/src/Data/Processed_Data"  # Update as needed
target_file = None

for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".csv"):
            target_file = os.path.join(root, file)
            break
    if target_file:
        break

if not target_file:
    raise FileNotFoundError("No CSV data files found in the specified directory.")

print(f"Loading data from: {target_file}")
df = pd.read_csv(target_file)

# Strictly isolate the discharge ('D') state.
# We drop 'R' states so the optimizer doesn't fight relaxation kinetics.
discharge_data = df[df['State'] == 'D'].copy()
discharge_data.reset_index(drop=True, inplace=True)

if discharge_data.empty:
    raise ValueError("No discharge data (State == 'D') found in the dataset.")

# Normalize time to start at t=0 for the active discharge segment]
V = discharge_data['Volts'].to_numpy()
I = discharge_data['Amps'].to_numpy()
time_all = len(V) - 1
t = np.linspace(0, time_all, len(V))

parameters['Current function [A]'] = 0.2

parameters['Nominal cell capacity [A.h]'] = 2.5
# samsung_25r_fixed = {
#     # Cathode dimensions from abstract: 875 mm length, 57 mm width
#     "Positive electrode width [m]": 0.057,
#     "Positive electrode height [m]": 0.875,   # Exact length from abstract
#
#     # Anode dimensions from abstract: 930 mm length, 57 mm width
#     "Negative electrode width [m]": 0.057,
#     "Negative electrode height [m]": 0.930,   # Exact length from abstract
#
#     # Note: PyBaMM sometimes uses a single "Electrode height/width" parameter
#     # if it assumes balanced lengths. If so, use the average or keep them distinct:
#     "Electrode width [m]": 0.057,
#     "Electrode height [m]": 0.875,
#
#     "Cell volume [m3]": 1.65e-05,
#     "Nominal cell capacity [A.h]": 2.5,
#     "Lower voltage cut-off [V]": 2.5,
#     "Upper voltage cut-off [V]": 4.2,
# }

# parameters.update(samsung_25r_fixed, check_already_exists=False)
model = pybamm.lithium_ion.SPMe()

sim = pybamm.Simulation(model, parameter_values=parameters)
sol = sim.solve(t,initial_soc=1)
VS = sol['Voltage [V]'].entries
s = len(VS)
plt.plot(sol['Time [s]'].entries,VS,label='Sim')
plt.plot(t,V,label='Data')
plt.legend()
plt.show()


h=1


