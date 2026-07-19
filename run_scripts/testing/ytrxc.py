import json
import pybamm
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import pybop

# ==============================================================================
# 1. Load PyBEP Data & Create Safe Interpolants
# ==============================================================================
with open('results.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

Cathode_SOC = np.array(data['Cathode SOC full'])
Cathode_OCP = np.array(data['Cathode OCP full'])
Anode_SOC = np.array(data['Anode SOC full'])
Anode_OCP = np.array(data['Anode OCP full'])

# Function to scrub duplicate noise and generate a strictly increasing interpolant
def create_OCP(SOC, OCP):
    soc_clean, idx = np.unique(SOC, return_index=True)
    ocp_clean = OCP[idx]
    def OCP_function(sto):
        return pybamm.Interpolant(soc_clean, ocp_clean, sto, interpolator="cubic")
    return OCP_function

cathode_func = create_OCP(Cathode_SOC, Cathode_OCP)
anode_func = create_OCP(Anode_SOC, Anode_OCP)

# ==============================================================================
# 2. Extract Exact Thermodynamic Limits[cite: 2]
# ==============================================================================
e, f, g, h = data['Best Parameters']

# Extract minimum and maximum stoichiometric bounds
y_min = Cathode_SOC[e]
y_max = Cathode_SOC[f]
x_min = Anode_SOC[g]
x_max = Anode_SOC[h]

# Map physical states for a discharge starting at 100% Cell SOC (~4.18V)
y_init = y_min   # Cathode is delithiated
x_init = x_max   # Anode is fully lithiated

# Map physical states for the end of discharge (0% Cell SOC, ~3.0V)
y_final = y_max  # Cathode is lithiated
x_final = x_min  # Anode is delithiated

print(f"Stoichiometric Limits -> x_init:{x_init:.4f}  x_final:{x_final:.4f}  y_init:{y_init:.4f}  y_final:{y_final:.4f}")

# Sanity Check
V_at_start = float(cathode_func(y_init).evaluate() - anode_func(x_init).evaluate())
V_at_end = float(cathode_func(y_final).evaluate() - anode_func(x_final).evaluate())
print(f"Predicted OCV at Start: {V_at_start:.3f} V")
print(f"Predicted OCV at End:   {V_at_end:.3f} V")

# ==============================================================================
# 3. Parameterize the Base Model
# ==============================================================================
parameters = pybamm.ParameterValues("Mohtat2020")

# Inject PyBEP thermodynamics and define physical geometry
parameters.update({
    "Positive electrode OCP [V]": cathode_func,
    "Negative electrode OCP [V]": anode_func,
    "Electrode height [m]": 0.875,
    "Electrode width [m]": 0.057,
    "Negative electrode thickness [m]": 8.887e-05,
    "Positive electrode thickness [m]": 6.814e-05,
    "Separator thickness [m]": 1.5e-05,
    "Cell volume [m3]": 1.65e-05,
    "Nominal cell capacity [A.h]": 2.5,
    "Current function [A]": 0.2,
    "Upper voltage cut-off [V]": V_at_start + 0.1,
    "Lower voltage cut-off [V]": V_at_end - 0.1,
}, check_already_exists=False)

# ==============================================================================
# 4. Analytically Force 2.5 Ah Capacity
# ==============================================================================
F = 96485.332
capacity_target = 2.5
area = 0.875 * 0.057

neg_window = x_init - x_final
pos_window = y_final - y_init

eps_n = parameters["Negative electrode active material volume fraction"]
eps_p = parameters["Positive electrode active material volume fraction"]

c_n_max = (capacity_target * 3600) / (F * neg_window * area * parameters["Negative electrode thickness [m]"] * eps_n)
c_p_max = (capacity_target * 3600) / (F * pos_window * area * parameters["Positive electrode thickness [m]"] * eps_p)

# Lock in the theoretical maximums and exact initial absolute concentrations
parameters.update({
    "Maximum concentration in negative electrode [mol.m-3]": c_n_max,
    "Maximum concentration in positive electrode [mol.m-3]": c_p_max,
    "Initial concentration in negative electrode [mol.m-3]": x_init * c_n_max,
    "Initial concentration in positive electrode [mol.m-3]": y_init * c_p_max,
})

# ==============================================================================
# 5. Load Processed Discharge Data
# ==============================================================================
directory = "/Users/derekslack/Pybamm-Embedded-GP-live/src/Data/Processed_Data/ESPSCoR_Char_B2 - 011.csv"
# target_file = "ESPSCoR_Char_B2 - 011.csv"
# for root, dirs, files in os.walk(directory):
#     for file in files:
#         if file.endswith(".csv"):
#             target_file = os.path.join(root, file)
#             break
#     if target_file:
#         break

# if not target_file:
#     raise FileNotFoundError("No CSV data files found.")

# print(f"Loading data from: {target_file}")
df = pd.read_csv(directory)
discharge_data = df[df['State'] == 'D'].copy().reset_index(drop=True)

V = discharge_data['Volts'].to_numpy()
I = discharge_data['Amps'].to_numpy()
time_all = len(V) - 1
t = np.linspace(0, time_all, len(V))

dataset = pybop.Dataset({
    "Time [s]": t,
    "Current [A]": I,
    "Voltage [V]": V,
})

# ==============================================================================
# 6. Base Validation (Before Kinetic Optimization)
# ==============================================================================
model = pybamm.lithium_ion.SPMe()
sim = pybamm.Simulation(model, parameter_values=parameters)

# # NO initial_soc override. We let the calculated concentrations dictate the start.
sol = sim.solve(t)

VS = sol['Voltage [V]'].entries
plt.figure(figsize=(10, 6))
plt.plot(sol['Time [s]'].entries, VS, label='Simulated Baseline')
plt.plot(t, V, label='Experimental Data')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Voltage [V]')
plt.title('Validation: PyBEP Thermodynamics vs Measured Discharge')
plt.show()

# ==============================================================================
# 7. PyBOP Kinetic Optimization (Particle Radii)
# ==============================================================================
free_parameters = {
    "Negative particle radius [m]": pybop.Parameter(
        distribution=pybop.Uniform(1e-6, 15e-6),
    ),
    "Positive particle radius [m]": pybop.Parameter(
        distribution=pybop.Uniform(1e-6, 15e-6),
    ),
}

parameters.update(free_parameters)

print(f"Initializing {model.name} Simulator...")
simulator = pybop.pybamm.Simulator(
    model,
    parameter_values=parameters,
    protocol=dataset
)

cost = pybop.SumSquaredError(dataset)
problem = pybop.Problem(simulator, cost)

options = pybop.PintsOptions(
    verbose=True,
    max_iterations=100,
    max_unchanged_iterations=20,
)
optim = pybop.XNES(problem, options=options)
optim.set_population_size(8)

print("Starting XNES Optimization...")
result = optim.run()

print(f"\n| Optimization Complete | Cost: {result.f} |")
print("Optimized Kinetic Parameters:")
for key in free_parameters.keys():
    print(f"  {key}: {result.best_inputs.get(key):.4e}")

pybop.plot.problem(problem, result.best_inputs, title="Optimized Full Cell Fit")