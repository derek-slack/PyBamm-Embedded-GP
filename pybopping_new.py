import numpy as np
import pybamm
import pybop
import pandas as pd
import os
import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
# ==============================================================================
# 1. Data Loading & Slicing (Isolating Active Discharge)
# ==============================================================================
directory = "src/Data/Processed_Data"  # Update as needed
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

# Normalize time to start at t=0 for the active discharge segment
V = discharge_data['Volts'].to_numpy()
I = discharge_data['Amps'].to_numpy()
time_all = len(V) - 1
t = np.linspace(0, time_all, len(V))

dataset = pybop.Dataset(
    {
        "Time [s]": t,
        "Current [A]": I,
        "Voltage [V]": V,
    }
)

# ==============================================================================
# 2. Base Model & Fixed Physical Parameters
# ==============================================================================
parameter_values = pybamm.ParameterValues("Mohtat2020")

# Lock in the physical geometry of the Samsung 25R.
samsung_25r_fixed = {
    "Electrode width [m]": 0.057,
    "Cell volume [m3]": 1.65e-05,
    "Negative electrode thickness [m]": 8.887e-05,
    "Positive electrode thickness [m]": 6.814e-05,
    "Separator thickness [m]": 1.5e-05,
    "Positive particle radius [m]": 4.5e-06,
    "Negative particle radius [m]": 7.0e-06,
    "Nominal cell capacity [A.h]": 2.5,

    # Relax the lower voltage cutoff to prevent solver crashes during bad guesses
    "Lower voltage cut-off [V]": 2.5,
    "Upper voltage cut-off [V]": 4.5,
}

parameter_values.update(samsung_25r_fixed, check_already_exists=False)

# ==============================================================================
# 3. Thermodynamic Optimization Parameters (Current PyBOP API)
# ==============================================================================
# Optimize the Maximum and Initial concentrations simultaneously to allow a 4.2V start.
parameters_to_optimize = {
    "Maximum concentration in negative electrode [mol.m-3]": pybop.Parameter(
        distribution=pybop.Uniform(20000, 45000)
    ),
    "Maximum concentration in positive electrode [mol.m-3]": pybop.Parameter(
        distribution=pybop.Uniform(40000, 75000)
    ),
    "Initial concentration in negative electrode [mol.m-3]": pybop.Parameter(
        distribution=pybop.Uniform(15000, 40000)  # High for ~4.2V start
    ),
    "Initial concentration in positive electrode [mol.m-3]": pybop.Parameter(
        distribution=pybop.Uniform(100, 25000)  # Low for ~4.2V start
    ),
}

# Pass the PyBOP parameters directly into the parameter dictionary
parameter_values.update(parameters_to_optimize, check_already_exists=False)

# ==============================================================================
# 4. Model Setup & Optimization
# ==============================================================================
model = pybamm.lithium_ion.SPMe()
# | Iter: 50 | Evals: 406| Best Parameters: [36419.82799321 54867.2719805  30949.5359084    396.13023598] | Best Cost: 106.5242345971389
# [36961.29087254 55073.82335454 30833.66422677   105.10321423]
parameters_best = {'Maximum concentration in negative electrode [mol.m-3]': np.float64(36961.29087254), 'Maximum concentration in positive electrode [mol.m-3]': np.float64(55073.82335454), 'Initial concentration in negative electrode [mol.m-3]': np.float64(30833.66422677 ), 'Initial concentration in positive electrode [mol.m-3]': np.float64(105.10321423)}
parameter_values['Current function [A]'] = 0.2
parameter_values.update(parameters_best)

sim = pybamm.Simulation(model, parameter_values=parameter_values)
sol = sim.solve(t,initial_soc=1)
VS = sol['Voltage [V]'].entries
s = len(VS)
plt.plot(sol['Time [s]'].entries,VS,label='Sim')
plt.plot(t,V,label='Data')
plt.legend()
plt.show()


print(f"Initializing {model.name} Simulator...")
# Simulator takes parameter_values directly, NO standalone parameters list
simulator = pybop.pybamm.Simulator(
    model,
    parameter_values=parameter_values,
    protocol=dataset
)

# Standard problem definition pipeline
cost = pybop.SumSquaredError(dataset)
problem = pybop.Problem(simulator, cost)

options = pybop.PintsOptions(
    verbose=True,
    max_iterations=150,
    max_unchanged_iterations=30,
)
optim = pybop.XNES(problem, options=options)
optim.set_population_size(10)

print("Starting XNES Optimization...")
result = optim.run()

# ==============================================================================
# 5. Results & Visualization
# ==============================================================================
print(f"| Model: {model.name} | Cost: {result.f} |")
print("Optimized Thermodynamic Parameters:")

# Extract specific keys from the PyBOP result dictionary
for key in parameters_to_optimize.keys():
    print(f"  {key}: {result.best_inputs.get(key)}")

pybop.plot.problem(problem, result.best_inputs, title="Optimized OCP Fit - 0.2A Discharge")