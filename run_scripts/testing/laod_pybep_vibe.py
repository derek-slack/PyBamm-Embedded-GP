import json
import pybamm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pybop

with open('results.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

Cathode_SOC = np.array(data['Cathode SOC'])
Cathode_OCP = np.array(data['Cathode OCP'])
Anode_SOC = np.array(data['Anode SOC'])
Anode_OCP = np.array(data['Anode OCP'])

def create_OCP(SOC, OCP):
    def OCP_function(sto):
        return pybamm.Interpolant(SOC, OCP, sto, interpolator="cubic")
    return OCP_function

parameters = pybamm.ParameterValues("Mohtat2020")

e, f, g, h = data['Best Parameters']  # these are INDICES, not fractions

# --- FIX: index directly into the SOC arrays, don't divide by 1001 ---
# Anode: axis already matches pybamm convention, leave as-is
anode_func = create_OCP(Anode_SOC, Anode_OCP)
x_0   = Anode_SOC[e-1]
x_100 = Anode_SOC[f-1]

# Cathode: axis runs opposite to pybamm convention — invert it
Cathode_sto = 1.0 - Cathode_SOC[::-1]   # reverse so it's increasing again
Cathode_OCP_sto = Cathode_OCP[::-1]     # keep OCP paired correctly after reversal
cathode_func = create_OCP(Cathode_sto, Cathode_OCP_sto)

y_0   = 1.0 - Cathode_SOC[g-1]   # was 0.9929 -> now correctly high (lithiated at 0% SOC)
y_100 = 1.0 - Cathode_SOC[h-1] + 0.01  # was 0.0836 -> now correctly low (delithiated at 100% SOC)

print(f"x_0={x_0:.4f}  x_100={x_100:.4f}  y_0={y_0:.4f}  y_100={y_100:.4f}")

# --- Sanity check #1: does this window reproduce the right terminal voltages? ---
V_at_100 = cathode_func(y_100).evaluate() - anode_func(x_100).evaluate()
V_at_0   = cathode_func(y_0).evaluate()   - anode_func(x_0).evaluate()
print(f"Predicted OCV at 100% SOC: {float(V_at_100):.3f} V")
print(f"Predicted OCV at 0% SOC:   {float(V_at_0):.3f} V")

c_n_max = parameters["Maximum concentration in negative electrode [mol.m-3]"]
c_p_max = parameters["Maximum concentration in positive electrode [mol.m-3]"]

parameters.update({
    "Initial concentration in negative electrode [mol.m-3]": x_100 * c_n_max,
    "Initial concentration in positive electrode [mol.m-3]": y_100 * c_p_max,
    "Positive electrode OCP [V]": cathode_func,
    "Negative electrode OCP [V]": anode_func,
    # tie cutoffs to the fitted window so discharge stops where your data says it should
    "Upper voltage cut-off [V]": 4.25,
    "Lower voltage cut-off [V]": 3.0,
})

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
if discharge_data.empty:
    raise ValueError("No discharge data (State == 'D') found.")

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


# parameters.update(samsung_25r_fixed, check_already_exists=False)


parameters['Current function [A]'] = 0.2
parameters['Nominal cell capacity [A.h]'] = 2.5

model = pybamm.lithium_ion.SPM()

area = 0.875 * 0.057
neg_thickness = 8.887e-05
pos_thickness = 6.814e-05

parameters.update({
    "Electrode height [m]": 0.875,
    "Electrode width [m]": 0.057,
    "Negative electrode thickness [m]": neg_thickness,
    "Positive electrode thickness [m]": pos_thickness,
    "Separator thickness [m]": 1.5e-05,
    "Cell volume [m3]": 1.65e-05,
    "Nominal cell capacity [A.h]": 2.5,
}, check_already_exists=False)

# --- FIXED: c_max, derived analytically to force 2.5 Ah exactly ---
F = 96485.332
capacity_target = 2.5
neg_window = x_100 - x_0
pos_window = y_0 - y_100

eps_n = parameters["Negative electrode active material volume fraction"]
eps_p = parameters["Positive electrode active material volume fraction"]
print(f"eps_n = {eps_n}, eps_p = {eps_p}")

c_n_max = (capacity_target * 3600) / (F * neg_window * area * neg_thickness * eps_n)
c_p_max = (capacity_target * 3600) / (F * pos_window * area * pos_thickness * eps_p)

parameters.update({
    "Maximum concentration in negative electrode [mol.m-3]": c_n_max,
    "Maximum concentration in positive electrode [mol.m-3]": c_p_max,
    "Initial concentration in negative electrode [mol.m-3]": x_100 * c_n_max,
    "Initial concentration in positive electrode [mol.m-3]": y_100 * c_p_max,
})

sim = pybamm.Simulation(model, parameter_values=parameters)
sol = sim.solve(t)

VS = sol['Voltage [V]'].entries
plt.plot(sol['Time [s]'].entries, VS, label='Sim')
plt.plot(t, V, label='Data')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Voltage [V]')
plt.title('Validation: fitted stoichiometric window vs measured discharge')
plt.show()

# --- FREE: only particle radius — governs diffusion limitation / knee sharpness,
#     NOT total capacity, so it can't reintroduce the magnitude blow-up ---
free_parameters = {
    "Negative particle radius [m]": pybop.Parameter(
        distribution=pybop.Uniform(4e-6, 10e-6),
    ),
    "Positive particle radius [m]": pybop.Parameter(
        distribution=pybop.Uniform(3e-6, 7e-6),
    ),
}


parameters.update(free_parameters)

print(f"Initializing {model.name} Simulator...")
# Simulator takes parameter_values directly, NO standalone parameters list
simulator = pybop.pybamm.Simulator(
    model,
    parameter_values=parameters,
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
# print(f"| Model: {model.name} | Cost: {result.x} |")
# print("Optimized Thermodynamic Parameters:")

# Extract specific keys from the PyBOP result dictionary
for key in free_parameters.keys():
    print(f"  {key}: {result.best_inputs.get(key)}")

pybop.plot.problem(problem, result.best_inputs, title="Optimized OCP Fit - 0.2A Discharge")

# # --- FIX: no initial_soc — it would overwrite the concentrations you just set ---
# sol = sim.solve(t)
#
# VS = sol['Voltage [V]'].entries
# plt.plot(sol['Time [s]'].entries, VS, label='Sim')
# plt.plot(t, V, label='Data')
# plt.legend()
# plt.xlabel('Time [s]')
# plt.ylabel('Voltage [V]')
# plt.title('Validation: fitted stoichiometric window vs measured discharge')
# plt.show()