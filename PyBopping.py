import numpy as np
import pybamm
import pybop
import pandas as pd
import matplotlib
matplotlib.use('Tkagg')
import matplotlib.pyplot as plt
import os
from io import StringIO

directory = "/Users/derekslack/Pybamm-Embedded-GP-live/src/Data/Processed_Data"


for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".csv"):
            file_os = os.path.join(directory, file)
            f = pd.read_csv(file_os)

            # 1. Find all row indices where the state is 'D' (Discharge)
            d_indices = f.index[f['State'] == 'D'].tolist()

            if not d_indices:
                continue  # Skip the file if it doesn't contain a discharge

            first_d_idx = d_indices[0]
            last_d_idx = d_indices[-1]

            # 2. Walk backward to find the start of the pre-discharge rest
            start_idx = first_d_idx
            # while start_idx > 0 and f.loc[start_idx - 1, 'State'] == 'R':
            #     start_idx -= 1
            # start_idx+=1
            # 3. Walk forward to find the end of the post-discharge rest
            end_idx = last_d_idx
            # while end_idx < len(f) - 1 and f.loc[end_idx + 1, 'State'] == 'R':
            #     end_idx += 1
            # end_idx-=1
            # 4. Slice the dataframe using these exact boundaries
            target_data = f.loc[start_idx:end_idx].copy()

            V = target_data['Volts'].to_numpy()
            I = target_data['Amps'].to_numpy()



time_rest_1 = first_d_idx - start_idx
time_discharge = len(d_indices)
time_rest_2 = end_idx - last_d_idx

time_all = len(V) - 1
t = np.linspace(0, time_all, time_all + 1)
dataset = pybop.Dataset(
    {
        "Time [s]": t,
        "Current [A]":0.2*np.ones(len(V)),
        "Voltage [V]": V,
    }
)
#
experiment = pybamm.Experiment(
    [
        f"Discharge at 0.2 A for {time_discharge-1} seconds",

    ]
)

# [37623.28985312 54363.69593941 16231.60734993 17758.58944916]
# Initial
# parameters: [19145.42028406 68924.70679509  7209.48216776 30995.45183295]
# Optimised
# parameters: [59298.32155588 78120.42094269 39420.56601211 59148.68352712]
# {'Maximum concentration in negative electrode [mol.m-3]': np.float64(16807.77395121404), 'Maximum concentration in positive electrode [mol.m-3]': np.float64(20202.901826779485), 'Initial concentration in negative electrode [mol.m-3]': np.float64(3059.616664487098), 'Initial concentration in positive electrode [mol.m-3]': np.float64(13095.698991886176)}

parameters = {
    "Electrode height [m]": pybop.Parameter(
        distribution=pybop.Uniform(0.3, 2.0)),
    "Negative electrode thickness [m]": pybop.Parameter(
            distribution=pybop.Uniform(1e-6, 1e-4)),
    "Positive electrode thickness [m]": pybop.Parameter(
            distribution=pybop.Uniform(1e-6, 1e-4)),
    # ),
    "Maximum concentration in negative electrode [mol.m-3]": pybop.Parameter(
        distribution=pybop.Uniform(29000, 35000),
    ),
    "Maximum concentration in positive electrode [mol.m-3]": pybop.Parameter(
            distribution=pybop.Uniform(41217.0, 61217.0),
        ),
    "Initial concentration in negative electrode [mol.m-3]": pybop.Parameter(
            distribution=pybop.Uniform(29000, 34000),
        ),
    "Initial concentration in positive electrode [mol.m-3]": pybop.Parameter(
            distribution=pybop.Uniform(9000, 15000),
        ),

}
parameter_values = pybamm.ParameterValues("Mohtat2020")
#
# samsung_25r_updates = {
#     # Cell Geometry
#     "Electrode height [m]": 0.875,  # Unwound length from Kartini paper
#     "Electrode width [m]": 0.057,  # Unwound width from Kartini paper
#     "Electrode area [m2]": 0.875*0.057,
#     "Cell volume [m3]": 1.65e-05,  # Volume of an 18650 cylinder
#
#     # Electrode Thicknesses (kept from prior optimisation — plausible range for 18650 NMC/graphite)
#     "Negative electrode thickness [m]": 8.88722531e-05,
#     "Positive electrode thickness [m]": 6.81458654e-05,
#     "Separator thickness [m]": 1.5e-05,
#
#     # Particle Radii — corrected to secondary/agglomerate scale, not primary crystallite scale
#     # Kartini et al. (2023) reports NMC111 primary crystallites at 300-500nm, but explicitly
#     # notes these are "greatly agglomerate" and does not report secondary particle size.
#     # Using NMC111 secondary particle literature value instead (Rueß et al. 2020, ACS Appl.
#     # Energy Mater.: 9 μm secondary particle diameter for fine-grained NMC111 processing).
#     "Positive particle radius [m]": 4.5e-06,  # ~9 micron secondary particle diameter / 2
#     # Graphite anodes are typically less agglomerated; primary particle size is closer to
#     # the effective diffusion radius. Testing within the 5-10 micron literature range.
#     "Negative particle radius [m]": 7.0e-06,
#     "Nominal cell capacity [A.h]": 2.5
# }
samsung_25r_updates = {
        # Unwound dimensions extracted from reverse engineering
        # "Electrode height [m]": 0.875,  # Unwound length from Kartini paper
        "Electrode width [m]": 0.057,
     #  875 mm length, 57 mm width[cite: 1]
        "Cell volume [m3]": 1.65e-05,  # Standard cylindrical 18650 interior volume
        #
        # # Using standardized thicknesses (rejecting the physically impossible 1.41mm/1.28mm errors)[cite: 1]
        "Negative electrode thickness [m]": 8.887e-05,
        "Positive electrode thickness [m]": 6.814e-05,
        "Separator thickness [m]": 1.5e-05,  # Corrected from the 0.15mm error[cite: 1]
        #
        # # Particle Radii: Sized for secondary agglomerates, NOT the 300-500nm primary crystallite scale[cite: 1]
        "Positive particle radius [m]": 4.5e-06,
        "Negative particle radius [m]": 7.0e-06,

        "Nominal cell capacity [A.h]": 2.5,
# --- NEW: Expanding Voltage Limits for Optimization ---
        "Upper voltage cut-off [V]": 4.5,  # Give the optimizer headroom (Default is often too tight)
        "Lower voltage cut-off [V]": 3.0,  # Give the optimizer floor room
    # --- Literature-Backed Thermodynamic Constants ---
    #
    #     # Maximum theoretical lattice concentrations
    #     "Maximum concentration in negative electrode [mol.m-3]": 33133.0,
    #     "Maximum concentration in positive electrode [mol.m-3]": 51217.0,

        # Starting stoichiometry for a 4.2V fully charged state
        # "Initial concentration in negative electrode [mol.m-3]": 33176.0,  # ~95% full
        # "Initial concentration in positive electrode [mol.m-3]": 10804.0,  # ~25% full


    }


parameter_values.update(samsung_25r_updates, check_already_exists=False)
# parameter_values = pybamm.get_size_distribution_parameters(parameter_values)
# parameter_values["Negative particle radius [m]"] = 7.77983119e-07
# parameter_values["Positive particle radius [m]"] = 5.96532330e-06

# Initial
# parameters: [1.26397504e-05 1.61884775e-05 2.43799087e+00 4.14049531e+04
#              5.26329984e+04 1.27325701e+04 1.86737934e+04]
# Optimised
# parameters: [8.88722531e-05 6.81458654e-05 2.32234349e+00 4.51571679e+04
#              6.37991454e+04 3.52763550e+04 1.46025037e+03]
# #
# samsung_25r_updates = {
#     # Cell Geometry
#     "Electrode height [m]": 0.875,  # Unwound length from Kartini paper
#     "Electrode width [m]": 0.057,  # Unwound width from Kartini paper
#     "Cell volume [m3]": 1.65e-05,  # Volume of an 18650 cylinder
#
#     # Electrode Thicknesses (Correcting the decimal error in the paper)
#     "Negative electrode thickness [m]": 8.88722531e-05,
#     "Positive electrode thickness [m]": 6.81458654e-05,
#     "Separator thickness [m]": 1.5e-05,
#
#     # Particle Radii (Crucial for fixing the GP relaxation artifact)
#     "Positive particle radius [m]": 2.0e-05,  # 400nm diameter from SEM
#     # Note: Assuming graphite anode radius is fairly standard if not in the paper
#     "Negative particle radius [m]": 2.0e-06,
#     # "Nominal cell capacity [A.h]": 2.32234349e+00,
#
# }

# parameter_values["Maximum concentration in positive electrode [mol.m-3]"] = 6.35463501e+04
# parameter_values["Maximum concentration in negative electrode [mol.m-3]"] = 4.05762239e+04
# parameter_values["Initial concentration in negative electrode [mol.m-3]"] = 16231.60734993
# parameter_values["Initial concentration in positive electrode [mol.m-3]"] = 17758.58944916

#
# current_interpolant = pybamm.Interpolant(t, I, pybamm.t)
parameter_values["Current function [A]"] = 0.2
# parameter_values["Lower voltage cut-off [V]"]= 3.0
parameters_best = {'Negative electrode thickness [m]': np.float64(8.912356257418112e-05), 'Positive electrode thickness [m]': np.float64(8.648675085560248e-05), 'Electrode height [m]': np.float64(0.9612615730011502), 'Maximum concentration in negative electrode [mol.m-3]': np.float64(33681.923831265995), 'Maximum concentration in positive electrode [mol.m-3]': np.float64(54682.77277474358)}#, 'Initial concentration in negative electrode [mol.m-3]': np.float64(31908.773612908684), 'Initial concentration in positive electrode [mol.m-3]': np.float64(9199.20397282335)}
# parameter_values.update(parameters)
# parameter_values.set_initial_state(1)
parameter_values.update(parameters_best)


model = pybamm.lithium_ion.SPMe()
sim = pybamm.Simulation(model, parameter_values=parameter_values)
sol = sim.solve(t,initial_soc=1)
VS = sol['Voltage [V]'].entries
s = len(VS)
plt.plot(sol['Time [s]'].entries,VS,label='Sim')
plt.plot(t,V,label='Data')
plt.legend()
plt.show()
print(f"Running {model.name}")
simulator = pybop.pybamm.Simulator(
    model,
    parameter_values=parameter_values,
    protocol=dataset
)
# simulator.debug_mode = True
cost = pybop.SumSquaredError(dataset)
problem = pybop.Problem(simulator, cost)
options = pybop.PintsOptions(
    verbose=True,
    max_iterations=100,
    max_unchanged_iterations=20,
)
optim = pybop.XNES(problem, options=options)
optim.set_population_size(8)
result = optim.run()

print(f"| Model: {model.name} | Results: {result.x} |")
print(result.best_inputs)
pybop.plot.problem(problem, result.best_inputs, title="Optimised Comparison")
# pybop.plot.surface(result, title=model.name)

# {'Maximum concentration in negative electrode [mol.m-3]': np.float64(42540.685716710934), 'Maximum concentration in positive electrode [mol.m-3]': np.float64(20253.76809468743), 'Initial concentration in negative electrode [mol.m-3]': np.float64(12130.49868510342), 'Initial concentration in positive electrode [mol.m-3]': np.float64(345.2146429764973)}
# {'Maximum concentration in negative electrode [mol.m-3]': np.float64(59561.4229235437), 'Maximum concentration in positive electrode [mol.m-3]': np.float64(22516.16640625289), 'Initial concentration in negative electrode [mol.m-3]': np.float64(8797.36686394968), 'Initial concentration in positive electrode [mol.m-3]': np.float64(19.115157043760888)}
# {'Maximum concentration in negative electrode [mol.m-3]': np.float64(29256.171095205038), 'Maximum concentration in positive electrode [mol.m-3]': np.float64(51580.23461329089), 'Initial concentration in negative electrode [mol.m-3]': np.float64(17449.28372777705), 'Initial concentration in positive electrode [mol.m-3]': np.float64(6029.300038327217)}
