import os

from prompt_toolkit.contrib.telnet import protocol

os.environ['JAX_PLATFORM_NAME'] = 'cpu'
import pybamm
import pybamm as pb
import numpy as np
import jax
import jax.numpy as jnp
import pybop

from src.embedded_gp import OPTUNA_HMC_GPS

import pandas as pd
import matplotlib
# matplotlib.use()
import matplotlib.pyplot as plt
import warnings
from Create_Params import ParamUpdate
from src.Data.Processed_Data.Samsung_25R_Parameters import samsung_25r_params


warnings.filterwarnings("ignore")

k = "symmetric Butler-Volmer"
pb.set_logging_level("NOTICE")


save_folder = "figures"
os.makedirs(save_folder, exist_ok=True)

# Set Parameter values

param1 = pybamm.ParameterValues('Chen2020')

def normalize_inputs(inputs, min, max):
    normalized = (inputs - min) / (max - min)
    return normalized

testing = pd.read_csv('/Users/derekslack/Pybamm-Embedded-GP-live/src/Data/Processed_Data/EPSCoR_CP - 013.csv')

i_begin = 828
# i_end = 8763
i_c = 3285
# i_begin = 827
# i_end = 2688
# i_c = i_end
# i_c = i_end

Volt = testing['Volts'].to_numpy()[i_begin:i_c]
Amps = testing['Amps'].to_numpy()[i_begin:i_c]
# Amps[2464:] = -Amps[2464:]
time = testing['TestTime'].to_numpy()[i_begin:i_c]
Temp = testing['EV Temp'].to_numpy()[i_begin:i_c]



testing = []

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

normtime = convert_time(time)
t = normtime - normtime[0]

current_interpolant = pybamm.Interpolant(t, Amps, pybamm.t)#, interpolator="JAX")  # , _num_derivatives=0)
param1["Current function [A]"] = current_interpolant


output_variables = ["Voltage [V]"]

samsung_25r_updates = samsung_25r_params()

param1.update(samsung_25r_updates, check_already_exists=False)
param1["Current function [A]"] = current_interpolant
solver = pybamm.IDAKLUSolver(atol=1e-4, rtol=1e-2, output_variables=output_variables, options={'num_threads':os.cpu_count()-2})



dataset = pybop.Dataset(
    {
        "Time [s]": t,
        "Current [A]": Amps,
        "Voltage [V]": Volt,
    }
)

parameters = {
    "Positive electrode exchange-current density [A.m-2]": pybop.Parameter(
        pybop.Uniform(0.01, 30),
    ),
    "Negative electrode exchange-current density [A.m-2]": pybop.Parameter(
        pybop.Uniform(0.01, 30),
    ),
    'Positive particle diffusivity [m2.s-1]': pybop.Parameter(
        pybop.Uniform(1e-15, 1e-7),
    ),
    'Negative particle diffusivity [m2.s-1]': pybop.Parameter(
        pybop.Uniform(1e-15, 1e-7),
    )
}

parameter_values = pybamm.get_size_distribution_parameters(param1)

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
    max_iterations=200,
    max_unchanged_iterations=30,
)
optim = pybop.XNES(problem, options=options)
optim.set_population_size(15)
result = optim.run()

print(f"| Model: {model.name} | Results: {result.best_inputs} |")
pybop.plot.problem(result.problem, inputs=result.best_inputs, title=model.name)

# best_inputs = {'Negative particle diffusivity [m2.s-1]': np.float64(2.707628901312098e-08), 'Negative electrode exchange-current density [A.m-2]': np.float64(23.36842035601129), 'Positive particle diffusivity [m2.s-1]': np.float64(7.49814418902937e-08), 'Positive electrode exchange-current density [A.m-2]': np.float64(0.4749398343750321)}
