from pybamm import Simulation

from src.embedded_gp import post_process
import pandas as pd
from src.Data import from_paper_params
import pybamm
import numpy as np
from src.embedded_gp import Create_Params

import matplotlib.pyplot as plt

samples = '/home/WVU-AD/ds0172/Desktop/PyBamm-Embedded-GP-main/Embeddded-GPs/src/Data/samples_j0_10_30.csv'
BIC = '/home/WVU-AD/ds0172/Desktop/PyBamm-Embedded-GP-main/Embeddded-GPs/results/BIC_10_30.csv'

testing = pd.read_csv('/home/WVU-AD/ds0172/Desktop/PyBamm-Embedded-GP-main/Embeddded-GPs/src/Data/modEpscorData.csv')

i_end = 4365

Volt = testing['Volts'].to_numpy()[820:i_end]

PP = post_process.PostProcess(samples, BIC, Volt)
PP.plot_NLL()
betas = PP.average_samples(100)

param1 = from_paper_params.get_samsung_25r_parameters_v7()

# Define the phis (basis functions) used

def normalize_inputs(inputs, min, max):
    normalized = (inputs - min) / (max - min)
    return normalized

Amps = testing['Amps'].to_numpy()[820:i_end]
Amps[2464:] = -Amps[2464:]
time = testing['TestTime'].to_numpy()[820:i_end]
Temp = testing['Temp 2'].to_numpy()[820:i_end]

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

def current_func(time_I):
    if time_I <= 819-220:
        I = 0.
    elif 819-220 < time_I <= 1720-220:
        I = 1.25
    elif 1720-220 < time_I <= 1780-220:
        I = 0.
    elif 1780-220 < time_I <= 2682-220:
        I = 2.5

    return I


# Create GP function parameters
beta_f = list(betas[:-1].reshape((4,2)))
beta_f.append(0.12**2)
Param_Updater = Create_Params.ParamUpdate(param1,beta_f, [[[1]],[[1]],[[1]],[[1]]])
Param_Updater.add_function('Positive electrode exchange-current density [A.m-2]', [],0, div_arg=[[1,2]], exp=True)
Param_Updater.add_function('Negative electrode exchange-current density [A.m-2]', [],1, div_arg=[[1,2]], exp=True)
Param_Updater.add_function('Positive particle diffusivity [m2.s-1]', [0],2, exp=True)
Param_Updater.add_function('Negative particle diffusivity [m2.s-1]', [0],3, exp=True)
param = Param_Updater.get_param()

current_interpolant = pybamm.Interpolant(t, Amps, pybamm.t)#, interpolator="JAX")  # , _num_derivatives=0)
param["Current function [A]"] = current_interpolant

output_variables = ["Voltage [V]"]

model = pybamm.lithium_ion.SPMe()
solver = pybamm.IDAKLUSolver(atol=1e-2, output_variables=output_variables)
sim = pybamm.Simulation(model,parameter_values=param)
input_dict = Param_Updater._to_input_dict(betas, flat=True)
sol = sim.solve(t,t_interp=t, inputs=input_dict)
V = sol['Voltage [V]'].entries
plt.plot(t,Volt, label='Data')
plt.plot(t,V, label='Embdedded Pybamm')
plt.legend()
plt.ylabel('Voltage [V]')
plt.show()
h=1