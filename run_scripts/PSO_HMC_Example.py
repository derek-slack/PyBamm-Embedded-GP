import os
import timeit

from src.embedded_gp.post_process import PostProcess

os.environ['JAX_PLATFORM_NAME'] = 'cpu'
import pybamm
import pybamm as pb
import numpy as np
import jax
import jax.numpy as jnp


from src.embedded_gp import PSO_HMC_GPs

import pandas as pd
import matplotlib.pyplot as plt
from FoKL import getKernels
import warnings

from pathlib import Path

# csv_path = Path("./src/Data/modEpscorData.csv")

phis = getKernels.sp500()
warnings.filterwarnings("ignore")

k = "symmetric Butler-Volmer"
pb.set_logging_level("NOTICE")



save_folder = "figures"
os.makedirs(save_folder, exist_ok=True)

# Set Parameter values

from src.Data import from_paper_params
# from from_paper_params import get_samsung_25R_parameters

param1 = from_paper_params.get_samsung_25r_parameters_v7()

# Define the phis (basis functions) used

def normalize_inputs(inputs, min, max):
    normalized = (inputs - min) / (max - min)
    return normalized

testing = pd.read_csv('/Users/derekslack/Pybamm-Embedded-GP-live/src/Data/modEpscorData.csv')

i_end = 4365

Volt = testing['Volts'].to_numpy()[820:i_end]
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

def current_func(time):
    I = 1.25 * (time <= 900) + 0 * (time <= 960) + 2.5 * (time <= 1860) + 0 * (time <= 2460) + -1.25 * (time > 2460)
    return I


# Create object for each individual GP
GPj0p = PSO_HMC_GPs.GP()
GPj0n = PSO_HMC_GPs.GP()
GPUP = PSO_HMC_GPs.GP()
GPUN = PSO_HMC_GPs.GP()


# Create of model and define the number of GP's in it
model = PSO_HMC_GPs.Embedded_GP_Model(GPj0p, GPj0n, GPUP, GPUN)

# Define appropriate parameters to model
# model.inputs = np.transpose(np.vstack([inputs_norm, inputs_norm]))
model.inputs = np.transpose(np.array([t]))
model.phis = phis
model.data = np.transpose(Volt)

#
plt.plot(t,Amps)
plt.show()

current_interpolant = pybamm.Interpolant(t, Amps, pybamm.t)#, interpolator="JAX")  # , _num_derivatives=0)
param1["Current function [A]"] = current_interpolant


# samples = np.loadtxt("/home/WVU-AD/ds0172/Desktop/PyBamm-Embedded-GP-main/src-non-jax/src/Data/samples_j0_10_14.csv")
# beta0 = np.array([2.2,0,0, 0.4,0,0,-32,0,0, 0.4])

# DP = np.mean(samples[:,4][-200:])
DP = -35
# beta0 = np.array([DP, 0, 0,DP,0,0, 2.2,0,0,0.4,0,0, 0.4 ])
# beta0 = np.array([DP, DP, 2.2, 0.4, 0.4])
# beta0 = np.array(samples[-1,:])
# beta0 = [[np.log(11),0], [np.log(14),0], [-24.04,0], [-31.04,0]]
beta0 = [[np.log(14),0],[-31.04,0]]
num_betas=len(beta0)
#
from src.embedded_gp import Create_Params

def process_tree(symbol: pybamm.Symbol):
    if isinstance(symbol, pybamm.Parameter) and symbol.name == "My Parameter":
        return symbol
    else:
        new_children = [process_tree(child) for child in symbol.children]
        return symbol.create_copy(new_children)

# new_symbol = process_tree(pybamm.Symbol('Positive electrode exchange-current density [A.m-2]', children=[pybamm.Symbol('Current [A]')]))

# Param_Updater = Create_Params.ParamUpdate(param1,beta0, [[[1]],[[1]],[[1]],[[1]]])
# {'Beta00':(-5,2),'Beta01':(-25,25),'Beta10':(-5,2),'Beta11':(-25,25),'Beta20':(-35,0),'Beta21':(-25,25),'Beta30':(-35,0),'Beta31':(-25,25)}
# GP_dict_j0_pos = {'Name':'Positive electrode exchange-current density [A.m-2]', 'arg_inds':[], 'div_arg':[[1,2]], 'exp':True, 'inputs_function':1, 'B0':(-5,3)}
GP_dict_j0_neg = {'Name':'Negative electrode exchange-current density [A.m-2]', 'arg_inds':[], 'div_arg':[[1,2]], 'exp':True, 'inputs_function':1, 'B0':(-5,3)}
# GP_dict_D_pos = {'Name':'Positive particle diffusivity [m2.s-1]', 'arg_inds':[0],  'exp':True, 'inputs_function':1,'div_arg':None, 'B0':(-35,-9)}
GP_dict_D_neg = {'Name':'Negative particle diffusivity [m2.s-1]', 'arg_inds':[0],  'exp':True, 'inputs_function':1, 'div_arg':None, 'B0':(-35,-9)}

# GP_dict_list = [GP_dict_j0_pos, GP_dict_j0_neg, GP_dict_D_pos, GP_dict_D_neg]
GP_dict_list = [GP_dict_j0_neg, GP_dict_D_neg]
PSO_options = {'n_particles':  48, 'n_iterations':50}
i=0
# input_dict_init = Param_Updater._to_input_dict(beta0)

batmodel = pybamm.lithium_ion.SPMe()


# Set-up the model
# geometry = batmodel.default_geometry
# param1.process_geometry(geometry)
# param1.process_model(batmodel)
# var = pybamm.standard_spatial_vars
# var_pts = {var.x_n: 20, var.x_s: 20, var.x_p: 20, var.r_n: 10, var.r_p: 10}
# mesh = pybamm.Mesh(geometry, batmodel.default_submesh_types, var_pts)
# disc = pybamm.Discretisation(mesh, batmodel.default_spatial_methods)
# disc.process_model(batmodel)
output_variables = ["Voltage [V]"]


solver = pybamm.IDAKLUSolver(atol=1e-4, rtol=1e-2, output_variables=output_variables, options={'num_threads':os.cpu_count()})
sim = pybamm.Simulation(batmodel, parameter_values=param1, solver=solver)

model.t = t


model.solution = None
betas, mtx, evs = model.full_routine(t, 5, param1, GP_dict_list, batmodel, solver, PSO_options, way3 = 0, init_betas = beta0)


def equation(input_dict):
    t_start = timeit.default_timer()
    sol = sim.solve(t, t_interp = t, inputs=input_dict)
    t_end = timeit.default_timer() - t_start
    print(f'Time to solve all: {t_end}')
    V = []
    if isinstance(sol, list):
        for s in sol:
            V.append(s['Voltage [V]'].entries)
    else:
        V = sol['Voltage [V]'].entries
    return V

