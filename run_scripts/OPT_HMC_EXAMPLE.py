import os
import timeit

from sympy.printing.pretty.pretty_symbology import line_width

os.environ['JAX_PLATFORM_NAME'] = 'cpu'
import pybamm
import pybamm as pb
import numpy as np
import jax
import jax.numpy as jnp


from src.embedded_gp import OPTUNA_HMC_GPS

import pandas as pd
import matplotlib
# matplotlib.use()
import matplotlib.pyplot as plt
from FoKL import getKernels
import warnings
from Create_Params import ParamUpdate
from src.Data.Processed_Data.Samsung_25R_Parameters import samsung_25r_params

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

# param1 = from_paper_params.get_samsung_25r_parameters_v7()
param1 = pybamm.ParameterValues('Mohtat2020')
# Define the phis (basis functions) used

def normalize_inputs(inputs, min, max):
    normalized = (inputs - min) / (max - min)
    return normalized

testing = pd.read_csv('/Users/derekslack/Pybamm-Embedded-GP-live/src/Data/Processed_Data/EPSCoR_CP - 013.csv')

i_begin = 823
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

# Create object for each individual GP
# GPj0p = OPTUNA_HMC_GPS.GP()
# GPj0n = OPTUNA_HMC_GPS.GP()
GPUP = OPTUNA_HMC_GPS.GP()
GPUN = OPTUNA_HMC_GPS.GP()
GPUe = OPTUNA_HMC_GPS.GP()


# Create of model and define the number of GP's in it
model = OPTUNA_HMC_GPS.Embedded_GP_Model(GPUP, GPUN, GPUe)

# Define appropriate parameters to model
# model.inputs = np.transpose(np.vstack([inputs_norm, inputs_norm]))
model.inputs = np.transpose(np.array([t]))
model.phis = phis
model.data = np.transpose(Volt)

current_interpolant = pybamm.Interpolant(t, Amps, pybamm.t)#, interpolator="JAX")  # , _num_derivatives=0)
param1["Current function [A]"] = current_interpolant


# samples = np.loadtxt("/home/WVU-AD/ds0172/Desktop/PyBamm-Embedded-GP-main/src-non-jax/src/Data/samples_j0_10_14.csv")
# beta0 = np.array([2.2,0,0, 0.4,0,0,-32,0,0, 0.4])

# DP = np.mean(samples[:,4][-200:])
DP = -35
# beta0 = np.array([DP, 0, 0,DP,0,0, 2.2,0,0,0.4,0,0, 0.4 ])
# beta0 = np.array([DP, DP, 2.2, 0.4, 0.4])
# beta0 = np.array(samples[-1,:])
# beta0 = [[np.log(11),0], [np.log(14),0], [np.log(-24.04),0], [np.log(-31.04),0]]
#

beta0 = [[np.log(1e-6),0], [np.log(1e-6),0],[-24,0],[-32,0]]
#[I 2026-01-14 17:15:36,016] Trial 91 finished with value: 0.000559875001732329 and parameters: {'Beta00': 0.6668933738331815, 'Beta01': -20.89446095794654, 'Beta02': -25.035551389790516, 'Beta10': 2.82829197497284, 'Beta11': 0.28923765640049015, 'Beta12': -17.581813898668265, 'Beta20': -11.661717831866273, 'Beta21': -11.248503672366766, 'Beta22': 0.10651268580057194, 'Beta30': -18.124407967099472, 'Beta31': -14.742907441717046, 'Beta32': 0.28138889792779964}. Best is trial 91 with value: 0.000559875001732329.
# Time to solve all: 3.3091935419943184
# beta0 = [[-24,-29]]
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
GP_dict_j0_pos = {'Name':"Positive electrode reference exchange-current density [A.m-2(m3.mol)1.5]", 'arg_inds':[], 'div_arg':[[0,1]], 'exp':True, 'inputs_function':1, 'B0':(-16,-4), 'div_const':None}
GP_dict_j0_neg = {'Name':"Negative electrode reference exchange-current density [A.m-2(m3.mol)1.5]", 'arg_inds':[], 'div_arg':[[0,1]], 'exp':True, 'inputs_function':1, 'B0':(-16,-4), 'div_const':None}
GP_dict_D_pos = {'Name':'Positive particle diffusivity [m2.s-1]', 'arg_inds':[0],  'exp':True, 'inputs_function':1,'div_arg':None, 'B0':(-35,-9), 'div_const':None}
GP_dict_D_neg = {'Name':'Negative particle diffusivity [m2.s-1]', 'arg_inds':[0],  'exp':True, 'inputs_function':1, 'div_arg':None, 'B0':(-33,-12), 'div_const':None}
# GP_dict_D_e = {'Name':'Electrolyte diffusivity [m2.s-1]', 'arg_inds':[0],  'exp':True, 'inputs_function':1, 'div_arg':None, 'B0':(-33,-12), 'div_const':[2000]}



GP_dict_list = [GP_dict_j0_pos, GP_dict_j0_neg, GP_dict_D_pos, GP_dict_D_neg]

# GP_dict_list = [GP_dict_D_neg]
PSO_options = {'n_particles':  6, 'n_iterations':150}
i=0
# input_dict_init = Param_Updater._to_input_dict(beta0)

batmodel = pybamm.lithium_ion.SPMe()
rhs = batmodel.rhs
l = []
for k in rhs.keys():
    l.append(k)
l_sub = []
for k in batmodel.submodels.keys():
    l_sub.append(k)

def process_tree(symbol: pybamm.Symbol, parameter_name, new_input):
    if isinstance(symbol, pybamm.FunctionParameter) and symbol.name == parameter_name:
        original_inputs = symbol.input_names
        original_children = list(symbol.children)
        extended_inputs = dict(zip(original_inputs, original_children))
        extended_inputs.update(new_input)
        return pybamm.FunctionParameter(parameter_name, extended_inputs)

    else:
        new_children = [process_tree(child, parameter_name, new_input) for child in symbol.children]
        return symbol.create_copy(new_children)

I_max = pybamm.Parameter("Nominal cell capacity [A.h]")

Iin = {'Current [A]':(pybamm.FunctionParameter(
            "Current function [A]", {"Time [s]": pybamm.t}
        )-np.min(Amps))/(np.max(Amps)-np.min(Amps))}
def process_full_sol(battery_model, param_name, function):
    for i in range(len(l)):
        B = process_tree(rhs[l[i]],param_name,function)
        battery_model.rhs[l[i]] = B

    for i in range(len(l_sub)):
        if battery_model.submodels[l_sub[i]].rhs:
            for k in battery_model.submodels[l_sub[i]].rhs.keys():
                B = process_tree(battery_model.submodels[l_sub[i]].rhs[k],param_name, function)
                battery_model.submodels[l_sub[i]].rhs[k] = B
    for bc_key in battery_model.boundary_conditions:
        B_new = process_tree(battery_model.boundary_conditions[bc_key]['right'][0],param_name, function)
        battery_model.boundary_conditions[bc_key]['right'] = (B_new, 'Neumann')

    return battery_model
batmodel2 = pybamm.lithium_ion.SPMe()
# batmodel = process_full_sol(batmodel, 'Electrolyte diffusivity [m2.s-1]',Iin)
# batmodel = process_full_sol(batmodel, 'Negative particle diffusivity [m2.s-1]',Iin)
# batmodel = process_full_sol(batmodel, 'Positive particle diffusivity [m2.s-1]',Iin)

# Set-up the model
# geometry = batmodel.default_geometry
# param1.process_geometry(geometry)
# param1.process_model(batmodel)
# var = pybamm.standard_spatial_vars
# var_pts = {var.x_n: 20, var.x_s: 20, var.x_p: 20, var.r_n: 10, var.r_p: 10}
# mesh = pybamm.Mesh(geometry, batmodel.default_submesh_types, var_pts)
# disc = pybamm.Discretisation(mesh, batmodel.default_spatial_methods)
# disc.process_model(batmodel)
output_variables = ["Voltage [V]", "Positive particle effective diffusivity [m2.s-1]"]
# param1["Negative electrode thickness [m]"] = 5.55148335e-04
# param1["Positive electrode thickness [m]"]  = 6.50653019e-05
# param1.update({'Beta00': 1.2173296437832817, 'Beta01': -9.245791064580327, 'Beta10': -0.49194874119979054, 'Beta11': 3.874952819403981, 'Beta20': -31.544335922734724, 'Beta21': 1.7556420666704216, 'Beta30': -15.616266049850648, 'Beta31': 30.477701755503592, 'Beta40': -14.282171660505345, 'Beta41': -20.00183111749591})
# [I 2026-04-03 16:25:21,589] Trial 99 finished with value: 1.480884346961134e-05 and parameters: {'Beta00': -32.25072597763535, 'Beta01': -3.348199365435015, 'Beta10': -23.619205679498524, 'Beta11': -20.93819981216272, 'Beta02': 13.340974421514176, 'Beta03': -20.379364577720317, 'Beta12': -31.96480606605158, 'Beta13': 20.963098897605313}. Best is trial 99 with value: 1.480884346961134e-05.


# param2 = pybamm.ParameterValues('Chen2020')
samsung_25r_updates = samsung_25r_params()

param1.update(samsung_25r_updates, check_already_exists=False)
param1["Current function [A]"] = current_interpolant
# param1["Negative electrode thickness [m]"] = 7.95011190e-05
# param1["Positive electrode thickness [m]"]  = 2.68659159e-05
# param1["Nominal cell capacity [A.h]"] = 2.32532356e+00
# param1["Maximum concentration in positive electrode [mol.m-3]"] =  6.80316296e+04
# param1["Maximum concentration in negative electrode [mol.m-3]"] = 3.23761910e+04



# param1["Initial concentration in negative electrode [mol.m-3]"] = 1.56431275e+04
# param1["Initial concentration in positive electrode [mol.m-3]"] =  1.82098527e+04

# best_params = {'Beta00': 2.3780408310015293, 'Beta01': 4.792779231694654, 'Beta10': 0.887001250585174, 'Beta11': 19.09087887202484, 'Beta20': -27.535424984248685, 'Beta21': 25.743803349957393, 'Beta30': -21.11562285975143, 'Beta31': 20.280000767567966, 'Beta40': -22.203990091520758, 'Beta41': 10.975168387303858, 'Beta02': -14.376561708956425, 'Beta03': -12.217815878115093, 'Beta12': 6.873879312718643, 'Beta13': 34.89743581745051, 'Beta22': -7.609235952846025, 'Beta23': 11.593698323508024, 'Beta32': -32.34356924316246, 'Beta33': 26.7010691027816, 'Beta42': -32.730537168308445, 'Beta43': 20.564377872899925}
solver = pybamm.IDAKLUSolver(atol=1e-4, rtol=1e-2, output_variables=output_variables, options={'num_threads':os.cpu_count()-2})
# sim = pybamm.Simulation(batmodel, parameter_values=param1, solver=solver)
# sol1 = sim.solve(t,t_interp=t)
# #
# sim2 = pybamm.Simulation(batmodel2, parameter_values=param2, solver=solver)
# sol2 = sim2.solve(t,t_interp=t,initial_soc=1)
# # plt.plot(t,sol1['Voltage [V]'].entries,label='PyBamm w/ GPs',linestyle=':')
# plt.plot(t,sol2['Voltage [V]'].entries,label='PyBamm Default',linestyle='--')
# plt.plot(t,Volt,label='Data')
# plt.legend()
# plt.xlabel('Time [s]')
# plt.ylabel('Voltage [V]')

def mse(data, Vsim):
    return np.sum((data-Vsim)**2)/len(data)

# res = mse(Volt, sol1['Voltage [V]'].entries)
# plt.title(f'mse={res}')
# plt.show()
model.t = t


model.solution = None
# model.pybamm_default_data = sol2['Voltage [V]'].entries

damtx = [ np.array([[1.],
                    ]).astype(int),
          np.array([[1.],
                    ]).astype(int),
          np.array([[1.],
                    ]).astype(int),
          np.array([[1.],
                    ]).astype(int)]

# [[0. 1.]
#  [1. 0.]
#  [0. 2.]
#  [2. 0.]]
# -9103.61530756552 and parameters: {'Beta00': -30.372784184576552, 'Beta01': 14.117244051266038, 'Beta10': -32.46033837079, 'Beta11': -8.378155789624923, 'Beta20': -12.719067431093192, 'Beta21': 27.880580499255856, 'Beta02': 18.67986914419243, 'Beta03': 9.484670163548538, 'Beta04': -16.544623087653978, 'Beta05': -8.58467847307578, 'Beta06': 9.113104560986619, 'Beta12': -18.283329099389, 'Beta13': -19.483423256841462, 'Beta14': -26.96219079145742, 'Beta15': 2.433016369250007, 'Beta16': 19.46263850557794, 'Beta22': -4.0295422673206485, 'Beta23': 18.334934263147918, 'Beta24': 22.88602474412472, 'Beta25': -0.17032601082345966, 'Beta26': 26.294068331471863}
# best = {'Beta00': -30.372784184576552, 'Beta01': 14.117244051266038, 'Beta10': -32.46033837079, 'Beta11': -8.378155789624923, 'Beta20': -12.719067431093192, 'Beta21': 27.880580499255856, 'Beta02': 18.67986914419243, 'Beta03': 9.484670163548538, 'Beta04': -16.544623087653978, 'Beta05': -8.58467847307578, 'Beta06': 9.113104560986619, 'Beta12': -18.283329099389, 'Beta13': -19.483423256841462, 'Beta14': -26.96219079145742, 'Beta15': 2.433016369250007, 'Beta16': 19.46263850557794, 'Beta22': -4.0295422673206485, 'Beta23': 18.334934263147918, 'Beta24': 22.88602474412472, 'Beta25': -0.17032601082345966, 'Beta26': 26.294068331471863}
best = {'Beta00': 4.368844168378062, 'Beta01': -18.069578497342974, 'Beta10': -1.6258703200476665, 'Beta11': 1.8134287122031123, 'Beta20': -19.544229844062063, 'Beta21': 16.03612699039317, 'Beta30': -24.89521126732811, 'Beta31': 12.43977490513863}

def create_beta_inputs(len_mtx, GP_num):
    betas_function = []
    betas_keys = {}
    for i in range(len_mtx):
        key_str = 'Beta' + str(GP_num) + str(i)
        betas_function.append(pybamm.InputParameter(key_str))
        betas_keys.update({key_str: 0})
    return betas_function, betas_keys

P = ParamUpdate(param1,[])

for i, GP in enumerate(GP_dict_list):
    betas_function, betas_keys = create_beta_inputs(len(damtx[i])+1, i)
    P.add_function(GP['Name'], damtx[i], GP['arg_inds'], betas_function, exp=GP['exp'], div_arg=GP['div_arg'], div_const=GP['div_const'])

P_validate = P.get_param()

orig_neg_j0_func = P_validate["Negative electrode exchange-current density [A.m-2]"]
orig_pos_j0_func = P_validate["Positive electrode exchange-current density [A.m-2]"]

m_ref_neg = 1.061e-6
m_ref_pos = 4.824e-06


def dynamic_neg_j0_wrapper(c_e, c_s_surf, c_s_max, T):
    # Evaluate original function to get the actual pybamm.Symbol tree
    orig_symbol_tree = orig_neg_j0_func(c_e, c_s_surf, c_s_max, T)

    # Grab the dynamic reference parameter
    dynamic_ref = P_validate["Negative electrode reference exchange-current density [A.m-2(m3.mol)1.5]"](c_s_surf,
                                                                                                         c_s_max)

    # Perform math on the evaluated SYMBOLS, bypassing the error
    return dynamic_ref * (orig_symbol_tree / m_ref_neg)


def dynamic_pos_j0_wrapper(c_e, c_s_surf, c_s_max, T):
    # Evaluate original function to get the actual pybamm.Symbol tree
    orig_symbol_tree = orig_pos_j0_func(c_e, c_s_surf, c_s_max, T)

    # Grab the dynamic reference parameter
    dynamic_ref = P_validate[
        "Positive electrode reference exchange-current density [A.m-2(m3.mol)1.5]"](c_s_surf, c_s_max)
    # Perform math on the evaluated SYMBOLS, bypassing the error
    return dynamic_ref * (orig_symbol_tree / m_ref_pos)


# -------------------------------------------------------------------------
# 3. Inject the wrappers back into the dictionary
# -------------------------------------------------------------------------
P_validate["Negative electrode exchange-current density [A.m-2]"] = dynamic_neg_j0_wrapper
P_validate["Positive electrode exchange-current density [A.m-2]"] = dynamic_pos_j0_wrapper


best = {'Beta00': -9.92724638318497, 'Beta01': 3.2738806098683697, 'Beta10': -14.52530153425246, 'Beta11': 1.5797308324261015, 'Beta20': -15.512450048980867, 'Beta21': 1.8902924768972538, 'Beta30': -18.784412642194486, 'Beta31': -0.7178834034852439}

sim = pybamm.Simulation(batmodel, parameter_values=P_validate, solver=solver)
sol1 = sim.solve(t,t_interp=t, inputs=best)
plt.plot(t, Volt, 'r-',label='Data')
plt.plot(t[0:len(sol1['Voltage [V]'].entries)],sol1['Voltage [V]'].entries,'b:' ,label='PyBaMM w/ GPs', linewidth=2)


rmse = np.sqrt(np.sum((sol1['Voltage [V]'].entries - Volt)**2)/len(Volt))

plt.text(200,3.7, f'RMSE = {rmse:.3f}',
         fontsize=12, color='black',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

plt.title('Training on Power Discharge')
# plt.title('Validation on CC Discharge')
plt.xlabel('Time [s]')
plt.ylabel('Voltage [V]')
plt.legend()
plt.show()
betas, mtx, evs = model.full_routine(t, 1, param1, GP_dict_list, batmodel, solver, PSO_options, way3 = 0, init_betas = beta0)


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

# {'Beta00': -16.706699716257287, 'Beta01': 1.985789257860028, 'Beta10': -17.84407945392173, 'Beta11': -20.77156803640722, 'Beta20': -14.095366762698887, 'Beta21': -31.546115051998655, 'Beta02': 28.118082246360874, 'Beta03': -21.880533117252746, 'Beta04': -19.202501047514072, 'Beta12': 15.618006279636127, 'Beta13': -5.378903308480867, 'Beta14': 8.51790601481931, 'Beta22': -25.8379929988852, 'Beta23': -9.056361165698595, 'Beta24': -28.8900996325147}


# [[1.]]
# {'Beta00': -20.048846936123304, 'Beta01': 29.316550682811524, 'Beta10': -25.06336596985507, 'Beta11': 21.56215085263448, 'Beta20': -14.463512065721662, 'Beta21': -28.526718510672143}


# 7.271519953682185e-05 and parameters: {'Beta00': 2.3780408310015293, 'Beta01': 4.792779231694654, 'Beta10': 0.887001250585174, 'Beta11': 19.09087887202484, 'Beta20': -27.535424984248685, 'Beta21': 25.743803349957393, 'Beta30': -21.11562285975143, 'Beta31': 20.280000767567966, 'Beta40': -22.203990091520758, 'Beta41': 10.975168387303858, 'Beta02': -14.376561708956425, 'Beta03': -12.217815878115093, 'Beta12': 6.873879312718643, 'Beta13': 34.89743581745051, 'Beta22': -7.609235952846025, 'Beta23': 11.593698323508024, 'Beta32': -32.34356924316246, 'Beta33': 26.7010691027816, 'Beta42': -32.730537168308445, 'Beta43': 20.564377872899925}. Best is trial 437 with value: 7.271519953682185e-05.