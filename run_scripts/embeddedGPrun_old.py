import pybamm
import pybamm as pb
import numpy as np
import jax.numpy as jnp
import timeit
import jax

from src.embedded_gp import Experimental_Embedded_GPs

from src.embedded_gp.new_eval import evaluate_pybamm
import pandas as pd
import matplotlib.pyplot as plt
from FoKL import getKernels
import warnings
import os

phis = getKernels.sp500()
warnings.filterwarnings("ignore")

k = "symmetric Butler-Volmer"
pb.set_logging_level("NOTICE")
batmodel = pybamm.lithium_ion.SPMe()
# batmodel.events = []
# batmodel.convert_to_format = 'jax'

save_folder = "figures"
os.makedirs(save_folder, exist_ok=True)

# remove events (not supported in jax)
batgeometry = batmodel.default_geometry

# Set Parameter values
# param1 = pb.ParameterValues("Mohtat2020")

from src.Data import from_paper_params
# from from_paper_params import get_samsung_25R_parameters

param1 = from_paper_params.get_samsung_25r_parameters_v7()

# Define the phis (basis functions) used


def normalize_inputs(inputs, min, max):
    normalized = (inputs - min) / (max - min)
    return normalized


# C = pd.read_csv('/home/WVU-AD/ds0172/Desktop/PyBamm-Embedded-GP-main/src-non-jax/src/Data/ChargeCycle.csv', header=None)
# D = pd.read_csv('/home/WVU-AD/ds0172/Desktop/PyBamm-Embedded-GP-main/src-non-jax/src/Data/DischargeCycle.csv', header=None)
testing = pd.read_csv('/home/WVU-AD/ds0172/Desktop/PyBamm-Embedded-GP-main/src-non-jax/src/Data/modEpscorData.csv')
# full_charge_discharge = pd.read_csv('/home/WVU-AD/ds0172/Desktop/PyBamm-Embedded-GP-main/src-non-jax/src/Data/EPSCoR_CC - 024.csv.csv')
from src.embedded_gp.create_OCV import create_OCV_full_cell, V_to_pos_half, V_to_neg_half



# filename = '/home/WVU-AD/ds0172/Desktop/PyBamm-Embedded-GP-main/src-non-jax/src/Data/EPSCoR_Char_B4 - 024.csv'
# i_D_start = 6419
# i_D_end = 51115
#
# SOC, Volt_ocv = create_OCV_full_cell(filename, i_D_start, i_D_end, 2.500*3600)
# neg_ocp = V_to_neg_half(SOC, Volt_ocv)
#
# param1['Positive electrode OCP [V]'] = neg_ocp
# # Volt_charge = full_charge_discharge['Volts'].to_numpy()
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

# t = np.linspace(0, len(time), len(time))
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

# Create object for each individual GP
GPj0p = Experimental_Embedded_GPs.GP()
GPj0n = Experimental_Embedded_GPs.GP()
GPUP = Experimental_Embedded_GPs.GP()
GPUN = Experimental_Embedded_GPs.GP()
GPOCP = Experimental_Embedded_GPs.GP()

# Create of model and define the number of GP's in it
model = Experimental_Embedded_GPs.Embedded_GP_Model(GPj0p, GPj0n, GPUP, GPUN)

# Define appropriate parameters to model
# model.inputs = np.transpose(np.vstack([inputs_norm, inputs_norm]))
model.inputs = np.transpose(np.array([t]))
model.phis = phis
model.data = np.transpose(Volt)


from src.embedded_gp.create_OCV import create_OCV_full_cell, V_to_pos_half
import matplotlib.pyplot as plt
import pybamm as pb
import numpy as np

# param = pb.ParameterValues("Mohtat2020")

# filename = '/home/WVU-AD/ds0172/Desktop/PyBamm-Embedded-GP-main/src-non-jax/src/Data/EPSCoR_Char_B4 - 024.csv'
# i_D_start = 6419
# i_D_end = 51115
#
# SOC, Volt_ocv = create_OCV_full_cell(filename, i_D_start, i_D_end, 2.500*3600)
# Volt_pos_half = V_to_pos_half(SOC, Volt_ocv)
#
# stos = np.linspace(0,1,1000)
#
# param_pocp = param['Negative electrode OCP [V]'](stos)
# param_nocp = param['Positive electrode OCP [V]'](stos[::-1])
#
# param_ocp = param_nocp - param_pocp
# vp = []
# for sto in stos:
#     vp.append(Volt_pos_half(sto).evaluate()[0][0])
#
# data_ocp = vp - param_pocp
#
# plt.plot(stos, data_ocp, label='data pos')
# plt.plot(stos, param_ocp, label='Positive')
# plt.plot(stos, param_nocp, label = 'negative')
# plt.legend()
# plt.show()

from src.LFP import LFP_OCP

# param1['Positive electrode OCP [V]'] = LFP_OCP


# minI = -4.0304
# maxI = 1.5145
#
# polyfit = [-4186.54559829646, 18363.2314210941, -33927.0311238327,
#            34177.2778419589, -20221.9632429130, 6960.48498715931,
#            -1220.79587333825, 33.6148046197656, 24.1935530311832,
#            -4.65707497961982, 4.31012644510537]
#
#
# def polyfit_ocv(sto):
#     X = 0
#     l = len(polyfit)
#     for i in range(l):
#         X += polyfit[i] * sto**(l-i-1)
#     return X
#
# def polyfit_ocv_neg(sto):
#     X = 0
#     sto = 1- sto
#     l = len(polyfit)
#     for i in range(l):
#         X += polyfit[i] * sto**(l-i-1)
#     return X * - 0.5


# soc_init = np.linspace(0,1,50)
# ocv_p = polyfit_ocv(soc_init)
# ocv_n = polyfit_ocv_neg(soc_init)
# on = graphite_ocp_PeymanMPM(soc_init)
# op = NMC_ocp_PeymanMPM(soc_init)
# ogocv = op - on
# ocvs = ocv_p - on
#
# plt.plot(ogocv, label = "Mohtat")
# dOCV = []
# for i in range(1,50):
#     dOCV.append(ocvs[i] - ocvs[i-1])
# # plt.plot(dOCV)
# plt.plot(ocvs, label="My fit")
# plt.legend()
# plt.show()
# # for i in range(len(soc_init)):
#     ocv_init = polyfit_ocv_neg(soc_init[i]) - polyfit_ocv_neg(soc_init[i])
#     print(f"Expected Initial Voltage @ soc {soc_init[i]: .3f}= [{ocv_init:.3f}]")
#
# experiment = pybamm.Experiment(
#     [
#         "Discharge at 4A for 10 seconds", "Rest at 0 A for 10 seconds"
#     ] * 168
# )
plt.plot(t,Amps)
plt.show()

current_interpolant = pybamm.Interpolant(t, Amps, pybamm.t)#, interpolator="JAX")  # , _num_derivatives=0)
param1["Current function [A]"] = current_interpolant
# param1["Positive electrode OCP [V]"] = polyfit_ocv
# param1["Negative electrode porosity"] = 0.5
# param1["Negative electrode OCP [V]"] = 0
param1["Open-circuit voltage at 0% SOC [V]"] = 0
param1["Open-circuit voltage at 100% SOC [V]"] = 4.2
# param1["Initial concentration in positive electrode [mol.m-3]"] = 25000

#
# param1["Lower voltage cut-off [V]"] = 2.5
# param1["Upper voltage cut-off [V]"] = 4.2
# # param1["Contact resistance [Ohm]"] = 0.156
# param1["Open-circuit voltage at 0% SOC [V]"] = 2.5
# param1["Open-circuit voltage at 100% SOC [V]"] = Volt[0]


solver = pybamm.IDAKLUSolver(atol=1e-2, rtol=1e-2)


    # input_dict.update({beta_str: bm[0][i]})
# param1.update({"mtx": mtx[0:8]},check_already_exists=False)


#
# param1["Positive electrode exchange-current density [A.m-2]"] = 1
# param1["Negative electrode exchange-curre-8.346150877549220240e+03nt density [A.m-2]"] = 2
# param1["Positive particle diffusivity [m2.s-1]"] = 12e-4
# param1["Negative particle diffusivity [m2.s-1]"] = 2e-5
# param1["Current function [A]"] = current_interpolant

#
# # param1["Initial concentration in positive electrode [mol.m-3]"] = 47513.0 * 0.58
sim = pybamm.Simulation(batmodel,parameter_values=param1, solver=solver)
solution = sim.solve(t, initial_soc=1, t_interp=t)
solution.plot()
# solution = sim.solve(t, initial_soc=1, t_interp=t)
# Vpbi = solution['Voltage [V]'].entries
# plt.plot(t, Vpbi)
# plt.plot(t, Volt)
# plt.show()

# param2 = pybamm.ParameterValues("Mohtat2020")
# stos = np.linspace(0,1,1000)
# lfp = []
# default = []
# for sto in stos:
#     lfp.append(param1['Positive electrode OCP [V]'](pybamm.Scalar(sto)).evaluate())
#     default.append(param2['Positive electrode OCP [V]'](sto))
#
# plt.plot(stos, lfp, label = 'LFP')
# plt.plot(stos, default, label = 'Default')
# plt.legend()
# plt.plot()

model.solution = None
model.tt = 0
c = solution['Negative particle surface concentration [mol.m-3]'].entries[0]
j0p_i = solution['Positive electrode exchange current density [A.m-2]'].entries[0]
j0n_i = solution['Negative electrode exchange current density [A.m-2]'].entries[0]
Dp_i = np.mean(solution["Positive particle effective diffusivity [m2.s-1]"].entries[0][0])
Dn_i = np.mean(solution["Negative particle effective diffusivity [m2.s-1]"].entries[0][0])

# beta0 = np.array([2.2,0,0,0.47,0,0, np.log(Dp_i),0, 0,np.log(Dn_i),0, 0,0.4])
# beta0 = np.array([2.2,1e-5,1e-5, 0.47,1e-5,1e-5, 0.4])
# beta0 = np.array([2.2,0.47,np.log(Dp_i),np.log(Dn_i),0.4])
# beta0 = np.array([2.2,0, 0.4,0, -32, -32, 0.4])#, -3.069504657677171267e+01, 1.554412011549055350e+00, 4.000000000000000222e-01])
samples = np.loadtxt("/home/WVU-AD/ds0172/Desktop/PyBamm-Embedded-GP-main/src-non-jax/src/Data/samples_j0_10_14.csv")
input_dict_init = {}
# beta0 = np.array([2.2,0,0, 0.4,0,0,-32,0,0, 0.4])

# DP = np.mean(samples[:,4][-200:])
DP = -35
# beta0 = np.array([DP, 0, 0,DP,0,0, 2.2,0,0,0.4,0,0, 0.4 ])
# beta0 = np.array([DP, DP, 2.2, 0.4, 0.4])
# beta0 = np.array(samples[-1,:])
beta0 = np.array([11,0, 14,0, -24.04,0, -31.04,0, 0.12**2])
num_betas=len(beta0)



for i in range(num_betas):
    beta_str = "Beta" + str(i)
    param1.update({beta_str: "[input]"},check_already_exists=False)
    input_dict_init[beta_str] = beta0[i]



# for i in range(num_betas):
#     beta_i = samples[:,i]
#     l = "Beta" + str(i)
#     plt.plot(beta_i, label = l)
#     if i < 4:
#         input_dict_init.update({l:np.mean(beta_i[-100])})
#     if i == 4:
#         input_dict_init.update({"Positive particle diffusivity [m2.s-1]":np.exp(np.mean(beta_i[-100]))})
#     if i == 5:
#         input_dict_init.update({"Negative particle diffusivity [m2.s-1]":np.exp(np.mean(beta_i[-100]))})

    # plt.title(l)
    # plt.show()
# plt.show()


mtx = param1.update({'mtx':[[1]]},check_already_exists=False)

def j0p(c_e, c_s_surf, c_s_max, T):
    # This evaluation cannot currently be used in JAX until PyBamm Interpolation can be used in JAX Solver
    betas = []
    for i in range(0,2):
        beta_str_i = "Beta" + str(i)
        betas.append(param1[beta_str_i])
    mtx = param1["mtx"]
    res = evaluate_pybamm(betas, mtx,  [c_s_surf/c_s_max])
    return res


def j0n(c_e, c_s_surf, c_s_max, T):
    betas = []
    for i in range(2,4):
        beta_str_i = "Beta" + str(i)
        betas.append(param1[beta_str_i])
    mtx = param1["mtx"]
    res = evaluate_pybamm(betas, mtx,[c_s_surf/c_s_max])
    return res


def U1(sto, T):

    betas = []
    for i in range(4, 6):
        beta_str_i = "Beta" + str(i)
        betas.append(param1[beta_str_i])
    mtx = param1["mtx"]
    res = np.exp(evaluate_pybamm(betas, mtx, [sto]))
    return res


def U2(sto, T):
    betas = []
    for i in range(6,8):
        beta_str_i = "Beta" + str(i)
        betas.append(param1[beta_str_i])
    mtx = param1["mtx"]
    res = np.exp(evaluate_pybamm(betas, mtx, [sto]))
    return res

    # def U3(sto):
    #     res = evaluate_pybamm(betas_list[4], mtx, [sto], phis)
    #     return res
#
#
# param1["Positive electrode exchange-current density [A.m-2]"] = j0p
# param1["Negative electrode exchange-current density [A.m-2]"] = j0n
# param1["Positive particle diffusivity [m2.s-1]"] = U1
# param1["Negative particle diffusivity [m2.s-1]"] = U2

# param1["Positive electrode exchange-current density [A.m-2]"] = j0p
# param1["Negative electrode exchange-current density [A.m-2]"] = j0n
# param1["Positive particle diffusivity [m2.s-1]"] = U2
# param1["Negative particle diffusivity [m2.s-1]"] = U1
param1["Positive electrode exchange-current density [A.m-2]"] = j0p
param1["Negative electrode exchange-current density [A.m-2]"] = j0n
param1["Positive particle diffusivity [m2.s-1]"] = U1
param1["Negative particle diffusivity [m2.s-1]"] = U2


var_list = ["Negative particle diffusivity [m2.s-1]"]

sim = pybamm.Simulation(batmodel,parameter_values=param1, solver=solver)

solution = sim.solve(t, initial_soc=1, t_interp=t, inputs = input_dict_init)
Vpbi = solution['Voltage [V]'].entries
plt.plot(t, Vpbi)
plt.plot(t, Volt)
plt.show()


# t = np.linspace(0,1787,1787)
def equation(betas_list, mtx, d=True):
    model.tt += 1
    # mtx = [[1]]
    neg_vars = False
    ii = 0
    input_dict = {}
    for func in betas_list:
        for beta in func:
            beta_str_in = 'Beta' + str(ii)
            input_dict.update({beta_str_in: beta})
            ii+=1
    # for i, var in enumerate(var_list):
    #     input_dict.update({var: np.exp(betas_list[i+2][0])})
    #
    ttstart = timeit.default_timer()

    # sim = pybamm.Simulation(batmodel, parameter_values=param1, solver=solver)

    solution = sim.solve(t, initial_soc=1, inputs=input_dict, calculate_sensitivities=d, t_interp=t)
    # tt1 = timeit.default_timer()
    Vpb = solution["Voltage [V]"].entries
    # Vpb[5474] = 0
    # Vpb[5475] = 0

    c = np.min(solution['Negative particle surface concentration [mol.m-3]'].entries)
    # j0p_i = np.min(solution['Positive electrode exchange current density [A.m-2]'].entries)
    # j0n_i = np.min(solution['Negative electrode exchange current density [A.m-2]'].entries)
    Dp_i = np.min(solution["Positive particle effective diffusivity [m2.s-1]"].entries)
    # Dn_i = np.min(solution["Negative particle effective diffusivity [m2.s-1]"].entries)
    #
    # j0p_i = np.max(solution['Positive electrode exchange current density [A.m-2]'].entries)
    # j0n_i = np.max(solution['Negative electrode exchange current density [A.m-2]'].entries)
    Dp_i_max = np.max(solution["Positive particle effective diffusivity [m2.s-1]"].entries)
    # Dn_i = np.max(solution["Negative particle effective diffusivity [m2.s-1]"].entries)

    mins = [c,Dp_i_max,Dp_i]

    if any(m < 0 for m in mins):
        neg_vars = True
    # if i < 3:
    #     var = 'Positive electrode exchange current density [A.m-2]'
    # if i >= 3 and i < 6:
    #     var = 'Negative electrode exchange current density [A.m-2]'
    # if i >= 6:
    #     var = 'Negative particle effective diffusivity [m2.s-1]'
    if d:
        sens = []
        for beta in range(len(beta0)-1):
            beta_str_sens = 'Beta' + str(beta)
            # if beta < 3:
            #     var = 'Positive electrode exchange current density [A.m-2]'
            # if beta >= 3 and beta < 6:
            #     var = 'Negative electrode exchange current density [A.m-2]'
            # if beta >= 6:
            #     var = 'Negative particle effective diffusivity [m2.s-1]'
            # if beta < 3:
            #     dresdfunc = np.log(solution['Voltage [V]'].sensitivities[beta_str_sens])
            dresdfunc = solution['Voltage [V]'].sensitivities[beta_str_sens]
            if beta > 3:
                dresdfunc = dresdfunc*np.exp(input_dict[beta_str_sens])
            dresdfunc[-1] = 0
            # dfuncdbeta = solution[var].sensitivities[beta_str_sens]
            sens.append(dresdfunc)


        tt2 = timeit.default_timer() - ttstart
        print(f"{tt2} to solve + grad")
        # for i, var in enumerate(var_list):
        #     if var == "Positive particle diffusivity [m2.s-1]":
        #         sens.append(solution['Voltage [V]'].sensitivities[var]*solution["Positive particle effective diffusivity [m2.s-1]"].entries[0][0])
        #     elif var == "Negative particle diffusivity [m2.s-1]":
        #         sens.append(solution['Voltage [V]'].sensitivities[var] *
        #                     solution["Negative particle effective diffusivity [m2.s-1]"].entries[0][0])
        #     else:
        #         sens.append(solution['Voltage [V]'].sensitivities[var])

    # ttint = timeit.default_timer() - tt1
    # print(f"no interp:{ttint}")

    n = len(Vpb)
    # print(n)
    if not d:
        plt.plot(t[0:n],Vpb,label='Pybamm')
        plt.plot(t,Volt,label='Experimental')
        plt.title(f"{model.draw}")
        plt.xlabel('Time')
        plt.ylabel('Voltage')
        plt.legend()
        plt.show()

    # if not model.d:
        # plt.plot(Vpb,label='Pybamm')
        # plt.plot(V,label='data')
        # plt.legend()
    # output_variable = ['X-averaged positive particle concentration','X-averaged negative particle concentration']
    # cp = solution['X-averaged positive particle concentration'].entries
    # cn = solution['X-averaged negative particle concentration'].entries
    # plt.plot(cp[0,:])
    # plt.plot(cn[0,:])
    #     save_path = os.path.join(save_folder, f"figure_{model.tt}.png")
    #     plt.savefig(save_path, dpi=300)
    # plt.close()
    # print(f"{model.tt}, {betas_list}")
    #     plt.show()
    # if model.tt > 68:
    #     stop = 1
    if d:
        return Vpb, sens, neg_vars
    else:
        return Vpb


model.set_equation(equation)
# beta0 = np.array(
#     [-5.59750181, -1.70157791, -4.52797968, 2.05051943, -4.81077465, -1.82435794, -7.61931092, 2.71472927, 0.3])

samples, matrix, BIC = model.full_routine(draws=2000, init_betas=beta0, tolerance=0)

np.savetxt('../results/samples_j0_10_14.csv', samples)
np.savetxt('../results/matrix_10_14.csv', matrix)
np.savetxt('../results/BIC_10_14.csv', BIC)

bj0p = samples[0:1, -1]
bj0n = samples[2:3, -1]
bU1 = samples[4:5, -1]
bU2 = samples[6:7, -1]

betas_list_final = [bj0p, bj0n, bU1, bU2]
mtx = np.array([[1]])


def j0p_final(c_e, c_s_surf, c_s_max, T):
    # This evaluation cannot currently be used in JAX until PyBamm Interpolation can be used in JAX Solver
    res = evaluate_pybamm(betas_list_final[0], mtx, [c_s_surf / c_s_max], phis)
    return res


def j0n_final(c_e, c_s_surf, c_s_max, T):
    res = np.exp(evaluate_pybamm(betas_list_final[1], mtx, [c_s_surf / c_s_max], phis))
    return res


def U1_final(sto, T):
    # This evaluation cannot currently be used in JAX until PyBamm Interpolation can be used in JAX Solver
    betas = betas_list_final[2]
    res = np.exp(evaluate_pybamm(betas, mtx, [sto], phis))
    return res


def U2_final(sto, T):
    betas = betas_list_final[3]
    res = np.exp(evaluate_pybamm(betas, mtx, [sto], phis))
    return res


param1["Positive electrode exchange-current density [A.m-2]"] = j0p_final
param1["Negative electrode exchange-current density [A.m-2]"] = j0n_final
param1["Positive particle diffusivity [m2.s-1]"] = U1_final
param1["Negative particle diffusivity [m2.s-1]"] = U2_final
param1["Current function [A]"] = current_interpolant

solver = pybamm.CasadiSolver(mode="fast with events")
sim = pybamm.Simulation(batmodel, parameter_values=param1, solver=solver)

solution = sim.solve(jnp.array(t))
Vpb = solution["Voltage [V]"].entries

plt.plot(t, Vpb, 'r')
plt.plot(t, V, 'g')
plt.show()
