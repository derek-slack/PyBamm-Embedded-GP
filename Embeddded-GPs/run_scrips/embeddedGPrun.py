import pybamm
import pybamm as pb
import numpy as np
import jax.numpy as jnp
from embedded_gp import Experimental_Embedded_GPs

from new_eval import evaluate_pybamm
import pandas as pd
import matplotlib.pyplot as plt
from FoKL import getKernels
import warnings
import os

phis = getKernels.sp500()
warnings.filterwarnings("ignore")

k = "symmetric Butler-Volmer"
pb.set_logging_level("NOTICE")
batmodel = pybamm.lithium_ion.SPM({"intercalation kinetics": k})
# batmodel.events = []
# batmodel.convert_to_format = 'jax'

save_folder = "figures"
os.makedirs(save_folder, exist_ok=True)

# remove events (not supported in jax)
batgeometry = batmodel.default_geometry

# Set Parameter values
param1 = pb.ParameterValues("Mohtat2020")


# Define the phis (basis functions) used


def normalize_inputs(inputs, min, max):
    normalized = (inputs - min) / (max - min)
    return normalized


C = pd.read_csv('Chargecycle.csv', header=None)
D = pd.read_csv('Dischargecycle.csv', header=None)

C = C.to_numpy()
D = D.to_numpy()

TC = C[0, 3:]
TD = D[0, 3:] - max(C[0, 3:]) + 0.000001

IC = -C[1, 3:]
ID = -D[1, 3:]

VC = C[2, 3:]
VD = D[2, 3:]

ID_M = []
VD_M = []
TD_M = []

for i in range(336):
    if np.mod(i, 2) == 0:
        ID_M.append(ID[i])
        VD_M.append(VD[i])
        TD_M.append(TD[i])

I = np.array(ID_M)
t = np.array(TD_M) - 30
V = np.array(VD_M)
IJ = jnp.array(I)
# Define inputs and normalize
inputs = jnp.array([I])
inputs_norm = normalize_inputs(inputs, np.min(I), np.max(I))

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
model.inputs = np.transpose(inputs_norm)
model.phis = phis
model.data = np.transpose(V)



minI = -4.0304
maxI = 1.5145

polyfit = [-4186.54559829646, 18363.2314210941, -33927.0311238327,
           34177.2778419589, -20221.9632429130, 6960.48498715931,
           -1220.79587333825, 33.6148046197656, 24.1935530311832,
           -4.65707497961982, 4.31012644510537]


def polyfit_ocv(sto):
    X = 0
    l = len(polyfit)
    for i in range(l):
        X += polyfit[i] * sto**(l-i-1)
    return X

def polyfit_ocv_neg(sto):
    X = 0
    sto = 1- sto
    l = len(polyfit)
    for i in range(l):
        X += polyfit[i] * sto**(l-i-1)
    return X * - 0.5


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

experiment = pybamm.Experiment(
    [
        "Discharge at 4A for 10 seconds", "Rest at 0 A for 10 seconds"
    ] * 168
)


current_interpolant = pybamm.Interpolant(t, I, pybamm.t)#, interpolator="JAX")  # , _num_derivatives=0)
param1["Current function [A]"] = 4
param1["Positive electrode OCP [V]"] = polyfit_ocv
param1["Electrode width [m]"] = 0.25
# param1["Negative electrode porosity"] = 0.5
# param1["Negative electrode OCP [V]"] = 0
# param1["Open-circuit voltage at 0% SOC [V]"] = 0
# param1["Open-circuit voltage at 100% SOC [V]"] = 4.31
param1["Initial concentration in positive electrode [mol.m-3]"] = 31513.0 * 0.64
# param1['Nominal cell capacity [A.h]'] = 2.1
param1["Lower voltage cut-off [V]"] = 1.5
param1["Upper voltage cut-off [V]"] = 4.4
# param1["Contact resistance [Ohm]"] = 0.156
param1["Open-circuit voltage at 0% SOC [V]"] = 0.4
param1["Open-circuit voltage at 100% SOC [V]"] = 4.2


solver = pybamm.IDAKLUSolver()

num_betas=9

for i in range(num_betas):
    beta_str = "Beta" + str(i)
    param1.update({beta_str: "[input]"},check_already_exists=False)
    # input_dict.update({beta_str: bm[0][i]})
# param1.update({"mtx": mtx[0:8]},check_already_exists=False)


#
# param1["Positive electrode exchange-current density [A.m-2]"] = 1
# param1["Negative electrode exchange-current density [A.m-2]"] = 2
# param1["Positive particle diffusivity [m2.s-1]"] = 12e-4
# param1["Negative particle diffusivity [m2.s-1]"] = 2e-5
# param1["Current function [A]"] = current_interpolant

#
# # param1["Initial concentration in positive electrode [mol.m-3]"] = 47513.0 * 0.58
sim = pybamm.Simulation(batmodel,parameter_values=param1, solver=solver)

solution = sim.solve(t, initial_soc=0.97)
Vpbi = solution['Voltage [V]'].entries
plt.plot(Vpbi)
plt.plot(V)
plt.show()
model.solution = None
model.tt = 0
j0p_i = np.mean(solution['Positive electrode exchange current density [A.m-2]'].entries[0])
j0n_i = np.mean(solution['Negative electrode exchange current density [A.m-2]'].entries[0])
Dp_i = np.mean(solution["Positive particle effective diffusivity [m2.s-1]"].entries[0][0])
Dn_i = np.mean(solution["Negative particle effective diffusivity [m2.s-1]"].entries[0][0])

beta0 = np.array([j0p_i,1,j0n_i,-1, np.log(Dp_i),-1, np.log(Dn_i),1, 1e-2])
mtx = param1.update({'mtx':[[1]]},check_already_exists=False)

def j0p(c_e, c_s_surf, c_s_max, T):
    # This evaluation cannot currently be used in JAX until PyBamm Interpolation can be used in JAX Solver
    betas = []
    for i in range(2):
        beta_str_i = "Beta" + str(i)
        betas.append(param1[beta_str_i])
    mtx = param1["mtx"]
    res = evaluate_pybamm(betas, mtx, [c_s_surf / c_s_max])
    return res


def j0n(c_e, c_s_surf, c_s_max, T):
    betas = []
    for i in range(2, 4):
        beta_str_i = "Beta" + str(i)
        betas.append(param1[beta_str_i])
    mtx = param1["mtx"]
    res = np.exp(evaluate_pybamm(betas, mtx, [c_s_surf / c_s_max]))
    return res


def U1(sto, T):
    # This evaluation cannot currently be used in JAX until PyBamm Interpolation can be used in JAX Solver
    betas = []
    for i in range(4, 6):
        beta_str_i = "Beta" + str(i)
        betas.append(param1[beta_str_i])
    mtx = param1["mtx"]
    res = np.exp(evaluate_pybamm(betas, mtx, [sto]))
    return res


def U2(sto, T):
    betas = []
    for i in range(6, 8):
        beta_str_i = "Beta" + str(i)
        betas.append(param1[beta_str_i])
    mtx = param1["mtx"]
    res = np.exp(evaluate_pybamm(betas, mtx, [sto]))
    return res

    # def U3(sto):
    #     res = evaluate_pybamm(betas_list[4], mtx, [sto], phis)
    #     return res


param1["Positive electrode exchange-current density [A.m-2]"] = j0p
param1["Negative electrode exchange-current density [A.m-2]"] = j0n
param1["Positive particle diffusivity [m2.s-1]"] = U1
param1["Negative particle diffusivity [m2.s-1]"] = U2

# t = np.linspace(0,1787,1787)
def equation(betas_list, mtx, d=True):
    model.tt += 1
    # mtx = [[1]]
    ii = 0
    input_dict = {}
    for func in betas_list:
        for beta in func:
            beta_str_in = 'Beta' + str(ii)
            input_dict.update({beta_str_in: beta})
            ii+=1


    sim = pybamm.Simulation(batmodel, parameter_values=param1, solver=solver)

    solution = sim.solve(t, initial_soc=0.97, inputs=input_dict,calculate_sensitivities=d, t_interp=t)

    Vpb = solution["Voltage [V]"].entries
    if d:
        sens = []
        for beta in range(len(beta0)-1):
            beta_str_sens = 'Beta' + str(beta)
            sens.append(solution['Voltage [V]'].sensitivities[beta_str_sens])

    n = len(Vpb)
    print(n)
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
        return Vpb, sens
    else:
        return Vpb


model.set_equation(equation)
# beta0 = np.array(
#     [-5.59750181, -1.70157791, -4.52797968, 2.05051943, -4.81077465, -1.82435794, -7.61931092, 2.71472927, 0.3])

samples, matrix, BIC = model.full_routine(draws=1000, init_betas=beta0, tolerance=0)

np.savetxt('../Data/samples.csv', samples)
np.savetxt('matrix.csv', matrix)

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
