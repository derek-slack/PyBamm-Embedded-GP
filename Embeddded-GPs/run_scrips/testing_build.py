import pybamm
import numpy as np
from examples.scripts.new_eval import evaluate_pybamm
import pandas as pd
from FoKL import getKernels
import matplotlib.pyplot as plt

batmodel = pybamm.lithium_ion.SPM()

phis = getKernels.sp500()
param1 = pybamm.ParameterValues("Mohtat2020")

samples = np.genfromtxt("../Data/samples.csv", delimiter=',')

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

betas_final = samples[-200:,:]
b = np.mean(betas_final, axis=0)

betas_list_final = np.zeros([4,2])
n=0
for i in range(4):
    for k in range(2):
        betas_list_final[i,k] = b[n]
        n+=1

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

mtx = [[1.]]
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
param1["Current function [A]"] = 4
param1["Positive electrode OCP [V]"] = polyfit_ocv
param1["Electrode width [m]"] = 0.25

solver = pybamm.CasadiSolver(mode="fast with events")
sim = pybamm.Simulation(batmodel, parameter_values=param1, solver=solver)

solution = sim.solve(np.array(t), initial_soc=0.97)
Vpb = solution["Voltage [V]"].entries

plt.plot(t, Vpb, 'r')
plt.plot(t, V, 'g')
plt.show()

h=1