import pybamm as pb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


samples = np.genfromtxt('../Data/samples_5_8 - samples1500.csv', delimiter=',')

last_samples = samples[-250: ,:]
j0p = samples[-250:,0]
j0n = samples[-250:,1]
Dp = samples[-250:,2]
Dn = samples[-250:,3]

last_sorted = np.sort(last_samples, axis=0)

fiveConfidence = last_sorted[13,:]
ninetyfiveConfidence = last_sorted[238,:]
meanConfidence = np.mean(last_samples,axis=0)

sms = np.array([fiveConfidence,meanConfidence,ninetyfiveConfidence])

param = pb.ParameterValues('Mohtat2020')

k = "symmetric Butler-Volmer"
batmodel = pb.lithium_ion.SPM({"intercalation kinetics": k})
solver = pb.CasadiSolver(dt_max=30, mode="fast with events")
plt.plot(t,V)
dash = ["--", "-","--"]
lb = ["5% bound","Mean Estimate", "95% Bound"]
for i in range(3):
    j0p = sms[i,0]
    j0n = sms[i,1]
    Dp = sms[i,2]
    Dn = sms[i,3]
    param["Lower voltage cut-off [V]"] = min(V)
    param["Upper voltage cut-off [V]"] = 4.4
    # param1["Contact resistance [Ohm]"] = 0.156
    param["Open-circuit voltage at 0% SOC [V]"] = 0.4
    param["Open-circuit voltage at 100% SOC [V]"] = 4.2
    param['Current function [A]'] = 4
    param["Positive electrode OCP [V]"] = polyfit_ocv
    param["Electrode width [m]"] = 0.25
    param["Positive electrode exchange-current density [A.m-2]"] = j0p
    param["Negative electrode exchange-current density [A.m-2]"] = j0n
    param["Positive particle diffusivity [m2.s-1]"] = np.exp(Dp)
    param["Negative particle diffusivity [m2.s-1]"] = np.exp(Dn)

    sim = pb.Simulation(batmodel, parameter_values=param, solver = solver)
    solution = sim.solve(t, initial_soc=0.97)
    Vpb = solution["Voltage [V]"].entries
    plt.plot(t,Vpb, label=lb[i], linestyle=dash[i])

plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")
plt.show()