import timeit

import pybamm
import numpy as np
from new_eval import *
import FoKL
from FoKL import FoKLRoutines
model = pybamm.lithium_ion.SPM()
solver = pybamm.AlgebraicSolver(output_variables=['Voltage [V]'])

DN_model = FoKL.FoKLRoutines.load("../modesl/DN_model.fokl")
betas = DN_model.betas
mtx = DN_model.mtx
bm = np.mean(betas, axis=0).reshape(1, -1)

T_max_liion = 80 + 273.15
T_min_liion = -30 + 273.15

params = pybamm.ParameterValues("Chen2020")
input_dict = {}
for i in range(len(bm[0]-7)):
    beta_str = "Beta" + str(i)
    params.update({beta_str: "[input]"},check_already_exists=False)
    input_dict.update({beta_str: bm[0][i]})
params.update({"mtx": mtx[0:8]},check_already_exists=False)
# , (T - T_min_liion) / (T_max_liion - T_min_liion)]
def DP_func(sto, T):
    betas = []
    for i in range(len(bm[0])):
        beta_str_i = "Beta" + str(i)
        betas.append(params[beta_str_i])
    mtx = params["mtx"]
    return np.exp(evaluate_pybamm(betas, mtx, [sto, (T - T_min_liion) / (T_max_liion - T_min_liion)]))
params["Positive particle diffusivity [m2.s-1]"] = DP_func
for i in range(1):
    for i in range(len(bm[0])):
        beta_str = "Beta" + str(i)
        params.update({beta_str: "[input]"}, check_already_exists=False)
        input_dict.update({beta_str: bm[0][i] + np.random.normal(0.1)})
    params.update({"mtx": mtx}, check_already_exists=False)
    t = timeit.default_timer()
    sim = pybamm.Simulation(model, parameter_values=params, solver=solver)
    sol = sim.solve([0, 3600], initial_soc=1., inputs = input_dict, calculate_sensitivities=True)
    t2 = timeit.default_timer() - t
    print(t2)
# sol["Positive electrode diffusivity [m2.s-1]"].entries
sim.plot()