import timeit

import pybamm
import jax.numpy as jnp
import jax
import numpy as np

batmodel = pybamm.lithium_ion.SPM()
import multiprocessing

# Set Parameter values
param = pybamm.ParameterValues("Mohtat2020")
batsolver = pybamm.CasadiSolver()
Is = np.array([1, 1.5, 2, 2.5, 3, 3.5])


def solve_pb(model, solver, params):
    param['Current function [A]'] = params
    sim = pybamm.Simulation(model,parameter_values=param,solver=solver)
    return sim.solve([0, 3600], initial_soc=0.8)


def solve_pb_single(I):
    return solve_pb(batmodel, batsolver, I)

if __name__ == "__main__":
    t1 = timeit.default_timer()
    with multiprocessing.Pool() as pool:
        sols = pool.map(solve_pb_single, Is)
    t2 = timeit.default_timer() - t1
    t3 = timeit.default_timer()
    for i in range(len(Is)):
        sol = solve_pb_single(Is[i])

    t4 = timeit.default_timer() - t3

    print(f"Pooled = {t2}, serial = {t4}")
