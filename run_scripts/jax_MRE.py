import os
import timeit
import pybamm
from src.embedded_gp.new_eval import *
import os

from post_process import PostProcess

os.environ['JAX_PLATFORM_NAME'] = 'cpu'
import jax
model_jax = pybamm.lithium_ion.SPMe()
model_jax.convert_to_format = 'jax'
model_jax.events = []

model_idak = pybamm.lithium_ion.SPMe()
model_idak.convert_to_format = 'jax'
model_idak.events = []


geometry_jax = model_jax.default_geometry
geometry_idak = model_idak.default_geometry
param_jax = pybamm.ParameterValues("Chen2020")
param_idak = pybamm.ParameterValues("Chen2020")

beta0 = np.array([-32, 0.05, 0.05])
num_betas=len(beta0)
input_dict = {}

for i in range(num_betas):
    beta_str = "Beta" + str(i)
    param_idak.update({beta_str: "[input]"},check_already_exists=False)
    param_jax.update({beta_str: "[input]"}, check_already_exists=False)
    input_dict[beta_str] = beta0[i]

param_jax.update({'mtx':np.array([[1],[2]])}, check_already_exists=False)
param_idak.update({'mtx':np.array([[1],[2]])}, check_already_exists=False)


def U1_jax(sto, T):
    betas = []
    for i in range(0,3):
        beta_str_i = "Beta" + str(i)
        betas.append(param_jax[beta_str_i])
    mtx = param_jax["mtx"]
    res = np.exp(evaluate_pybamm_bernoulli(betas, mtx, [[sto]]))
    return res

def U1_idak(sto, T):
    betas = []
    for i in range(0,3):
        beta_str_i = "Beta" + str(i)
        betas.append(param_idak[beta_str_i])
    mtx = param_idak["mtx"]
    res = np.exp(evaluate_pybamm_bernoulli(betas, mtx, [[sto]]))
    return res


def current_func(time):
    I = 1.25
    return I

param_jax["Current function [A]"] = current_func
param_idak["Current function [A]"] = current_func
param_jax["Positive particle diffusivity [m2.s-1]"] = U1_jax
param_idak["Positive particle diffusivity [m2.s-1]"] = U1_idak
t_eval = np.linspace(0,3600,500)
output_variables = ["Voltage [V]"]

param_jax.process_geometry(geometry_jax)
param_jax.process_model(model_jax)
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 20, var.x_s: 20, var.x_p: 20, var.r_n: 10, var.r_p: 10}
mesh = pybamm.Mesh(geometry_jax, model_jax.default_submesh_types, var_pts)
disc = pybamm.Discretisation(mesh, model_jax.default_spatial_methods)
disc.process_model(model_jax)

param_idak.process_geometry(geometry_idak)
param_idak.process_model(model_idak)
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 20, var.x_s: 20, var.x_p: 20, var.r_n: 10, var.r_p: 10}
mesh = pybamm.Mesh(geometry_idak, model_idak.default_submesh_types, var_pts)
disc = pybamm.Discretisation(mesh, model_idak.default_spatial_methods)
disc.process_model(model_idak)

solver = pybamm.JaxSolver(atol=1e-4, rtol=1e-4)
solve1 = solver.solve(model_jax, t_eval, inputs = input_dict)
jax_solver = solver.create_solve(model_jax, t_eval)


Idak_solver = pybamm.IDAKLUSolver(atol=1e-4, rtol=1e-4, output_variables=output_variables)
Idak_solver_new = pybamm.IDAKLUSolver(atol=1e-4, rtol=1e-4, output_variables=output_variables, options={"num_threads": 32})
IJ = Idak_solver.jaxify(model_idak,t_eval)
f = IJ.get_jaxpr()
sim = pybamm.Simulation(model_idak,parameter_values=param_idak, solver=Idak_solver_new)
sol_base = sim.solve(t_eval,inputs=input_dict, t_interp=t_eval)

# Create 'Data' for rmse
data_premade = sol_base['Voltage [V]'].entries
data_noise = data_premade + np.random.normal(0,1e-2,size=len(data_premade))

jax_evaluator = pybamm.EvaluatorJax(model_jax.variables['Voltage [V]'])


def get_voltage_jax(input_dict):
    ins = {"Beta0": input_dict['Beta0'], 'Beta1': input_dict['Beta1'], 'Beta2': input_dict['Beta2']}
    ys = jax_solver(ins)
    V = jax.vmap(jax_evaluator.__call__,in_axes=(None,1),out_axes=0)(t_eval,ys)
    return V

def rmse_jax(input_dict, data):
    V = get_voltage_jax(input_dict)
    V = V.reshape(-1,1)
    data = data.reshape(-1,1)
    rmse = jnp.sqrt(jnp.sum((V - data)**2)/len(data))
    return rmse

# Get Value and Gradient of Inputs
jax_grad = jax.grad(rmse_jax, argnums=0)
grad = jax_grad(input_dict, data_noise)
h=1


