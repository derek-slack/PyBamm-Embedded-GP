import matplotlib
matplotlib.use("Tkagg")
import matplotlib.pyplot as plt
import numpy as np

import pybamm
import optuna

# Import default version of the FoKL package
import FoKL
from FoKL import Experimental_Embedded_GPs

from embedded_gp.new_eval import evaluate_pybamm

from FoKL import getKernels



if __name__ == '__main__':
    # Create PyBamm Model
    model = pybamm.BaseModel()

    x = pybamm.Variable("x")

    u = pybamm.sin(pybamm.t)

    a = pybamm.InputParameter("a")
    b = pybamm.InputParameter("b")

    dxdt = a * x + (b * u)

    model.rhs = {x: dxdt}

    model.initial_conditions = {x: pybamm.Scalar(1)}

    model.variables = {"x": x}

    disc = pybamm.Discretisation()  # use the default discretisation
    disc.process_model(model)

    solver = pybamm.IDAKLUSolver()
    t = np.linspace(0, 5, 75)

    # GP models
    # Create object for each individual GP
    A = Experimental_Embedded_GPs.GP()
    B = Experimental_Embedded_GPs.GP()


    # Create of model and define the number of GP's in it
    GP_model = Experimental_Embedded_GPs.Embedded_GP_Model(A,B)

    GP_model.inputs = np.transpose(np.array([t]))
    phis = getKernels.sp500()
    GP_model.phis = phis


    def toy_solution(t, a, b, x0):
        """
        Analytical solution for dx/dt = a*x + b*t with x(0) = x0
        """

        return (x0 + b/(a**2 + 1))*np.exp(a*t) - b/(a**2 + 1) * (a*np.sin(t) + np.cos(t))

    # Generate Data
    a_in = 0.6
    b_in = 0.25
    x0 = 1.

    x_data = toy_solution(t, a_in, b_in, x0) + np.random.normal(0,1e-1,len(t))


    GP_model.data = np.transpose(x_data)
    num_betas = 4
    beta0 = [0.5,0.5,1]
    var_list = ['a','b']

    def optuna_equation(trial: optuna.trial):
        a = trial.suggest_float('a', 0,1)
        b = trial.suggest_float('b',0,1)
        solution = solver.solve(model, t, t_interp=t, inputs={'a':a, 'b':b})
        res = solution['x'](t)

        MSE = sum((res-x_data)**2)/len(res)

        return MSE

    # sampler = optuna.samplers.GPSampler()
    study = optuna.create_study()
    study.optimize(optuna_equation, n_trials=100)
    print(study.best_value)
    print(study.best_params)

    h=1
    # def equation(beta, mtx, d=True):
    #     solution = solver.solve(model, t, t_interp=t, inputs={'a':beta[0], 'b':beta[1]}, calculate_sensitivities=d)
    #     res = solution['x'](t)
    #     if d:
    #         sens = []
    #         for var in var_list:
    #             beta_str_sens = var
    #             sens.append(solution['x'].sensitivities[beta_str_sens])
    #
    #         return res, sens, False
    #     else:
    #         # plt.plot(t,res, label = 'model')
    #         # plt.plot(t, x_data, label = 'data')
    #         # plt.legend()
    #         # plt.show()
    #         return res
    #
    #
    # GP_model.set_equation(equation)
    #
    # draws = 100
    #
    # samples, matrix, BIC = GP_model.full_routine(draws=draws, init_betas=beta0, tolerance=0)
    # print(samples)
    # d_vec = np.linspace(0,draws,draws+1)
    # ones = np.ones(d_vec.shape)
    # plt.plot(d_vec,samples[:,0], label = 'a prediction')
    # plt.plot(d_vec,samples[:,1], label = 'b prediction')
    # plt.plot(d_vec,a_in*ones,'k--',label="a actual")
    # plt.plot(d_vec, b_in*ones,'k--', label="b actual")
    # plt.ylim(0,1)
    # plt.xlabel("Iterations")
    # plt.title("HMC samples")
    # plt.legend()
    # plt.show()
    # t_fine = np.linspace(0, t[-1], 1000)
    # #
    # # t_sol, y_sol = solution.t, solution.y  # get solution times and states
    # # x = solution["x"]  # extract and process x from the solution
    # # y = solution["y"]  # extract and process y from the solution
    #
    # x_sol = toy_solution(t_fine, a_in, b_in, x0)
    #
    # a_50 = np.mean(samples[-50:, 0])
    # b_50 = np.mean(samples[-50:, 1])
    #
    # x_sol_50 = toy_solution(t_fine, a_50, b_50, x0)
    #
    # a_sort = np.sort(samples[-50:,0])
    # a_5 = a_sort[2]
    # a_95 = a_sort[-2]
    #
    # b_sort = np.sort(samples[-50:, 1])
    # b_5 = b_sort[2]
    # b_95 = b_sort[-2]
    #
    # x_sol_5 = toy_solution(t_fine, a_5, b_5, x0)
    # x_sol_95 = toy_solution(t_fine, a_95, b_95, x0)
    #
    # fig, (ax1) = plt.subplots(1, 1, figsize=(13, 4))
    # ax1.plot(t_fine, x_sol)
    # ax1.plot(t_fine, x_sol_50)
    # ax1.plot(t, x_data, 'o')
    # ax1.plot(t_fine, x_sol_5,'k--', t_fine, x_sol_95, 'k--')
    # ax1.set_xlabel("t")
    # ax1.set_ylabel("x(t)")
    # ax1.legend(["analytical", "model prediction","data","bounds"], loc="best")
    #
    # plt.show()
    #
