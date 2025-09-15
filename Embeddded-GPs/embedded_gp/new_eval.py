import timeit

import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

import FoKL

from FoKL.getKernels import sp500

phis = sp500()

import src.pybamm as pybamm


def evaluate_pybamm(betas, mtx, inputs, coeff = None):
    """
    Pybamm Function evaluation
    betas: indexed from beta list that relates to
    """
    # tt = timeit.default_timer()
    #
    # print(f"time to spline eval: {timeit.default_timer()-tt}")

    m = jnp.shape(betas)[0]
    mbets = 1
    n = jnp.shape(inputs)[0]  # Size of normalized inputs
    num_basis_terms = len(mtx)
    num_inputs = len(mtx[0])
    X_sol = []
    init = 0

    mtx = jnp.array(mtx)
    phind = []
    for i in range(num_inputs):
        phind_temp = inputs[i]*499
        sett = (phind_temp == 0)
        phind_temp = phind_temp + sett
        r = 1 / 499  # interval of when basis function changes (i.e., when next cubic function defines spline)
        xmin = (phind_temp - 1) * r
        X = (inputs[i]) / r
        phind.append(phind_temp - 1)
        xsm = 499 * inputs[i] - phind_temp

    A = [1, 2, 3]

    X_sc = [(1-inputs[0])**a for a in A]

    lspace = []
    for i in range(num_inputs):
        lspace.append(np.linspace(0,499,499))
    lspace = np.array(lspace)

    # phind[0].children[0].size = 1



    for j in range(num_basis_terms):
        phi = 1
        for k in range(num_inputs):
            num = mtx[j][k]

            if num > 0:
                nid = int(num - 1)

                # coeff = []
                if coeff is None:
                    coeff = []
                    init = 1
                    for jj in range(4):
                        phispace = phis[nid][jj].reshape(1,-1)
                        # test = interpolator_500(lspace[0], phispace[0], phind[k])
                        phi_interp = pybamm.Interpolant(lspace[0], phispace[0], phind[k])#, interpolator="JAX")#,_num_derivatives=0)
                        coeff.append(phi_interp)

                # multiplies phi(x0)*phi(x1)*etc.
                phi *= coeff[0] + coeff[1] * X_sc[0] + coeff[2] * X_sc[1] + coeff[3] * X_sc[2]
        X_sol.append(phi)

    X_sol_ones = betas[0]
    mean = X_sol_ones
    for i in range(len(X_sol)):
        X_sol_betas = X_sol[i]*betas[i+1]
        mean += X_sol_betas

    # beta_matrix = np.eye(16)
    # X_sol_ones = pybamm.Inner(betas, beta_matrix[0, :])
    # mean = X_sol_ones
    #
    # for i in range(1, len(X_sol) - 1):
    #     here = pybamm.Inner(betas, beta_matrix[:, i])
    #     there = pybamm.Multiplication(X_sol[i], here)
    #     mean = pybamm.Addition(mean, there)
    # if init == 1:
    #     return mean, coeff
    # else:
    return mean


def evaluate_pybamm_test(betas, mtx, inputs, phis):
    """
    Pybamm Function evaluation
    betas: indexed from beta list that relates to
    """
    # tt = timeit.default_timer()
    #
    # print(f"time to spline eval: {timeit.default_timer()-tt}")
    m = jnp.shape(betas)[0]
    mbets = 1
    n = jnp.shape(inputs)[0]  # Size of normalized inputs
    mputs = len(inputs)
    X_sol = []


    mtx = jnp.array(mtx)
    phind = []
    for i in range(mputs):
        phind_temp = inputs[i]*499
        sett = (phind_temp == 0)
        phind_temp = phind_temp + sett
        r = 1 / 499  # interval of when basis function changes (i.e., when next cubic function defines spline)
        xmin = (phind_temp - 1) * r
        X = (inputs[i]) / r
        phind.append(phind_temp - 1)
        xsm = 499 * inputs[i] - phind_temp

    A = [1, 2, 3]
    X_sc = []
    for i in range(len(inputs)):
        X_sc.append([(1-inputs[i])**a for a in A])
    X_sc = np.array(X_sc)
    lspace = []
    for i in range(mputs):
        lspace.append(np.linspace(0,499,499))
    lspace = np.array(lspace)

    # phind[0].children[0].size = 1

    phi = 1

    for k in range(mputs):
        num = mtx[0][0]
        X_sc_i = X_sc[k,:]
        if num > 0:
            nid = int(num - 1)

            coeff = []

            for jj in range(4):
                phispace = phis[nid][jj].reshape(1,-1)

                phi_interp = jsp.interpolate.RegularGridInterpolator((lspace[0],), phispace[0])#, interpolator="JAX")#,_num_derivatives=0)
                ress = phi_interp(phind[k])
                coeff.append(ress)

            # multiplies phi(x0)*phi(x1)*etc.
            phi *= coeff[0] + coeff[1] * X_sc_i[0] + coeff[2] * X_sc_i[1] + coeff[3] * X_sc_i[2]
    X_sol.append(phi)

    X_sol_ones = betas[0]
    mean = X_sol_ones
    for i in range(len(X_sol)):
        X_sol_betas = X_sol[i]*betas[i+1]
        mean += X_sol_betas

    return mean


def interpolator_500(x_l, y_l, xi):
    x2 = pybamm.Ceil(xi)
    x1 = pybamm.Floor(xi)

    y1 = y_l[x1]
    y2 = y_l[x2]

    yi = y2 - (y2-y1)*(x2-xi)/(x2-x1)
    return yi