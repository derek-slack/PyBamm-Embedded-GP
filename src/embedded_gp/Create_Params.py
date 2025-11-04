import pybamm
import numpy as np
from new_eval import evaluate_pybamm, evaluate_pybamm_bernoulli

class ParamUpdate():
    def __init__(self, params, beta0, mtx, kernel = 'Bernoulli'):
        self.params = params
        self.beta_list = beta0[:-1]
        self.beta_inputs = self._to_input_params(self.beta_list)
        self.mtx = mtx
        if kernel == 'Bernoulli':
            self.evaluate_func = evaluate_pybamm_bernoulli
        elif kernel == 'Cubic':
            self.evaluate_func = evaluate_pybamm
        else:
            raise NotImplementedError('kernel must be either "Bernoulli" or "Cubic"')

    def add_function(self, name, arg_inds, list_index,exp=False, div_arg=None):
        beta_func = self.beta_inputs[list_index]
        mtx = self.mtx[list_index]
        if div_arg is not None:
            if exp:
                def pybamm_function(*args):
                    xs = []
                    for x in div_arg:
                        xs.append([args[x[0]]/args[x[1]]])
                    for x in arg_inds:
                        xs.append([args[x]])
                    res = np.exp(self.evaluate_func(beta_func, mtx, xs))
                    return res
            else:
                def pybamm_function(*args):
                    xs = []
                    for x in div_arg:
                        xs.append([args[x[0]]/args[x[1]]])
                    for x in arg_inds:
                        xs.append([args[x]])
                    res = self.evaluate_func(beta_func, mtx, xs)
                    return res
        else:
            if exp:
                def pybamm_function(*args):
                    xs = []
                    for x in arg_inds:
                        xs.append([args[x]])
                    res = np.exp(self.evaluate_func(beta_func, mtx, xs))
                    return res
            else:
                def pybamm_function(*args):
                    xs = []
                    for x in arg_inds:
                        xs.append([args[x]])
                    res = self.evaluate_func(beta_func, mtx, xs)
                    return res
        if type(self.params[name]) is not float:
            function_args = self.params[name].__code__.co_varnames
            function_args_mod = []
            if div_arg is not None:
                for x in div_arg:
                    function_args_mod.append(function_args[x[0]] + str('/') + function_args[x[1]])
            for x in arg_inds:
                function_args_mod.append(function_args[x])
            print(f"GP function created for {name} \n inputs are {function_args_mod}")
        self.params[name] = pybamm_function


    def get_param(self):
        return self.params

    @staticmethod
    def _to_input_params(beta_list):
        beta_IP = []
        ii = 0
        for i, func in enumerate(beta_list):
            beta_IP.append([])
            for j, beta in enumerate(func):
                beta_IP[i].append(pybamm.InputParameter('Beta' + str(i)+str(j)))
                ii+=1

        return beta_IP


    def _to_input_dict(self, beta_list, flat=False):
        input_dict = {}
        if flat:
            ii=0

            for i, func in enumerate(self.beta_list):
                for j, beta in enumerate(func):
                    input_dict.update({'Beta' + str(i) + str(j): beta_list[ii]})
                    ii+=1
        else:

            for i, func in enumerate(beta_list):
                for j, beta in enumerate(func):
                    input_dict.update({'Beta' + str(i)+str(j): beta})

        return input_dict