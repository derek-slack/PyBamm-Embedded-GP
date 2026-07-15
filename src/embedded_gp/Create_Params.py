import pybamm
import numpy as np
from new_eval import evaluate_pybamm, evaluate_pybamm_bernoulli

class ParamUpdate():
    """
    Class for modifying Parameter dictionary in PyBaMM
    """
    def __init__(self, params, beta0, kernel = 'Bernoulli'):
        self.params = params
        self.beta_list = beta0
        self.beta_inputs = self._to_input_params(self.beta_list)
        if kernel == 'Bernoulli':
            self.evaluate_func = evaluate_pybamm_bernoulli
        elif kernel == 'Cubic':
            self.evaluate_func = evaluate_pybamm
        else:
            raise NotImplementedError('kernel must be either "Bernoulli" or "Cubic"')

    def add_function(self, name, mtx, arg_inds, betas_function, exp=False, div_arg=None, div_const = None, new_children = None):
        """
        Creates Parameter function specified as a GP object
        inputs:
            name: str, Name of parameter to be estimated
            arg_inds: list of int, Index of inputs to parameter function from PyBaMM
            list_index: int, Index of function hyperparameters, betas and mtx, supplied in initialization
            exp: Bool, if function should be exponential
            div_arg: list of lists of int, optional, Index of arguments to be used as div
                ex: div_arg = [[1,4],[3,2]]
                inputs to GP would be GP((1/2),(3,2))
        """

        def process_tree(symbol: pybamm.Symbol):
            if isinstance(symbol, pybamm.Parameter) and symbol.name == "My Parameter":
                return symbol
            else:
                new_children = [process_tree(child) for child in symbol.children]
                return symbol.create_copy(new_children)

        beta_func = betas_function

        if div_arg is not None:

            if exp:
                def pybamm_function(*args):
                    xs = []
                    for x in div_arg:
                        xs.append([args[x[0]]/args[x[1]]])
                    for x in arg_inds:
                        xs.append([args[x]])

                    # xs.append([np.log(self.params['Current function [A]'])])
                    res = np.exp(self.evaluate_func(beta_func, mtx, xs))
                    return res
            else:
                def pybamm_function(*args):
                    xs = []
                    for x in div_arg:
                        xs.append([args[x[0]]/args[x[1]]])
                    for x in arg_inds:
                        xs.append([args[x]])
                    # xs.append([np.log(self.params['Current function [A]'])])
                    res = self.evaluate_func(beta_func, mtx, xs)
                    return res
        else:
            if exp:
                def pybamm_function(*args):
                    xs = []
                    for x in arg_inds:
                        if div_const:
                            xs.append([args[x]/div_const[0]])
                        else:
                            xs.append([args[x]])
                    # xs.append([np.log(self.params['Current function [A]'])])
                    res = np.exp(self.evaluate_func(beta_func, mtx, xs))
                    return res
            else:
                def pybamm_function(*args):
                    xs = []
                    for x in arg_inds:
                        xs.append([args[x]])
                    # xs.append([np.log(self.params['Current function [A]'])])
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