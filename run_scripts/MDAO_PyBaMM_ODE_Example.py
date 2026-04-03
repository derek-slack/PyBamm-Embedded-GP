
import numpy as np

import pybamm


import openmdao.api as om



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

    def toy_solution(t, a, b, x0):
        """
        Analytical solution for dx/dt = a*x + b*t with x(0) = x0
        """

        return (x0 + b/(a**2 + 1))*np.exp(a*t) - b/(a**2 + 1) * (a*np.sin(t) + np.cos(t))

    # Generate Data
    a_in = 0.6
    b_in = 0.25
    x0 = 1.

    x_data = toy_solution(t, a_in, b_in, x0) + np.random.normal(0,1e-8,len(t))

    prob = om.Problem()

    num_betas = 4
    beta0 = [0.5,0.5,1]
    var_list = ['a','b']

    class Toy_ODE(om.ExplicitComponent):
        def setup(self):
            self.add_input('a', val=0.0)
            self.add_input('b', val=0.0)

            self.add_output('z', val=0.0)

            self.model = model
            self.solver = solver

        def setup_partials(self):
            self.declare_partials('*','*')

        def compute_partials(self, inputs, partials):
            s_a = self.sol.sensitivities['a'].flatten()
            s_b = self.sol.sensitivities['b'].flatten()

            x = self.sol['x'].entries

            p_a = np.sum(2 * s_a * (x - x_data)) / len(t)
            p_b = np.sum(2 * s_b * (x - x_data)) / len(t)

            partials['z','a'] = p_a
            partials['z','b'] = p_b
        def compute(self, inputs, outputs):
            a = inputs['a']
            b = inputs['b']

            sol = self.solver.solve(self.model, t, t_interp=t, inputs={'a': a, 'b': b}, calculate_sensitivities=True)
            self.sol = sol
            outputs['z'] = np.sum((sol['x'].entries - x_data)**2,axis=0)/len(t)



    om_model = om.Group()
    om_model.add_subsystem('ode', Toy_ODE())
    prob = om.Problem(om_model)
    prob.setup()


    prob.set_val('ode.a', 0.6)
    prob.set_val('ode.b', 0.25)

    prob.run_model()
    print(prob['ode.z'])

    prob.set_val('ode.a', 0.7)
    prob.set_val('ode.b', 0.3)

    prob.run_model()
    print(prob.get_val('ode.z'))

    # setup the optimization
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['maxiter'] = 100

    prob.model.add_design_var('ode.a', lower=0, upper=1)
    prob.model.add_design_var('ode.b', lower=0, upper=1)
    prob.model.add_objective('ode.z')

    prob.setup()

    # Set initial values.
    prob.set_val('ode.a', 0.5)
    prob.set_val('ode.b', 0.1)

    # run the optimization
    prob.run_driver()

    print(prob.get_val('ode.a'))
    print(prob.get_val('ode.b'))


    h=1
