import pybamm

p = pybamm.ParameterValues("Chen2020")

j0 = p['Positive electrode exchange-current density [A.m-2]']

def new_func(current):
    return current*2
def process_tree(symbol: pybamm.Symbol):

    if isinstance(symbol, pybamm.Parameter) and symbol.name == "My Parameter":
        return symbol
    else:
        # Recursively process all children
        new_children = [process_tree(child) for child in symbol.children]
        # create_copy returns a new node with the new children
        return symbol.create_copy(new_children)

pybamm.Symbol('Positive electrode exchange-current density [A.m-2]', children=p['Current function [A]'])
# Process the tree to replace parameters
modified_f = process_tree(f)

h=1