import numpy as np

def gridmaker(minmax,n):

    param_grids = [np.linspace(start, stop, n) for start, stop in minmax]

    mesh = np.meshgrid(*param_grids)
    mesh_grid = np.array([axis.flatten() for axis in mesh]).T
    return mesh_grid

def grid_to_saved_predictions(mesh, fmodel):
    predictions = fmodel.evaluate(inputs=mesh, avgbetas=True)
    fmodel.predictions = predictions
    fmodel.save("builtFoKLModel.fokl")

