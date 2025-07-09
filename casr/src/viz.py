import numpy as np
from netCDF4 import Dataset


def load_matrix(file_path):
    with open(file_path, "rb") as fichier:
        # Lire les dimensions de la matrice
        dims = np.fromfile(fichier, dtype=np.int32, count=2)
        grid_size = tuple(dims)

        # Lire les données de la matrice
        matrix = np.fromfile(fichier, dtype=np.float32, count=np.prod(grid_size))
        matrix = np.reshape(matrix, grid_size, order="F")

    return matrix


def load_3darray(file_path):
    with open(file_path, "rb") as fichier:
        # Lire les dimensions de la matrice
        dims = np.fromfile(fichier, dtype=np.int32, count=3)
        array_shape = tuple(dims)

        # Lire les données de la matrice
        data = np.fromfile(fichier, dtype=np.float32, count=np.prod(array_shape))
        array = np.reshape(data, array_shape, order="F")

    return array


def load_ncfile_pr(file_path):
    dataset = Dataset(file_path, "r")
    var = dataset.variables["CaSR_v3.1_A_PR0_SFC"][:]
    dataset.close()
    return var


def load_ncfile_lat(file_path):
    dataset = Dataset(file_path, "r")
    var = dataset.variables["lat"][:]
    dataset.close()
    return var


def load_ncfile_lon(file_path):
    dataset = Dataset(file_path, "r")
    var = dataset.variables["lon"][:]
    dataset.close()
    return var


def ncinfo(file_path):
    with Dataset(file_path, "r") as nc:
        print(f"Dimensions: {nc.dimensions}")
        print(f"Variables: {nc.variables}")
        print(f"Attributes: {nc.ncattrs()}")


def reconstruct_grid(vector, bounds):
    return np.reshape(vector, bounds)


def computeReturnLevel(
    returnPeriod: int,
    mu=np.array([]),
    sigma=np.array([]),
    xi=0.0,
):
    return mu - sigma / xi * (1 - (-np.log(1 - 1 / returnPeriod)) ** (-xi))
