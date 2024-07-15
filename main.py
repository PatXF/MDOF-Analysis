import numpy as np

from stiff import *
from dynamics_solver import *
import matplotlib.pyplot as plt


def loader_linear(slope, x, b):
    return slope * x + b


def loader_harmonic(load_type, x, amplitude):
    if load_type == "sin":
        return amplitude * np.sin(x)

    elif load_type == "cos":
        return amplitude * np.cos(x)


elements = [
    [0, 1, 5.469e10, 1, 1],
    [1, 2, 5.469e10, 1, 1],
    [2, 3, 5.469e10, 1, 1],
    [3, 4, 5.469e10, 1, 1],
    [4, 5, 5.469e10, 1, 1],
]

nodes = [
    [0, 0],
    [120, 0],
    [240, 0],
    [360, 0],
    [480, 0],
    [600, 0]
]

boundary_conditions = {0: 0, 1: 0, 2: 0}
user_dofs = {4: 0, 7: 1, 10: 2, 13: 3,
             16: 4, 3: 5, 5: 6, 6: 7, 8: 8,
             9: 9, 11: 10, 12: 11, 14: 12,
             15: 13, 17: 14}

k = modified_stiff(elements, nodes, boundary_conditions, user_dofs, sub_axis=False)
k_hat = static_condensation(k, 10)

m = 208.6 * np.array([
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0.5]
])

c = np.zeros((5, 5))
u0 = np.zeros((5, 1))
u_0 = np.zeros((5, 1))

bias = np.zeros((5, 1))
bias[4][0] = 1000
slopes = np.zeros((5, 1))

response = solver(m, c, k_hat, u0, u_0, 0.1, 0.5, 0.167, 4, loader_linear, 4, 2, slopes, bias, "linear")

x = np.arange(0, 2.1, 0.1)
y = [0]

for i in range(20):
    y.append(response[i][0][0])

temp_y = [0, -0.001, -0.0055, -0.0086, -0.0035, 0.011, 0.0295, 0.0444, 0.0526, 0.0577, 0.0669, 0.0838, 0.1050, 0.1232, 0.1326, 0.1345, 0.1353, 0.1409, 0.1513, 0.1601, 0.1605]

plt.plot(x, y)
plt.scatter(x[:21], temp_y, color="red")
plt.title("Displacement of the "
          "first degree of freedom \nfrom program vs the scatter plot representation of the results \nobtained in the "
          "u1 column of Anil K. Chopra, Table 15.1 in page no. 573"
          )
plt.show()

