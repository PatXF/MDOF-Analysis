import numpy as np


def glob_stiff(elements, nodes):
    num_nodes = len(nodes)
    global_stiff = np.zeros((2 * num_nodes, 2 * num_nodes))
    for element in elements:
        n1 = element[0]
        n2 = element[1]
        E = element[2]
        I = element[3]
        x1 = nodes[n1][0]
        y1 = nodes[n1][1]
        x2 = nodes[n2][0]
        y2 = nodes[n2][1]

        L = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        lambda_x = (x2 - x1) / L
        lambda_y = (y2 - y1) / L
        transform_matrix = np.array([
            [lambda_x, lambda_y, 0, 0, 0, 0],
            [-lambda_y, lambda_x, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, lambda_x, lambda_y, 0],
            [0, 0, 0, -lambda_y, lambda_x, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        a = 12 / (L ** 3)
        b = 6 / (L ** 2)
        c = 4 / L
        d = 2 / L
        stiff = np.array([[a, b, -a, b],
                                 [b, c, -b, d],
                                 [-a, -b, a, -b],
                                 [b, d, -b, c]])
        stiff = np.dot(np.dot(transform_matrix.T, stiff), transform_matrix) * (E * I)

        l1 = 2 * n1 - 1
        l2 = 2 * n1
        l3 = 2 * n2 - 1
        l4 = 2 * n2
        indexes = [l1, l2, l3, l4]
        for i in range(4):
            for j in range(4):
                global_stiff[indexes[i] - 1][indexes[j] - 1] += stiff[i][j]

    return global_stiff