import numpy as np


def get_details(element, nodes):
    n1 = element[0]
    n2 = element[1]
    E = element[2]
    I = element[3]
    A = element[4]
    x1 = nodes[n1][0]
    y1 = nodes[n1][1]
    x2 = nodes[n2][0]
    y2 = nodes[n2][1]
    L = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    lambda_x = (x2 - x1) / L
    lambda_y = (y2 - y1) / L
    dofs = [3 * n1, 3 * n1 + 1, 3 * n1 + 2, 3 * n2, 3 * n2 + 1, 3 * n2 + 2]
    return n1, n2, E, I, A, x1, y1, x2, y2, L, lambda_x, lambda_y, dofs


def mem_stiff(element, nodes):
    n1, n2, E, I, A, x1, y1, x2, y2, L, lambda_x, lambda_y, dofs = get_details(element, nodes)
    transform_matrix = np.array([
        [lambda_x, lambda_y, 0, 0, 0, 0],
        [-lambda_y, lambda_x, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, lambda_x, lambda_y, 0],
        [0, 0, 0, -lambda_y, lambda_x, 0],
        [0, 0, 0, 0, 0, 1]
    ])

    stiff = np.array([
        [(A * E) / L, 0, 0, -((A * E) / L), 0, 0],
        [0, (12 * (E * I)) / (L ** 3), (6 * (E * I)) / (L ** 2), 0, -(12 * (E * I)) / (L ** 3), (6 * (E * I)) / (L ** 2)],
        [0, (6 * (E * I)) / (L ** 2), (4 * (E * I)) / L, 0, -(6 * (E * I)) / (L ** 2), (2 * (E * I)) / L],
        [-((A * E) / L), 0, 0, (A * E) / L, 0, 0],
        [0, -(12 * (E * I)) / (L ** 3), -(6 * (E * I)) / (L ** 2), 0, (12 * (E * I)) / (L ** 3), -(6 * (E * I)) / (L ** 2)],
        [0, (6 * (E * I)) / (L ** 2), (2 * (E * I)) / L, 0, -(6 * (E * I)) / (L ** 2), (4 * (E * I)) / L]
    ])
    stiff = np.dot(np.dot(transform_matrix.T, stiff), transform_matrix)
    return stiff


def glob_stiff(elements, nodes):
    k = np.zeros((18, 18))
    for element in elements:
        n1, n2, E, I, A, x1, y1, x2, y2, L, lambda_x, lambda_y, dofs = get_details(element, nodes)
        member_stiffness = mem_stiff(element, nodes)
        for i in range(6):
            for j in range(6):
                k[dofs[i]][dofs[j]] += member_stiffness[i][j]
    return k


def modified_stiff(elements, nodes, boundary_conditions, user_dofs, allow_axial_deformations=False, sub_axis=True):
    if allow_axial_deformations:
        return glob_stiff(elements, nodes)
    else:
        stiff = glob_stiff(elements, nodes)
        n = stiff.shape[0]

        m = 3 * len(nodes)
        # we only want the degrees of freedom that are not fixed or coupled
        m -= len(boundary_conditions)

        for element in elements:
            n1, n2, E, I, A, x1, y1, x2, y2, L, lambda_x, lambda_y, dofs = get_details(element, nodes)
            if lambda_x == 1 and sub_axis == True:
                m -= 1

        mod_stiff = np.zeros((m, m))

        for i in range(n):
            if i in user_dofs.keys() and i not in boundary_conditions:
                for j in range(n):
                    if j in user_dofs.keys() and j not in boundary_conditions:
                        try:
                            mod_stiff[user_dofs[i]][user_dofs[j]] += stiff[i][j]
                        except KeyError:
                            print("Boundary conditions might be incorrect")

        return mod_stiff


def transform_matrix(k, free_user_dofs):
    n = k.shape[0]
    temp = np.zeros((n, n))
    for i in free_user_dofs.keys():
        for j in free_user_dofs.keys():
            temp[free_user_dofs[i]][free_user_dofs[j]] = k[i][j]

        for l in range(n):
            if l not in free_user_dofs.keys():
                temp[free_user_dofs[i]][l] = k[i][l]
    for i in range(n):
        if i not in free_user_dofs.keys():
            for j in free_user_dofs.keys():
                temp[i][free_user_dofs[j]] = k[i][j]

            for l in range(n):
                if l not in free_user_dofs.keys():
                    temp[i][l] = k[i][l]
    return temp


def static_condensation(k, zero_mass):
    n = k.shape[0]
    t_size = n - zero_mass

    k_tt = k[0: t_size, 0: t_size]
    k_t0 = k[0: t_size, t_size: n]
    k_0t = k[t_size: n, 0: t_size]
    k_00 = k[t_size: n, t_size: n]

    k_tt_ = k_tt - ((k_t0 @ (np.linalg.inv(k_00))) @ k_0t)
    return k_tt_
