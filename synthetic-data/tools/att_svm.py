import cvxpy as cp


def att_svm_solver(X, z, alpha, p):
    """
    Solve the Att-SVM problem given the input sequences X, cross inputs
    z, the optimal token indices alpha, using the l_p objective
    """
    # Get the dimensions
    n, T, d = X.shape
    W = cp.Variable((d, d))

    # Get the constraints
    constraints = []
    for i in range(n):
        for j in range(T):
            if j == alpha[i]:
                continue
            constraints += [
                ((X[i, j] - X[i, alpha[i]]).reshape(1, -1) @ W @ (z[i]).reshape(-1, 1))
                <= -1
            ]

    # Solve the Att-SVM
    prob = cp.Problem(cp.Minimize(cp.pnorm(W, p)), constraints)
    prob.solve()

    return W.value


def joint_wv_att_svm_solver(X, Y, z, alpha, p):
    # Get the dimensions
    n, T, d = X.shape
    W = cp.Variable((d, d))
    v = cp.Variable((d))

    # Get the constraints
    W_constraints = []
    for i in range(n):
        for j in range(T):
            if j == alpha[i]:
                continue
            W_constraints += [
                ((X[i, j] - X[i, alpha[i]]).reshape(1, -1) @ W @ (z[i]).reshape(-1, 1))
                <= -1
            ]
    v_constraints = []
    for i in range(n):
        v_constraints += [Y[i] * (X[i, alpha[i]].reshape(1, -1) @ v) >= 1]

    # Solve the Att-SVM
    probW = cp.Problem(cp.Minimize(cp.pnorm(W, p)), W_constraints)
    probW.solve()
    probv = cp.Problem(cp.Minimize(cp.pnorm(v, p)), v_constraints)
    probv.solve()

    return W.value, v.value


def att_svm_solver_nuc(X, z, alpha):
    """
    Solve the Att-SVM problem given the input sequences X, cross inputs
    z, the optimal token indices alpha, using the nuclear objective
    """
    # Get the dimensions
    n, T, d = X.shape
    W = cp.Variable((d, d))

    # Get the constraints
    constraints = []
    for i in range(n):
        for j in range(T):
            if j == alpha[i]:
                continue
            constraints += [
                ((X[i, j] - X[i, alpha[i]]).reshape(1, -1) @ W @ (z[i]).reshape(-1, 1))
                <= -1
            ]

    # Solve the Att-SVM
    prob = cp.Problem(cp.Minimize(cp.norm(W, "nuc")), constraints)
    prob.solve()

    return W.value
