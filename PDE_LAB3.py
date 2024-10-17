import numpy as np
import matplotlib.pyplot as plt


"""Numerical solver class that solves the differential equation presented in labsheet (with varying boundary conditions) 
using the FD scheme outlien in the report. Also contains error calculation and visualization methods."""
class Solver:
    """Initialization function that assigns the following attributes:
     self.n - the number of partitions equally spaced on interval [0,1]
     self.alpha - the left boundary condition (at x=0)
     self.beta - the right boundary condition (at x=1)
     """
    def __init__(self, n, alpha, beta):
        self.n = n
        self.alpha = alpha
        self.beta = beta





def a(x=float):
    return (1.0 + x)

def f(x=float):
    return 1

def u_exact(alpha, beta, x=float):
    c2 = alpha
    c1 = (beta-alpha+1)/np.log(2) - 1
    return (c1+1)*np.log(1+x) - x + c2

def error_inf(X_vec, U_numeric):
    U_exact_vector = [u_exact(i) for i in X_vec]
    return np.max(np.abs(np.array(U_exact_vector) - np.array(U_numeric)))

def solve(N, alpha, beta):
    h = 1 / N
    A_vec = [a(i*h) for i in range(0, N+1)] # creates a vector of a(x) called on all the gridpoints x_i

    """ Trying to solve MU = F """

    M_Matrix = np.zeros((N + 1, N + 1))
    F_vec = np.zeros(N+1)

    # Dirichlet BC at left boundary
    M_Matrix[0][0] = 1
    F_vec[0] = alpha

    """For loop that will build up the matrix M and also the vector F."""
    for i in range(1,N):
        a_deriv_approx = ( A_vec[i+1] - A_vec[i-1] ) / ( 2*h ) # Approximates the derivative of a(x) at current gridpoint, 2nd order

        # creating elements in M_Matrix
        M_Matrix[i][i-1] = a_deriv_approx / (2*h) - A_vec[i] / (h**2)
        M_Matrix[i][i] = 2 * A_vec[i] / (h**2)
        M_Matrix[i][i+1] = -a_deriv_approx / (2*h) - A_vec[i] / (h**2)

        # inserting f(x) at current gridpoint in F_vec
        F_vec[i] = f(i*h)

    # Dirichlet BC at right boundary
    M_Matrix[N][N] = 1
    F_vec[N] = beta

    # Solves the system MU = F
    U_vec = np.linalg.solve(M_Matrix,F_vec)

    return U_vec

def visualize(U_vec, X_vec):

    pass

def main():
    alpha = 0
    beta = 1
    N = 4
    h = 1 / N
    X_vec = np.linspace(0, 1, N + 1)

    U_exact = [u_exact(alpha, beta, i*h) for i in range(0, N+1)]

    U_vec = solve(N, alpha, beta)

    print(U_exact)
    print(U_vec)

    plt.figure()
    plt.plot(X_vec, U_vec, '*')
    plt.plot(X_vec, U_exact, '-')
    plt.show()



if __name__ == '__main__':
    main()