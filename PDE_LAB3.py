import numpy as np
import matplotlib.pyplot as plt


"""Numerical solver class that solves the differential equation presented in labsheet (with varying boundary conditions) 
using the FD scheme outlined in the report. Also contains error calculation functionality."""
class Solver:
    """Initialization function that assigns the following attributes to the class instance:
     self.n - the number of partitions that the interval [0,1] is split into.
     self.h - the step size length (h) of each partition on interval [0,1].
     self.X_vec - array containing n equidistant values on interval [0,1] with length = self.h.
     self.alpha - the left boundary condition (at x=0) (dirichlet OR neumann).
     self.beta - the right boundary condition (at x=1) (dirichlet OR neumann).
     self.sol_vec_dirichlet - Attribute to hold the numerical solution obtained with dirichlet boundary conditions.
     self.sol_vec_mixed - Attribute to hold the numerical solution obtained with mixed boundary conditions.
     self.error_dirichlet - Attribute to hold the infinity error as outlined in exercise 3 for the dirichlet solution.
     self.error_mixed - Attribute to hold the infinity error as outlined in exercise 3 for the mixed solution. """
    def __init__(self, n, alpha, beta):
        self.n = n
        self.h = 1/self.n
        self.X_vec = np.linspace(0,1,self.n+1)
        self.alpha = alpha
        self.beta = beta
        self.sol_vec_dirichlet = []
        self.sol_vec_mixed = []
        self.error_dirichlet = None
        self.error_mixed = None

    """The function a(x). Takes in an x-value and returns a(x)."""
    def a(self, x):
        return 1.0 + x
    """The function f(x). Takes in an x-value and returns f(x)."""
    def f(self, x):
        return 1

    """The analytical solution to the PDE with dirichlet boundary conditions, sol_exact_dirichlet. Takes in an x-value 
    and returns sol_exact_dirichlet(x). """
    def sol_exact_dirichlet(self, x):
        c1 = (self.beta - self.alpha + 1) / np.log(2)
        c2 = self.alpha
        return c1 * np.log(1+x) - x + c2 # analytical solution from report
        pass

    """The analytical solution to the PDE with mixed boundary conditions, sol_exact_mixed. Takes in an x-value and returns
    sol_exact_mixed(x). """
    def sol_exact_mixed(self, x):
        c1 = 2*self.beta + 2
        c2 = self.alpha
        return c1*np.log(1+x) - x + c2 # analytical solution from report

    """Solves for u(x) in the following differential equation: 
    -(a(x)*u'(x))' = f(x) 
    u(0) = alpha (dirichlet boundary condition)
    u(1) = beta (dirichlet boundary condition)
    Uses a(x) and f(x) as defined above and sets the attribute self.sol_vec_dirichlet containing the numerical solution of u(x) at each gridpoint. """
    def solve_dirichlet(self):
        A_vec = [self.a(i*self.h) for i in range(0, self.n+1)] # creates a vector containing a(x) called on all the gridpoints x_i


        """Construct the matrices M and F in the linear system MU = F. This will then be solved to obtain U. """
        M_matrix = np.zeros((self.n+1, self.n+1))
        F_vec = np.zeros(self.n+1)

        # Dirichlet boundary condition at left boundary.
        M_matrix[0, 0] = 1
        F_vec[0] = self.alpha

        # Build matrix M and vector F
        for i in range(1, self.n):
            temp_a_deriv = ( A_vec[i+1] - A_vec[i-1] ) / (2 * self.h) # Approximates the derivative of a(x) at current gridpoint, 2nd order.

            # fills matrix M with coefficients
            M_matrix[i][i-1] = temp_a_deriv / (2 * self.h) - A_vec[i] / (self.h**2)
            M_matrix[i][i] = 2 * A_vec[i] / (self.h**2)
            M_matrix[i][i+1] = -temp_a_deriv / (2 * self.h) - A_vec[i] / (self.h**2)

            # adds f(x) at current gridpoint to vector F
            F_vec[i] = self.f(i*self.h)

        # Dirichlet boundary condition at right boundary
        M_matrix[self.n, self.n] = 1
        F_vec[self.n] = self.beta

        # Solves the system MU = F and returns U_vec
        sol_vec_dirichlet = np.linalg.solve(M_matrix, F_vec)
        self.sol_vec_dirichlet = sol_vec_dirichlet

    """Solves for u(x) in the following differential equation: 
    -(a(x)*u'(x))' = f(x) 
    u(0) = alpha (dirichlet boundary condition)
    u'(1) = beta (neumann boundary condition)
    Uses a(x) and f(x) as defined above and sets the attribute self.sol_vec_mixed containing the numerical solution of u(x) at each gridpoint. """
    def solve_mixed(self):
        A_vec = [self.a(i * self.h) for i in
                 range(0, self.n + 1)]  # creates a vector containing a(x) called on all the gridpoints x_i

        """Construct the matrices M and F in the linear system MU = F. This will then be solved to obtain U. """
        M_matrix = np.zeros((self.n + 1, self.n + 1))
        F_vec = np.zeros(self.n + 1)

        # Mixed boundary condition at left boundary.
        M_matrix[0, 0] = 1
        F_vec[0] = self.alpha

        # Build matrix M and vector F
        for i in range(1, self.n):
            temp_a_deriv = (A_vec[i + 1] - A_vec[i - 1]) / (
                        2 * self.h)  # Approximates the derivative of a(x) at current gridpoint, 2nd order.

            # fills matrix M with coefficients
            M_matrix[i][i - 1] = temp_a_deriv / (2 * self.h) - A_vec[i] / (self.h ** 2)
            M_matrix[i][i] = 2 * A_vec[i] / (self.h ** 2)
            M_matrix[i][i + 1] = -temp_a_deriv / (2 * self.h) - A_vec[i] / (self.h ** 2)

            # adds f(x) at current gridpoint to vector F
            F_vec[i] = self.f(i * self.h)

        # Neumann boundary condition at right boundary, implemented as a left-sided 2nd order accurate FD scheme
        M_matrix[self.n, self.n] = 3/(2*self.h)
        M_matrix[self.n, self.n-1] = -2/self.h
        M_matrix[self.n, self.n-2] = 1/(2*self.h)
        F_vec[self.n] = self.beta

        # Solves the system MU = F and returns U_vec
        sol_vec_mixed = np.linalg.solve(M_matrix, F_vec)
        self.sol_vec_mixed = sol_vec_mixed

    """Calculates the error_inf as outlined in exercise 3). Sets the attributes self.error_dirichlet and self.error_mixed. 
    Should be called after calculating at least 1 numerical solution. """
    def calculate_error(self):
        if len(self.sol_vec_dirichlet) !=0 :
            sol_vec_exact_dirichlet = [self.sol_exact_dirichlet(x_i) for x_i in self.X_vec] # vector holding exact solution at every gridpoint
            temp_vec = np.array(sol_vec_exact_dirichlet) - np.array(self.sol_vec_dirichlet) # vector holding error at each gridpoint
            self.error_dirichlet = max(np.abs(temp_vec))

        if len(self.sol_vec_mixed) !=0 :
            sol_vec_exact_mixed = [self.sol_exact_mixed(x_i) for x_i in self.X_vec] # vector holding exact solution at every gridpoint
            temp_vec = np.array(sol_vec_exact_mixed) - np.array(self.sol_vec_mixed) # vector holding error at each gridpoint
            self.error_mixed = max(np.abs(temp_vec))

"""Function to be called to generate all the plots for exercise 3 in the labsheet. Saves the generated figures
as .pdf files in working directory. """
def exercise3():

    # arrays to hold errors from solvers for different values of h = 1 / N
    errors_dirichlet = []
    errors_mixed = []

    X_axis = [] # array to hold x-axis values of h = 1 / N
    Slope2 = [] # array of y = exp(2*h) values, to plot the theoretical convergence of 2nd order

    # loops through N = 2^i for i ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} and for each value N:
    # 1) executes FD solver on both problems
    # 2) calculates the error of both solutions
    # 3) stores output to be plotted
    for i in range(1, 11):
        X_axis.append(1/ (2 ** i))

        tempSolver = Solver(2**i,0,1)
        tempSolver.solve_dirichlet()
        tempSolver.solve_mixed()
        tempSolver.calculate_error()

        errors_dirichlet.append(tempSolver.error_dirichlet)
        errors_mixed.append(tempSolver.error_mixed)
        Slope2.append(np.exp(2*i))

    # creates figure Exercise3a)Dirichlet
    plt.figure()
    plt.loglog(X_axis, errors_dirichlet, color = 'blue', marker = '*', label = 'Error of numerical solution')
    plt.loglog(X_axis, np.array(X_axis) ** 2, color = 'orange', label = 'Theoretical 2nd order behavior')
    plt.title('Convergence rate for Dirichlet boundary conditions, loglog scale')
    plt.xlabel('Step length h = 1/N')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig('Exercise3a)Dirichlet.pdf', format = 'pdf')
    plt.clf()

    # creates figure Exercise3b)Mixed
    plt.figure()
    plt.loglog(X_axis, errors_dirichlet, color = 'red', marker = '*', label = 'Error of numerical solution')
    plt.loglog(X_axis, np.array(X_axis) ** 2, color = 'orange', label = 'Theoretical 2nd order behavior')
    plt.title('Convergence rate for mixed boundary conditions, loglog scale')
    plt.xlabel('Step length h = 1/N')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig('Exercise3b)mixed.pdf', format = 'pdf')
    plt.clf()

"""Function to be called to generate all the plots for exercise 1 in the labsheet. Saves the generated figures 
as .pdf files in the working directory."""
def exercise1():

    """The analytical solution to the Dirichlet problem. Take in x, alpha, and beta and return the analytical solution
    at current point, u(x). """
    def u_dirichlet(x, alpha, beta):
        c2 = alpha
        c1 = (beta - alpha + 1) / np.log(2)
        return c1 * np.log(x + 1) - x + c2

    """The analytical solution to the mixed problem. Take in x, alpha, and beta and return the analytical solution 
        at current point, u(x). """
    def u_mixed(x, alpha, beta): # Function definition
        c2 = alpha
        c1 = -2 * (beta + 1)
        return -(c1 * np.log(x + 1)) - x + c2

    # Values for alpha and beta.
    Valpha = [1.0, 2.0, 3.0]
    Vbeta = [0.2, 0.5, 0.8]

    # Define the range for x.
    x = np.linspace(0, 1, 400)

    # Create the dirichlet plot.
    plt.figure(figsize=(10, 6))

    # Plot for each alpha and beta combination.
    for alpha in Valpha:
        for beta in Vbeta:
            y = u_dirichlet(x, alpha, beta)
            plt.plot(x, y, label = f'alpha={alpha}, beta = {beta}') # PLot for each alpha and beta combination

    # Labels and legend
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Plot of u(x) for various alpha and beta values with dirichlet boundary conditions')
    plt.legend(loc="upper right")
    plt.grid(True)

    plt.savefig('Exercise1a)Dirichlet.pdf', format = 'pdf') # Save the plot to a PDF file.
    plt.clf() # Clear figure so next figure can be created.

    # Create the mixed plot.
    plt.figure(figsize=(10, 6))

    # Plot for each alpha and beta combination.
    for alpha in Valpha:
        for beta in Vbeta:
            y = u_mixed(x, alpha, beta)
            plt.plot(x, y, label=f'alpha={alpha}, beta={beta}') # Plot for each alpha and beta combination

    # Labels and legend
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Plot of u(x) for various alpha and beta values with mixed boundary conditions')
    plt.legend(loc="upper left")
    plt.grid(True)

    plt.savefig('Exercise1b)mixed.pdf', format = 'pdf') # Save the plot to a PDF file.

"""Call the main function to generate the plots for exercise 1 and exercise 3."""
def main():
    exercise1()
    exercise3()

if __name__ == '__main__':
    main()