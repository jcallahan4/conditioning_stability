# condition_stability.py
"""Volume 1: Conditioning and Stability.
<Name>
<Class>
<Date>
"""

import numpy as np
import sympy as sy
import scipy.linalg as la
from matplotlib import pyplot as plt


# Problem 1
def matrix_cond(A):
    """Calculate the condition number of A with respect to the 2-norm."""
    #Get singular values of A
    sing_vals = la.svdvals(A)
    #Get min and max singular values
    sing_min, sing_max = np.min(sing_vals), np.max(sing_vals)
    #Check for singularity
    if np.abs(sing_min) == 0:
        return np.inf
    #Return condition number
    return sing_max/sing_min


# Problem 2
def prob2():
    """Randomly perturb the coefficients of the Wilkinson polynomial by
    replacing each coefficient c_i with c_i*r_i, where r_i is drawn from a
    normal distribution centered at 1 with standard deviation 1e-10.
    Plot the roots of 100 such experiments in a single figure, along with the
    roots of the unperturbed polynomial w(x).

    Returns:
        (float) The average absolute condition number.
        (float) The average relative condition number.
    """
    w_roots = np.arange(1, 21)

    # Get the exact Wilkinson polynomial coefficients using SymPy.
    x, i = sy.symbols('x i')
    w = sy.poly_from_expr(sy.product(x-i, (i, 1, 20)))[0]
    w_coeffs = np.array(w.all_coeffs())

    abs_cond = []
    rel_cond = []

    for n in range(100):
        #Get h to perturb coefficients
        h = np.random.normal(1,1e-10,21)

        #Get new coefficients and new roots
        new_coeffs = w_coeffs*h
        new_roots = np.roots(np.poly1d(new_coeffs))

        #Plot new new roots
        plt.scatter(new_roots.real, new_roots.imag, s=1, c='k', marker=',')

        # Sort the roots to ensure that they are in the same order.
        w_roots = np.sort(w_roots)
        new_roots = np.sort(new_roots)
        # Estimate the absolute condition number in the infinity norm.
        k = la.norm(new_roots - w_roots, np.inf) / la.norm(h, np.inf)
        abs_cond.append(k)
        # Estimate the relative condition number in the infinity norm.
        rel_cond.append(k * la.norm(w_coeffs, np.inf) / la.norm(w_roots, np.inf))


    #Compute and plot final trial
    h = np.random.normal(1,1e-10,21)
    new_coeffs = w_coeffs*h
    new_roots = np.roots(np.poly1d(new_coeffs))
    plt.scatter(new_roots.real, new_roots.imag, s=1, c='k', marker=',', label='Perturbed roots')


    #Plot new new roots
    plt.scatter(w_roots.real, w_roots.imag, label='Wilkinson roots')
    plt.legend()
    plt.ylabel("Imaginary Axis")
    plt.xlabel("Real Axis")
    plt.title('Wilkinson roots vs Perturbed roots')
    plt.show()

    return np.mean(abs_cond), np.mean(rel_cond)
# Problem 3
def eig_cond(A):
    """Approximate the condition numbers of the eigenvalue problem at A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) The absolute condition number of the eigenvalue problem at A.
        (float) The relative condition number of the eigenvalue problem at A.
    """
    #Construct agitation matrix
    reals = np.random.normal(0, 1e-10, A.shape)
    imags = np.random.normal(0, 1e-10, A.shape)
    H = reals + 1j*imags

    #Get eigenvalues of A and A+H
    lam = la.eigvals(A)
    lam_hat = la.eigvals(A+H)
    #Calculate absolute condition number
    k_hat = la.norm(lam - lam_hat, 2) / la.norm(H, 2)
    #Calculate relative condition number
    k = (la.norm(A, 2) / la.norm(lam, 2)) * k_hat

    return k_hat, k

# Problem 4
def prob4(domain=[-100, 100, -100, 100], res=50):
    """Create a grid [x_min, x_max] x [y_min, y_max] with the given resolution. For each
    entry (x,y) in the grid, find the relative condition number of the
    eigenvalue problem, using the matrix   [[1, x], [y, 1]]  as the input.
    Use plt.pcolormesh() to plot the condition number over the entire grid.

    Parameters:
        domain ([x_min, x_max, y_min, y_max]):
        res (int): number of points along each edge of the grid.
    """
    #Create tuples for creating matrix
    xy = np.mgrid[domain[0]:domain[1]:np.abs(domain[0]-domain[1])/res,
                    domain[2]:domain[3]:np.abs(domain[2]-domain[3])/res].reshape(2,-1).T
    #Initialize list to store relative condition numbers
    rel_cond = []
    #Create matrices, get condition number, and store it
    for vals in xy:
        mat = np.array([[1,vals[0]],[vals[1],1]])
        rel_cond.append(eig_cond(mat)[1])
    #Create domains to graph on and reshape rel_cond for pcolormesh
    X,Y = np.linspace(domain[0], domain[1], res), np.linspace(domain[2], domain[3], res)
    rel_cond = np.array(rel_cond).reshape((res,res))
    #Plot with colormesh
    plt.pcolormesh(X, Y, rel_cond, cmap = 'gray_r')
    plt.colorbar()
    plt.show()

# Problem 5
def prob5(n):
    """Approximate the data from "stability_data.npy" on the interval [0,1]
    with a least squares polynomial of degree n. Solve the least squares
    problem using the normal equation and the QR decomposition, then compare
    the two solutions by plotting them together with the data. Return
    the mean squared error of both solutions, ||Ax-b||_2.

    Parameters:
        n (int): The degree of the polynomial to be used in the approximation.

    Returns:
        (float): The forward error using the normal equations.
        (float): The forward error using the QR decomposition.
    """
    #Read in data and create A
    xk, yk = np.load("stability_data.npy").T
    A = np.vander(xk, n+1)
    #Get inverse using normal equations
    inv_method = la.inv(A.T @ A) @ A.T@yk
    #Get inverse using QR method
    Q,R = la.qr(A, mode = 'economic')
    qr_method = la.solve_triangular(R, Q.T@yk)

    #Plot values against each other
    domain = np.linspace(0, 1, 200)
    plt.scatter(xk, yk, marker='.')
    plt.plot(xk, np.polyval(qr_method, xk),label='QR Method')
    plt.plot(xk, np.polyval(inv_method, xk),label='Normal Equations')

    #Show plot
    plt.legend(loc='upper left')
    plt.title('Compare least squares solutions')
    plt.xlabel('Domain')
    plt.ylabel('Range')
    plt.show()




# Problem 6
def prob6():
    """For n = 5, 10, ..., 50, compute the integral I(n) using SymPy (the
    true values) and the subfactorial formula (may or may not be correct).
    Plot the relative forward error of the subfactorial formula for each
    value of n. Use a log scale for the y-axis.
    """
    #Initialize values
    x = sy.symbols('x')
    exact_list = []
    approx_list = []
    #Calculate approximate value and exact value for each n
    for n in range(5,55,5):
        #Exact value
        exact = (-1)**n * (sy.subfactorial(n) - sy.factorial(n) / np.exp(1))
        exact_list.append(exact)

        #Approximate value
        eq = x**n * sy.exp(x-1)
        approximated = sy.integrate(eq, (x,0,1))
        approx_list.append(approximated)
    #Cast to arrays
    exact_list = np.array(exact_list)
    approx_list = np.array(approx_list)
    #Get relative error
    rel_error = np.abs(exact_list - approx_list)
    #Plot relative error
    plt.semilogy(np.linspace(5,50,10), rel_error)
    plt.title('Relative forward error of I(n)')
    plt.xlabel("n")
    plt.ylabel('Error')
    plt.show()
