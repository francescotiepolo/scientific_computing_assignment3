import numpy as np
import scipy.sparse as sp
import scipy.linalg
import time

def M_matrix_square(N):
    ''' Construct the matrix M for the 2D wave equation for a square domain.
    Input:
        N: int, steps in each side.
    Output:
        M: scipy.sparse.csr_matrix, matrix M.
    '''
    diag = -4 * np.ones(N) # Diagonal of matrix A (and thus M)
    off_diag = np.ones(N - 1) # Off-diagonal of matrix A (and thus M)
    A = sp.diags([off_diag, diag, off_diag], offsets=[-1, 0, 1], shape=(N, N)) # Generate matrix A
    I = sp.eye(N) # Generate identity matrix
    M = (sp.kron(I, A) + sp.kron(A, I)).tolil() # Generate matrix M joining A's and I's
    M.setdiag(-4) # Ensure diagonal entriea are -4
    return M.tocsr()


def M_matrix_rectangle(Nx, Ny):
    ''' Construct the matrix M for the 2D wave equation for a rectangular domain.
    Input:
        Nx: int, steps in the x-axis.
        Ny: int, steps in the y-axis.
    Output:
        M: scipy.sparse.csr_matrix, matrix M.
    '''
    diag_x = -4 * np.ones(Nx) # Diagonal of matrix A_x
    off_diag_x = np.ones(Nx - 1) # Off-diagonal of matrix A_x
    A_x = sp.diags([off_diag_x, diag_x, off_diag_x], offsets=[-1, 0, 1], shape=(Nx, Nx)) # Generate matrix A_x
    
    diag_y = -4 * np.ones(Ny) # Diagonal of matrix A_y
    off_diag_y = np.ones(Ny - 1) # Off-diagonal of matrix A_y
    A_y = sp.diags([off_diag_y, diag_y, off_diag_y], offsets=[-1, 0, 1], shape=(Ny, Ny)) # Generate matrix A_y
    
    I_x = sp.eye(Nx) # Generate identity matrix for x-axis
    I_y = sp.eye(Ny) # Generate identity matrix for y-axis
    
    M = (sp.kron(I_y, A_x) + sp.kron(A_y, I_x)).tolil() # Generate matrix M joining A_x's and I_y's and viceversa
    M.setdiag(-4)  # Ensure diagonal entries are -4
    return M.tocsr()

def M_matrix_circle(N, L):
    ''' Construct the matrix M for the 2D wave equation for a circular domain.
    Input:
        N: int, steps in diameter.
    Output:
        M: scipy.sparse.csr_matrix, matrix M.
    '''
    h = L / (N + 1) # Step size
    x = np.linspace(h, L - h, N) # Grid points x-axis
    y = np.linspace(h, L - h, N) # Grid points y-axis
    X, Y = np.meshgrid(x, y)

    M = M_matrix_square(N) # Generate matrix M for square domain
    M = M.tolil() # Convert to lil format for modification

    R = L / 2 # Radius of circle
    not_circle = ((X - R)**2 + (Y - R)**2) > R**2 # Points outside circle

    for k, v in enumerate(not_circle.flatten()): # Find points outside circle and set corresponding row to zero
        if v:
            M[k, :] = 0

    M.setdiag(-4) # Ensure diagonal entries are -4
    return M.tocsr(), X, Y, not_circle

def obtain_sort_solutions(M, n=3):
    ''' Obtain the n solutions with the smallest(closest to 0) eigenvalues of the 2D wave equation.
    Input:
        M: scipy.sparse.csr_matrix, matrix M.
        n: int, number of solutions to obtain.
    Output:
        eigvals: numpy.ndarray, eigenvalues.
        eigvecs: numpy.ndarray, eigenvectors.
        lowest_sols: numpy.ndarray, indices of the n solutions with the smallest eigenvalues.
    '''
    eigvals, eigvecs = scipy.linalg.eigh(M) # Solve the eigenvalue problem
    # Sort eigenvalues/eigenvectors
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # Choose solutions with negative eigenvalues and the n closest to zero
    neg_indices = np.where(eigvals < 0)[0]
    lowest_sols = neg_indices[-n:]
    return eigvals, eigvecs, lowest_sols

def min_max_values(eigvecs, lowest_sols, Nx, Ny):
    ''' Obtain the minimum and maximum values of the solutions.
    Input:
        eigvecs: numpy.ndarray, eigenvectors.
        lowest_modes: numpy.ndarray, indices of the solutions we want to plot.
        Nx: int, steps in the x-axis.
        Ny: int, steps in the y-axis.
    Output:
        min: float, minimum value.
        max: float, maximum value.
    '''
    min_v = np.inf
    max_v = -np.inf
    for i in lowest_sols: # Loop over the solutions and find the minimum and maximum values
        sol = eigvecs[:, i].reshape(Ny, Nx)
        min_v = min(min_v, sol.min())
        max_v = max(max_v, sol.max())
    return min_v, max_v

def measure_time(N, runs=10):
    ''' Measure the time it takes to solve the eigenvalue problem for the 2D wave equation.
    Input:
        N: int, number of spatial steps.
        runs: int, number of desired runs from which compute average.
    Output:
        dict, containing the mean and confidence interval for the dense and sparse average time.
    '''
    time_dense=[]
    time_sparse=[]

    for _ in range(runs): # Loop over the number of runs
        M = M_matrix_square(N) # Generate matrix M
        M_dense = M.toarray()

        start = time.time()
        scipy.linalg.eigh(M_dense) # Solve the eigenvalue problem with dense solver
        time_dense.append(time.time() - start)

        start = time.time()
        sp.linalg.eigs(M) # Solve the eigenvalue problem with sparse solver
        time_sparse.append(time.time() - start)
    
    return {'dense mean': np.mean(time_dense),
            'dense ci': 1.96 * np.std(time_dense, ddof=1) / np.sqrt(runs),
            'sparse mean': np.mean(time_sparse),
            'sparse ci': 1.96 * np.std(time_sparse, ddof=1) / np.sqrt(runs)}

def compute_freq(N, L, k=10):
    '''Compute k largest frequencies for the 2D wave equation in a square domain.
    Input:
        N: int, number of spatial steps.
        L: float, length of the square domain.
        k: int, number of frequencies to compute.
    Output:
        numpy.ndarray, k frequencies.
    '''
    h = L / (N + 1)
    M = M_matrix_square(N) # Generate matrix M
    eigenvalues, _ = sp.linalg.eigs(M, k) # Solve the eigenvalue problem obtaining k largest eigenvalues
    frequencies = np.sqrt(np.abs(eigenvalues)) / h # Compute corresponding frequencies
    return np.sort(frequencies)