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
    x = np.linspace(0, L, N) # Grid points x-axis
    y = np.linspace(0, L, N) # Grid points y-axis
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
    eigvals, eigvecs = scipy.linalg.eigh(M) # Solve the eigenvalue problem
    # Sort eigenvalues/eigenvectors
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # Choose solutions with negative eigenvalues and the n closest to zero
    neg_indices = np.where(eigvals < 0)[0]
    lowest_sols = neg_indices[-n:]
    return eigvals, eigvecs, lowest_sols

def min_max_values(eigvecs, lowest_modes, Nx, Ny):
    global_min = np.inf
    global_max = -np.inf
    for i in lowest_modes:
        sol = eigvecs[:, i].reshape(Ny, Nx)
        global_min = min(global_min, sol.min())
        global_max = max(global_max, sol.max())
    return global_min, global_max

def measure_time(N, runs=10):
    time_dense=[]
    time_sparse=[]

    for _ in range(runs):
        M = M_matrix_square(N)
        M_dense = M.toarray()

        start = time.time()
        scipy.linalg.eigh(M_dense)
        time_dense.append(time.time() - start)

        start = time.time()
        sp.linalg.eigs(M)
        time_sparse.append(time.time() - start)
    
    return {'dense mean': np.mean(time_dense),
            'dense ci': 1.96 * np.std(time_dense, ddof=1) / np.sqrt(runs),
            'sparse mean': np.mean(time_sparse),
            'sparse ci': 1.96 * np.std(time_sparse, ddof=1) / np.sqrt(runs)}

def compute_freq(N, L, k=10):
    h = L / (N + 1)
    M = M_matrix_square(N)
    eigenvalues, _ = sp.linalg.eigs(M, k)
    frequencies = np.sqrt(np.abs(eigenvalues)) / h
    return np.sort(frequencies)