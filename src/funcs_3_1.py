import numpy as np
import scipy.sparse as sp

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