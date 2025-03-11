import numpy as np
import scipy.sparse as sp

def draw_matrix(N):
    ''' Draw the matrix M for the 2D wave equation (A is the diagon matrix is M is viewed as 4 by 4 and built using A and 4 by 4 identity matrix).
    Input:
        N: int, size of the side of the spacial grid.
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