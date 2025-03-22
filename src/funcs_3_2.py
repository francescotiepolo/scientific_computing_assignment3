import numpy as np
import scipy.sparse as sp

def M_matrix_square(N):
    diag = -4 * np.ones(N)  
    off_diag = np.ones(N - 1)  
    A = sp.diags([off_diag, diag, off_diag], offsets=[-1, 0, 1], shape=(N, N)) 
    I = sp.eye(N)  
    M = sp.kron(sp.eye(N), A) + sp.kron(A, sp.eye(N))  
    M = M.tocsr()
    M.setdiag(-4)  
    return M


def M_matrix_disk(N, L):
    h = L / (N + 1)  # Grid spacing
    x = np.linspace(-L/2 + h, L/2 - h, N)  # X centered at 0
    y = np.linspace(-L/2 + h, L/2 - h, N)  # Y centered at 0
    X, Y = np.meshgrid(x, y)

    M = M_matrix_square(N)  
    M = M.tolil() 

    R = L / 2  # Radius 
    # Mask points outside the circle
    mask = X**2 + Y**2 > R**2 

    for i in range(N):
        for j in range(N):
            idx = i * N + j  
            if mask[i, j]: 
                M[idx, :] = 0  
                M[idx, idx] = 1  # Set diagonal to 1 so matrix invertible

    b = np.zeros(N * N)

    source_x, source_y = 0.6, 1.2  # Source 
    # Adjust source coordinates 
    source_idx = np.argmin((X - source_x) ** 2 + (Y - source_y) ** 2)  
    M[source_idx, :] = 0  
    M[source_idx, source_idx] = 1  
    b[source_idx] = 1  # source value = 1

    return M.tocsr(), X, Y, mask, b

