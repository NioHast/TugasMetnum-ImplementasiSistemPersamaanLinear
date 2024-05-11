import numpy as np

def crout_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    
    for j in range(n):
        U[j, j] = 1
        for i in range(j, n):
            L[i, j] = A[i, j] - np.dot(L[i, :j], U[:j, j])
        for i in range(j+1, n):
            U[j, i] = (A[j, i] - np.dot(L[j, :j], U[:j, i])) / L[j, j]
    return L, U

def solve_linear_eq_crout(A, b):
    L, U = crout_decomposition(A)
    y = np.linalg.solve(L, b)
    x = np.linalg.solve(U, y)
    return x

# Testing
A = np.array([[1, 2], [3, 4]])
b = np.array([5, 6])
print("Solusi dengan metode dekomposisi Crout:", solve_linear_eq_crout(A, b))