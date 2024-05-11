import numpy as np

def LU_decomposition(A):
    n = len(A)
    L = np.eye(n)
    U = A.astype(float)  # Mengonversi tipe data matriks A ke float
    for k in range(n-1):
        for i in range(k+1, n):
            if U[k, k] == 0:
                return "Pembagian dengan nol terjadi, tidak dapat dilanjutkan"
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] -= factor * U[k, k:]
    return L, U

def solve_linear_eq_LU_gauss(A, b):
    L, U = LU_decomposition(A)
    y = np.linalg.solve(L, b)
    x = np.linalg.solve(U, y)
    return x

# Testing
A = np.array([[2, 3], [4, 5]], dtype=float)  # Mengonversi tipe data matriks A ke float
b = np.array([6, 7], dtype=float)              # Mengonversi tipe data vektor b ke float
print("Solusi dengan metode dekomposisi LU Gauss:", solve_linear_eq_LU_gauss(A, b))
