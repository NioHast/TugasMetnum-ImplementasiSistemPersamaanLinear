import numpy as np

def solve_linear_eq_matrix_inverse(A, b):
    if np.linalg.det(A) == 0:
        return "Matriks A singular, tidak dapat diinvers"
    
    A_inv = np.linalg.inv(A)
    x = np.dot(A_inv, b)
    return x

# Testing
A = np.array([[1, 2], [3, 4]])
b = np.array([5, 6])
print("Solusi dengan metode matriks balikan:", solve_linear_eq_matrix_inverse(A, b))