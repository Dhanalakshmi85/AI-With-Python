import numpy as np
A = np.array([[1, 2, 3],
              [0, 1, 4],
              [5, 6, 0]])
A_inv = np.linalg.inv(A)
print("Inverse of A:")
print(A_inv)
product1 = np.dot(A, A_inv)
print("\nA * A_inv:")
print(product1)
product2 = np.dot(A_inv, A)
print("\nA_inv * A:")
print(product2)
