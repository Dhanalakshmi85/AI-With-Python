import numpy as np

# Define the matrix A
A = np.array([
    [1, 2, 3],
    [0, 1, 4],
    [5, 6, 0]
])

# Calculate the inverse of A
A_inv = np.linalg.inv(A)

print("Inverse of A (A^-1):")
print(A_inv)

# Check A * A^-1
product1 = np.dot(A, A_inv)
print("\nA * A^-1:")
print(product1)

# Check A^-1 * A
product2 = np.dot(A_inv, A)
print("\nA^-1 * A:")
print(product2)

# Optional: round to 5 decimals to see closeness to the identity matrix
print("\nRounded checks (5 decimals):")
print("A * A^-1:\n", np.round(product1, 5))
print("A^-1 * A:\n", np.round(product2, 5))

