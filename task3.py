import numpy as np
import matplotlib.pyplot as plt

# 1. Load CSV data
# Assume the CSV file has columns: "Length", "Weight" or similar headers
data = np.genfromtxt('weight-height.csv', delimiter=',', skip_header=1)

# 2. Extract lengths and weights
length_inches = data[:, 0]  # first column: lengths in inches
weight_pounds = data[:, 1]  # second column: weights in pounds

# 3. Convert units
length_cm = length_inches * 2.54       # inches → centimeters
weight_kg = weight_pounds * 0.453592  # pounds → kilograms

# 4. Calculate means
mean_length = np.mean(length_cm)
mean_weight = np.mean(weight_kg)

print(f"Mean length: {mean_length:.2f} cm")
print(f"Mean weight: {mean_weight:.2f} kg")

# 5. Draw histogram of lengths
plt.hist(length_cm, bins=10, color='skyblue', edgecolor='black')
plt.title('Histogram of Lengths of Students')
plt.xlabel('Length (cm)')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
