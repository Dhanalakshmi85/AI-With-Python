import numpy as np
import matplotlib.pyplot as plt

# Create the vectors
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([-0.57, -2.57, -4.80, -7.36, -8.78, -10.52, -12.85, -14.69, -16.78])

# Create the scatter plot
plt.scatter(x, y, marker='+', color='blue', s=100)  # marker '+' for points, s=100 for size

# Set title and axis labels
plt.title('Scatter Plot of Given Points')
plt.xlabel('x-axis')
plt.ylabel('y-axis')

# Show grid
plt.grid(True)
plt.show()

# Display the plot
plt.show()
