import matplotlib.pyplot as plt
import numpy as np
# Create x values
x = np.linspace(-10, 10, 400)

# Define the lines
y1 = 2 * x + 1
y2 = 2 * x + 2
y3 = 2 * x + 3

# Plot the lines with different colors and line styles
plt.plot(x, y1, 'r-', label='y = 2x + 1')   # red solid
plt.plot(x, y2, 'g--', label='y = 2x + 2')  # green dashed
plt.plot(x, y3, 'b-.', label='y = 2x + 3')  # blue dash-dot

# Set title and labels
plt.title('Plot of Three Lines')
plt.xlabel('x-axis')
plt.ylabel('y-axis')

# Show legend
plt.legend()

# Show grid for clarity
plt.grid(True)

# Display the plot
plt.show()
