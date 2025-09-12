import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
file_path = '/home/dhanaarun/Documents/AIpython/CSV Files/mydata.csv'
df = pd.read_csv(file_path)
print(df.head())
data = df.to_numpy()
length = data[:, 0]
weight = data[:, 1]
length_cm = length * 2.54
weight_kg = weight * 0.453592
print("Mean length (cm):", np.mean(length_cm))
print("Mean weight (kg):", np.mean(weight_kg))
plt.hist(length_cm, bins=10, edgecolor='black')
plt.title("Histogram of Student Lengths")
plt.xlabel("Length (cm)")
plt.ylabel("Number of Students")
plt.show()
