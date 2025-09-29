import numpy as np
import matplotlib.pyplot as plt
n_values = [500, 1000, 2000, 5000, 10000]
for n in n_values:
    dice1 = np.random.randint(1, 7, n)
    dice2 = np.random.randint(1, 7, n)
    sums = dice1 + dice2
    counts, bins = np.histogram(sums, range(2, 14))
    plt.bar(bins[:-1], counts / n)   # frequency = counts / total
    plt.title("Dice sums with n=" + str(n))
    plt.xlabel("Sum of two dice")
    plt.ylabel("Frequency")
    plt.show()
