import numpy as np
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt

MAX_CAPACITY = 8
N_ITEMS = 5

table = pd.read_excel('table.xlsx').to_numpy()

wheigts = table[0]
utilities = table[1]

print('Weights:', wheigts)
print('Utilities:', utilities)

"""
z(i, w) = max(z(i-1, w), z(i-1, w-wi) + ui)
z(i, 0) = 0
z(0, w) = 0
"""

def knapsack(wheigts, utilities, max_capacity):
    """
    Get the z matrix of the knapsack problem based on the weights and utilities of the items
    """
    n = len(wheigts)
    z = np.zeros((n, max_capacity + 1))
    for i in range(1, n):
        for w in range(1, max_capacity + 1):
            if wheigts[i] <= w:
                print("z[{}, {}] = max(z[{}, {}], z[{}, {}] + {})".format(i, w, i-1, w, i-1, w-wheigts[i], utilities[i]))
                calculus = [z[i-1, w], z[i-1, w-wheigts[i]] + utilities[i]]
                argmax = np.argmax(calculus)

                z[i, w] = calculus[argmax]

                if argmax == 1:
                    print("Item {} added".format(i))
                else:
                    print("Item {} not added".format(i))

            else:
                z[i, w] = z[i-1, w]

    return z

def get_items(z, wheigts, utilities, max_capacity):
    """
    Get the items that are in the knapsack based on the z matrix
    """
    n = len(wheigts)
    w = max_capacity
    items = []
    for i in range(n-1, 0, -1):
        if z[i, w] != z[i-1, w]:
            items.append(i)
            w -= wheigts[i]

    items = [i + 1 for i in items]
    return items



z = knapsack(wheigts, utilities, MAX_CAPACITY)
print(z)
items = get_items(z, wheigts, utilities, MAX_CAPACITY)

print("Items:", items)

print("Total utility:", z[-1, -1])

# state diagram
plt.figure(figsize=(12, 8))
plt.imshow(z, cmap='viridis')
plt.colorbar()
plt.xlabel('Capacity')
plt.ylabel('Item')
plt.title('Knapsack problem')
plt.savefig('state_diagram.png')
