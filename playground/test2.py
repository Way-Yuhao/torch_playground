import numpy as np
from tabulate import tabulate

a = np.array([[3.345324], [4.51]])
#print(tabulate(a, floatfmt=".4f"))
table = tabulate(a, floatfmt=".2f", tablefmt="pipe")
print(table)
b = 12.943413
print("{:.3f}".format(b))