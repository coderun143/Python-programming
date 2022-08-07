from numpy import arange
from pandas import read_csv
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import numpy as np

 
# define the true objective function
def lognorm(x, mu, sigma) :
   return 1/(np.sqrt(2*np.pi)*sigma*x)*np.exp(-((np.log(x)-mu)**2)/(2*sigma**2))

# load the dataset
data = np.loadtxt('file_1.txt')
# choose the input and output variables
x, y = data[:, 0], data[:, 1]
# curve fit
popt, _ = curve_fit(lognorm, x, y)
# summarize the parameter values
mu, sigma = popt
# plot input vs output
plt.scatter(x, y)
# define a sequence of inputs between the smallest and largest known inputs
x_line = arange(min(x), max(x), 1)
# calculate the output for the range
y_line = lognorm(x_line, mu, sigma)
# create a line plot for the mapping function
plt.plot(x_line, y_line, '--', color='red')
plt.legend(["best fit curve", "actual curve"], loc="best")
plt.show()
