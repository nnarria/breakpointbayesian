# A Bayesian Change Point Analysis of the USD/CLP Series in Chile from 2018-2020: Understanding the Impact of Social Protests and the COVID-19 Pandemic

In this repository it is available the Python implementation for the detection of change points with functional part according to De la Cruz, R. et al. (2022) ‘A Bayesian Change Point Analysis of the USD/CLP Series in Chile from 2018-2020: Understanding the Impact of Social Protests and the COVID-19 Pandemic’, submitted to Mathematics. Applied to a simulated series of 100 values, with three change points, and a dictionary of 123 functions with: a constant, 100 Haar functions, 20 Fourier functions, one linear and one quadratic.

Example series with three change points.

![Simulated series](https://github.com/nnarria/breakpointbayesian/blob/main/images/simulated_serie_0.png)


Result, breakpoints selected.

![Breakpoints selected](https://github.com/nnarria/breakpointbayesian/blob/main/images/resMH_0.png)

To apply our method to this simulated series, execute the codes located in folder 'src' in this orden:

1. Run the auxiliary functions: "dpc_segmentation_functional_effect.py".
2. Run the auxiliary functions: "dpc_util_plot.py".
3. Run the main function: "dpc_bayesian.py".
