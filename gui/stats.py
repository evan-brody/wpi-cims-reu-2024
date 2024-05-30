from matplotlib import figure
import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.stats import weibull_min, rayleigh
from scipy.special import gamma
import matplotlib.pyplot as plt
import seaborn as sns

"""

   Name: weibull
   Type: function
   Description: Creates a histogram PDF plot of the weibull distrubtion based on LB/BE/UB. 
   Prints shape (k) and scale (lamda) in terminal (not present in GUI).

"""

def weibull_objective(params, values):
        k, lam = params
        # Calculate the estimated values for lower bound, geometric mean, and upper bound
        estimated_lower = weibull_min.ppf(0.05, k, scale=lam)  # 5% point probability (lower bound), given by Appendix B
        estimated_mean = lam * np.exp(gamma(1 + 1 / k) / 2)  # geometric mean of Weibull distribution
        estimated_upper = weibull_min.ppf(0.95, k, scale=lam)  # 95% point prbability (upper bound),given by Appendix B
        estimated_values = np.array([estimated_lower, estimated_mean, estimated_upper])
        # Calculate the difference between actual and estimated values
        return np.sum((values - estimated_values) ** 2)

def weibull(values):
    input = values
    
    # Set the lower bound, geometric mean, and upper bound of the failure rates
    values1 = np.array([20.8, 125.0, 4.17])  # replace with actual lower bound, geometric mean, and upper bound
    values2 = np.array([1.0, 30.0, 1000.0])
    values3 = np.array([4.17, 83.3, 417.0])
    values4 = np.array([41.7, 125.0, 375.0])
    values5 = np.array([83.3, 4170.0, 4.17])
    values6 = np.array([2.5, 33.3, 250.0])

    # Initial guess for k and lam
    initial_guess = np.array([1.0, 1.0])

    # Set bounds for k and lam
    bounds = Bounds([0.01, 0.01], [np.inf, np.inf])  # Avoid zero by setting lower bound to a small positive number

    # Perform the optimization
    result = minimize(weibull_objective, initial_guess, args=(input,), bounds=bounds)

    # Extract the optimized parameters
    k_opt, lam_opt = result.x

    # Generate a sample from the Weibull distribution with the optimized parameters
    sample = lam_opt * np.random.weibull(k_opt, 1000)

    print(f"Shape (k): {k_opt}")
    print(lam_opt)
    #print(f"Scale (λ): {lam_opt}")

    # Create a Figure and Axes object
    fig = plt.Figure(figsize=(8, 6))
    ax = fig.subplots()

    # Set x values and calculate the PDF
    x = np.linspace(np.min(sample), np.max(sample), 1000)
    pdf = weibull_min.pdf(x, k_opt, scale=lam_opt)

    # Plotting the histogram on the Axes
    sns.histplot(sample, bins=50, kde=False, color='#5f9ea0', label='Histogram', stat="density", ax=ax)

    # Set global font size
    plt.rc('font', size=18)  # controls default text sizes
    plt.rc('axes', titlesize=22)  # fontsize of the axes title
    plt.rc('axes', labelsize=20)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=18)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=18)  # fontsize of the tick labels
    plt.rc('legend', fontsize=20)  # legend fontsize
    plt.rc('figure', titlesize=24)  # fontsize of the figure title

    # Plotting the PDF on the Axes
    ax.plot(x, pdf, 'r-', label='Probability Density Function')

    ax.set_title('Motor Failure Weibull Distribution')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Probability Density')
    ax.legend()

    # Return the Figure
    return fig

"""

   Name: rayleigh
   Type: function
   Description: Creates a histogram PDF plot of the rayleigh distrubtion based on LB/BE/UB. 
   Prints scale (sigma) in terminal (not present in GUI).

"""


def rayleigh(values):

    # name = component
    input = values

    def rayleigh_objective(param, values):
        sigma = param[0]
        # Calculate the estimated values for lower bound, geometric mean, and upper bound
        estimated_lower = rayleigh.ppf(0.05, scale=sigma)
        estimated_mean = sigma * np.sqrt(np.pi / 2)  # mean of Rayleigh distribution
        estimated_upper = rayleigh.ppf(0.95, scale=sigma)
        estimated_values = np.array([estimated_lower, estimated_mean, estimated_upper])
        # Calculate the difference between actual and estimated values
        return np.sum((values - estimated_values) ** 2)

    # Set the lower bound, geometric mean, and upper bound of the failure rates
    values = np.array([1.0, 3.0, 1000.0])  # replace with actual lower bound, mean, and upper bound

    # Initial guess for sigma
    initial_guess = np.array([1.0])

    # Set bounds for sigma
    bounds = Bounds([0.01], [np.inf])  # Avoid zero by setting lower bound to a small positive number

    # Perform the optimization
    result = minimize(rayleigh_objective, initial_guess, args=(input,), bounds=bounds)

    # Extract the optimized parameter
    sigma_opt = result.x[0]

    # Generate a sample from the Rayleigh distribution with the optimized parameter
    sample = np.random.rayleigh(sigma_opt, 1000)

    print(f"Scale (σ): {sigma_opt}")

    # Plotting the histogram
    fig, ax = plt.subplots(figsize=(8, 6))
    # Set global font size
    plt.rc('font', size=18)  # controls default text sizes
    plt.rc('axes', titlesize=22)  # fontsize of the axes title
    plt.rc('axes', labelsize=20)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=18)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=18)  # fontsize of the tick labels
    plt.rc('legend', fontsize=20)  # legend fontsize
    plt.rc('figure', titlesize=24)  # fontsize of the figure title
    sns.histplot(sample, bins=50, kde=False, color='#5f9ea0', label='Histogram', stat="density", ax=ax)

    # Plotting the PDF
    x = np.linspace(np.min(sample), np.max(sample), 1000)
    pdf = rayleigh.pdf(x, scale=sigma_opt)
    ax.plot(x, pdf, 'r-', label='Probability Density Function')

    ax.set_title('Motor Failure Rayleigh Distribution')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Probability Density')
    ax.legend()

    return fig


def bathtub(N, T, t1, t2):
    # Time vector
    t = np.linspace(0, T, N)

    # Shape parameters for the three Weibull distributions
    # Assume shape parameters for "infant mortality", "normal life", and "wear-out" phases
    k1, k2, k3 = 0.5, 1, 2

    # Probability density functions
    pdf1 = weibull_min.pdf(t, k1, loc=0, scale=t1)
    pdf2 = weibull_min.pdf(t, k2, loc=0, scale=t2)
    pdf3 = weibull_min.pdf(t, k3, loc=0, scale=T)

    # Combined (summed) pdf
    pdf_combined = pdf1 + pdf2 + pdf3

    # Create new figure and add subplot
    fig = figure.Figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    # Plot the three Weibull distributions
    ax.plot(t, pdf1, label='Infant Mortality')
    ax.plot(t, pdf2, label='Useful Life')
    ax.plot(t, pdf3, label='Wear-Out')
    # Plot the combined pdf (bathtub curve)
    ax.plot(t, pdf_combined, label='Combined (Bathtub Curve)', linestyle='--')

    ax.set_title('Bathtub Curve with Weibull Distributions')
    ax.set_xlabel('Time')
    ax.set_ylabel('Failure Density')
    ax.legend()

    ax.grid(True)
    print("Reached end of bathtub")
    return fig


if __name__ == "__weibull__":
    # stuff only to run when not called via 'import' here
    weibull()

if __name__ == "__rayleigh__":
    # stuff only to run when not called via 'import' here
    rayleigh()

if __name__ == "__bathtub__":
    # stuff only to run when not called via 'import' here
    bathtub()
