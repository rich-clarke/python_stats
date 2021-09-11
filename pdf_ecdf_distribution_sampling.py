
"""{-------------------------------}"""
#    Title: ECDF - DataFrame Script
#    Idea: 'Everything starts with EDA'
#    Auth: Rich Clarke
#    Date: 27/08/21


#  imports 

import pandas as pd
import numpy as np
import plotly as plty

# config

pd.options.plotting.backend = "plotly"

def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y

# Display the plot
plt.show()

# Compute ECDFs

x_set, y_set = ecdf(setosa_petal_length)

x_vers, y_vers = ecdf(versicolor_petal_length)

x_virg, y_virg = ecdf(versicolor_petal_length)

# Plot all ECDFs on the same plot
_ = plt.plot(x_set, y_set, marker='.', linestyle='none')
_ = plt.plot(x_vers, y_vers, marker='.', linestyle='none')
_ = plt.plot(x_virg, y_virg, marker='.', linestyle='none')

# Annotate the plot
_ = plt.legend(('setosa', 'versicolor', 'virginica'), loc='lower right')
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')

# Plot

plt.show() 


""""


With a few lines of python, you can compute strong statistical
analysis and 'tell the whole story'

"""

# ECDF function is already built here we are looking to compare
# pecentiules of Iris versicolor petal lengths that I calculated in the 
# last exercise so that we. an see how percentiles relate to ECDF 

# Plot the ECDF

_ = plt.plot(x_vers, y_vers, '.')
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')

# Overlay percentiles as red diamonds.
_ = plt.plot(ptiles_vers, percentiles/100, marker='D', color='red',linestyle = 'none')

# Show the plot

plt.show()



# Create box plot with Seaborn's default settings

_ = sns.boxplot(x= df['species'], y = df['petal length (cm)'], data = df)

# Label the axes

_ = plt.xlabel('spcies')
_ = plt.ylabel('petal length (cm)')

# Show the plot

plt.show()


""" 

What other stats can we calculate? 


Variance and Standard Deviation? 

-- Florida seems to have more county to county
varaibility that Ohio. 

Variance: 

The mean squared distance of the data from their mean
Informally a measure of the spread of data.


- - Square the distance from the mean 

C


"""

np.sqrt(.np.var(dem_share_FL))

Results are the same as taking the sqrt of the variance 


std deviation is a reasonable
metric to get spread. 



# Array of differences to mean: differences
differences = versicolor_petal_length - np.mean(versicolor_petal_length)

# Square the differences: diff_sq
diff_sq = differences**2

# Compute the mean square difference: variance_explicit
variance_explicit = np.mean(diff_sq)

# Compute the variance using NumPy: variance_np
variance_np = np.var(versicolor_petal_length)

# Print the results
print(variance_explicit, variance_np)



# Compute the variance: variance

variance = np.var(versicolor_petal_length)

# Print the square root of the variance

print(np.sqrt(variance))


# Print the standard deviation

print(np.std(versicolor_petal_length))



#Next Exercise. 

# Make a scatter plot

_ = plt.plot(versicolor_petal_length, versicolor_petal_width, marker = '.', linestyle = 'none')

# Label the axes

_ = plt.xlabel('Verisocolor Petal Length')
_ = plt.ylabel('Versicolor Petal Width')


# Show the result

plt.show()



# Covariance Calculation


""""

Mean of X
Mean of Y

Then look for the distance from 
the mean of X and the mean of Y

Respective means together(if x and y are above)
and posotively correlated it means
they have a posotive covariance. 



# Compute the covariance matrix: covariance_matrix

covariance_matrix = np.cov(versicolor_petal_length, versicolor_petal_width)


# Print covariance matrix

print(covariance_matrix)

# Extract covariance of length and width of petals: petal_cov

petal_cov = covariance_matrix[0,1]


# Print the length/width covariance

print(petal_cov)


If we want a measure this withotu units
We can compute using numpy functions
however if we want more applicable measure of this
we want this to be diemnsionsless.


So we can divide the covariance / (std of x) (std of y) 

It is the variability due to codependance / independant varaibility(std deviation)

-1 anti correlation, +1 posotive correlation. 

""" 

def pearson_r(x, y): 
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat = np.corrcoef(x, y)

    # Return entry [0,1]
    return corr_mat[0,1]

# Compute Pearson correlation coefficient for I. versicolor: r

r = pearson_r(versicolor_petal_length, versicolor_petal_width)

# Print the result

print(r)


"""

Probalistic Thinking.

Statistical thinking is the core of this.

We go from probalistic data to thinking through the language of probability.

In this session we will start to learn the way to speak in this type of inference. 

Project: 

Generating a probability that all four coins will land on heads. 

< 0.5 - Heads
=> 0.5 tails 

Using numpy; np.randomn.random() 

This is called a bernoulli trial. 

Bernoulli: Where the result is either true or false is a bernoulli trial. It has only two options.

Integer fed into random number gen algo. Manually seed random num gen if you need reproduction. Specified using np.random.seed(3) for example. 

To do our coin flips.

"""

import numpy as np 

np.random.seed(42)
random_numbers = np.random.random(size=4) 

heads = random_numbers < 0.5
heads
np.sum(heads)
"""


Summing array as bools in the array 

If we were to repeat this we want to attain the probability. 
"""

n_all_heads = 0 # int number of 4-heads trials

for _ in range (10000):
	heads = np.randon.random(size=4) < 0.5
	n_heads = np.sum(heads) 
	if n_heads == 4: 
		n_all_heads += 1  #adds to count each time the result is all heads. 

		n_all_heads / 1000 
		#gets the average number of heads in all of the trials 


#with hacker statistics


# Seed the random number generator

np.random.seed(42)


# Initialize random numbers: random_numbers
random_numbers = np.empty(100000)

# Generate random numbers by looping over range(100000)
for i in range(100000):
    random_numbers[i] = np.random.random()

# Plot a histogram
_ = plt.hist(random_numbers)

# Show the plot
plt.show()


#Bernoulli Trial func and iteration

def perform_bernoulli_trials(n, p):
    """Perform n Bernoulli trials with success probability p
    and return number of successes."""
    # Initialize number of successes: n_success
    n_success = 0
    # Perform trials
    # probability of p = success and probability of (1 - p ) for tails.
    for i in range(n):
        # Choose random number between zero and one: random_number
        random_number = np.random.random()
        # If less than p, it's a success so add one to n_success
        if random_number < p:
            n_success += 1
    return n_success


#Using the function to generate, for 1000 iterations, out of 100 trials
#how many loans default(==True) 


#STATS: Probability Distributions 
""" 

* Probability Mass Function - the set of probabilities of discrete
  outcomes. 

* Discrete Unifrom (PMF):
	- Only a set of discrete values(pre defined, like each side of a dice.) 
	- The 'uniformity' of a distribution is if the p value(probability)
	  is even in each chance. 

	- The PMF is a property of a distribution.
* A distribution is a mathametaical description of outcomes(in probvility)

* A binomial distribution is as follows

the number r of successes in n bernoulli trials with a probability p of succes, is Binomully distrbuted

The number of result(r) of heads in n(4) with probablity of 0.5 of heads is binomially distributed. 

np.random.binomial(4, 0.5) = 2

np.random.binomial(4, 0.5, size =10)




"""

# PMF - Binomial Distribution plotted using a ECDF

# Take 10,000 samples out of the binomial distribution: n_defaults

n_defaults = np.random.binomial(n = 100, p= 0.05, size = 10000)

# Compute CDF: x, y
x, y = ecdf(n_defaults)

# Plot the CDF with axis labels

_ = plt.plot(x, y, marker = '.', linestyle='none')

_ = plt.xlabel('Number of Defaults (in each 100 Loans)')

_ = plt.ylabel('CDF Function')

# Show the plot

plt.show() 

# Draw 10,000 samples out of Poisson distribution: samples_poisson
samples_poisson = np.random.poisson(10, size=10000)

# Print the mean and standard deviation
print('Poisson:     ', np.mean(samples_poisson),
                       np.std(samples_poisson))

# Specify values of n and p to consider for Binomial: n, p
n = [20, 100, 1000]
p = [0.5, 0.1, 0.01]

# Draw 10,000 samples for each n,p pair: samples_binomial
for i in range(3):
    samples_binomial = np.random.binomial(n[i], p[i], size=10000)

    # Print results
    print('n =', n[i], 'Binom:', np.mean(samples_binomial),
                                 np.std(samples_binomial))



