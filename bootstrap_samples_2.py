#Bootstrap sampling using Numpy 
"""
In this session we build two functuions. bootstrap_replicate_1d to compute samples from the data in a 1d array
this allows us to pass in two variables, the data and a function we wish to apply to the output returned by the function.
We also calculate the length of the array passed into the function in order to understand the number of items
to return in the array. 


"""

def bootstrap_replicate_1d(data, func):
    """Generate bootstrap replicate of 1D data."""
    bs_sample = np.random.choice(data, len(data))
    return func(bs_sample)


def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)

    return bs_replicates


 # Take 10,000 bootstrap replicates of the mean: bs_replicates
bs_replicates = draw_bs_reps(rainfall,np.mean, size = 10000)

# Compute and print SEM
sem =  np.std(rainfall) / np.sqrt(rainfall)
print(sem)

# Compute and print standard deviation of bootstrap replicates
bs_std = np.std(rainfall)
print(bs_std)

# Make a histogram of the results
_ = plt.hist(bs_std, bins=50, normed=True)
_ = plt.xlabel('mean annual rainfall (mm)')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()



