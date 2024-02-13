import numpy as np
import random

# Define initial parameters
num_samples = 500
mu = 100
sigma = 15

# Generate initial samples
samples = np.zeros(num_samples)
samples[0] = np.random.normal(0, sigma)

# Loop to generate proposal samples
for i in range(1, num_samples):
  
 # Generate proposal value
 proposal = samples[i-1] + np.random.normal(0, sigma)
  
 # Accept or reject proposal
 acceptance_prob = np.exp(-(proposal-mu)**2 / (2 * sigma**2)) / np.exp(-(samples[i-1]-mu)**2 / (2 * sigma**2))
 if (acceptance_prob > random.random()):
    samples[i] = proposal
 else:
    samples[i] = samples[i-1]

# Display samples
print(samples)