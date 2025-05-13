import numpy as np 
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from imblearn.over_sampling import SMOTE
import torch
from torch import nn
from tqdm.auto import tqdm
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd 
# from google.colab import drive
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.utils import shuffle
import pickle
from sklearn.metrics import precision_recall_curve, auc

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def MCMC(X_train, y_train, X):
    # Identify minority and majority class labels
    class_counts = Counter(y_train)
    minority_class_label = min(class_counts, key=class_counts.get)
    majority_class_label = max(class_counts, key=class_counts.get)
    
    # Calculate the number of synthetic samples needed
    num_samples_to_generate = np.count_nonzero(y_train == majority_class_label) - np.count_nonzero(y_train == minority_class_label)
    burn_in_needed = int(0.2 * num_samples_to_generate)
    num_samples_to_generate += burn_in_needed
    
    # Initialize mean and standard deviation for MCMC
    mean = X_train[y_train == minority_class_label].mean(axis=0)
    std = X_train[y_train == minority_class_label].std(axis=0)

    # Generate walkers using MCMC
    walkers = np.random.normal(mean, std, (num_samples_to_generate, X.shape[1]))

    for i in range(1, num_samples_to_generate):
        new_mean = np.random.normal(mean, std)
        likelihood_new = np.exp(-np.sum((X - new_mean) ** 2, axis=1) / (2 * np.sum(std ** 2)))
        likelihood_current = np.exp(-np.sum((X - mean) ** 2, axis=1) / (2 * np.sum(std ** 2)))
        acceptance_probability = np.minimum(1, likelihood_new / likelihood_current)
        
        if acceptance_probability.any() >= np.random.rand():
            mean = new_mean

    # Extract post-burn-in walkers
    post_burn_in_walkers = walkers[burn_in_needed:]
    
    # Generate synthetic data
    synthetic_data = np.random.normal(post_burn_in_walkers, std, size=(num_samples_to_generate - burn_in_needed, X.shape[1]))

    # Concatenate synthetic data with original data
    concatenated_Xs = np.concatenate((X_train, synthetic_data), axis=0)
    concatenated_Ys = np.concatenate((y_train, np.ones(num_samples_to_generate - burn_in_needed)), axis=0)

    # Shuffle the combined dataset
    X_MCMC, y_MCMC = shuffle(concatenated_Xs, concatenated_Ys)


    return synthetic_data, X_MCMC, y_MCMC


# Define the model outside the loop
F1_model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1)
])

F1_model.compile(optimizer='adam',
              loss='mean_absolute_error',
              metrics=['accuracy'])

def callf1(xx, yy, xt, yt, ep):    
    F1_model.fit(xx, yy, epochs=ep, verbose=0)
    # Evaluation on test set
    test_loss, test_acc = F1_model.evaluate(xt, yt, verbose=0)
    tr_loss, tr_acc = F1_model.evaluate(xx, yy, verbose=0)

    ypr_predicted = F1_model.predict(xt).flatten()
    ypr = (ypr_predicted > 0.5).astype(int)
    
    precision = precision_score(yt, ypr, average='macro')
    recall = recall_score(yt, ypr, average='macro')
    f1score = f1_score(yt, ypr, average='macro')

    return tr_acc, test_acc, precision, recall, f1score,ypr_predicted

import numpy as np
from sklearn.utils import shuffle

def oversample_mcmc(x_train, y_train, oversample_ratio,num_iterations):
   
    # Count the number of samples in each class
    unique, counts = np.unique(y_train, return_counts=True)
    minority_class_label = unique[np.argmin(counts)]
    minority_class_count = np.min(counts)
    majority_class_label = unique[np.argmax(counts)]

    # Calculate the oversampling target for the minority class
    target_minority_count = int(minority_class_count * oversample_ratio)

    # Initialize the balanced dataset
    balanced_x_train = x_train.copy()
    balanced_y_train = y_train.copy()

    # Perform MCMC sampling to oversample the minority class
    for iteration in range(num_iterations):
        # Randomly choose a sample from the minority class
        minority_indices = np.where(balanced_y_train == minority_class_label)[0]
        chosen_index = np.random.choice(minority_indices)
        chosen_sample = balanced_x_train[chosen_index]

        # Create a proposal sample by adding small noise
        proposal_sample = chosen_sample + np.random.normal(scale=0.01, size=chosen_sample.shape)

        # Calculate the likelihood ratio (considering binary classification)
        likelihood_ratio = 1.0

        # Acceptance probability for the proposal
        acceptance_prob = min(1, likelihood_ratio)

        # Accept or reject the proposal based on acceptance probability
        if np.random.rand() < acceptance_prob:
            # Add the proposal sample to the balanced dataset
            balanced_x_train = np.vstack([balanced_x_train, proposal_sample])
            balanced_y_train = np.append(balanced_y_train, minority_class_label)

    # Shuffle the balanced dataset
    balanced_x_train, balanced_y_train = shuffle(balanced_x_train, balanced_y_train, random_state=42)

    return balanced_x_train, balanced_y_train

data = pd.read_csv('Data\ecoli.csv', sep=",", header='infer' )

randomstate = 2
t=() 
#gets the dimension of the array also the element in the array
t=data.shape 
X = data.values[:,0:(t[1]-1)].astype(float)
Y = data.values[:,(t[1]-1)].astype(float)

randomstate=1
epochfOne=30

# spilt the data between train test
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=randomstate)
n_neighbors = 3 # Ensure n_neighbors is <= number of samples

smote = SMOTE(random_state=1, k_neighbors=n_neighbors)
X_train_res,y_train_res = smote.fit_resample(X_train,y_train)

#PREPROCESSING 
t2=X_train.shape
X_oversampled=X_train_res[(t2[0]):]
z_dim = t2[1]
t4=X_oversampled.shape

#MCMC_Genrated = MCMC(X_train,y_train,X)
class_counts = Counter(y_train)
minority_class_label = min(class_counts, key=class_counts.get)
majority_class_label = max(class_counts, key=class_counts.get)
NumberOfSamplesToGenerate = np.count_nonzero(y_train == majority_class_label) - np.count_nonzero(y_train ==minority_class_label)
MCMC_Genrated = oversample_mcmc(X_train,y_train, 1.0,NumberOfSamplesToGenerate)


train_accuracy_MCMC = []
test_accuracy_MCMC = []
f1_score_MCMC = []
precision_MCMC = []
recall_MCMC = [] 
ypr_MCMC = []
MCMC = [] 

for i in range(30):
    
    MCMC = callf1(MCMC_Genrated[0],MCMC_Genrated[1],X_test,y_test,epochfOne)
    
    train_accuracy_MCMC.append(MCMC[0])
    test_accuracy_MCMC.append(MCMC[1])
    precision_MCMC.append(MCMC[2])
    recall_MCMC.append(MCMC[3])
    f1_score_MCMC.append(MCMC[4])  
    ypr_MCMC.append(MCMC[5])

    
# After collecting all precision and recall values for MCMC
precision_MCMC_avg = np.mean(precision_MCMC)
recall_MCMC_avg = np.mean(recall_MCMC)
precision_MCMC_curve, recall_MCMC_curve, _ = precision_recall_curve(y_test, np.mean(ypr_MCMC, axis=0))
pr_auc_MCMC = auc(recall_MCMC_curve, precision_MCMC_curve)

with open('results_MCMC.pkl', 'wb') as f:
    pickle.dump((precision_MCMC_curve, recall_MCMC_curve, pr_auc_MCMC), f)

print("MCMC Train - ",np.mean(train_accuracy_MCMC))
print("MCMC Test - ",np.mean(test_accuracy_MCMC))
print("MCMC F1-score - ",np.mean(f1_score_MCMC))
print("MCMC precision - ",np.mean(precision_MCMC))
print("MCMC recall - ",np.mean(recall_MCMC))


print("MCMC Train - ",train_accuracy_MCMC)
print("MCMC Test - ",test_accuracy_MCMC)
print("MCMC F1-score - ",f1_score_MCMC)
print("MCMC precision - ",precision_MCMC)
print("MCMC recall - ",recall_MCMC)