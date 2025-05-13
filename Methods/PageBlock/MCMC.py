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

def MCMC(X_train,y_train,X):
    
    class_counts = Counter(y_train)
    minority_class_label = min(class_counts, key=class_counts.get)
    majority_class_label = max(class_counts, key=class_counts.get)
    NumberOfSamplesToGenerate = np.count_nonzero(y_train == majority_class_label) - np.count_nonzero(y_train ==minority_class_label)
    burninNeeded = int( 0.2 * NumberOfSamplesToGenerate)
    NumberOfSamplesToGenerate = NumberOfSamplesToGenerate + burninNeeded
    mean = X_train[y_train == minority_class_label].mean()
    std = X_train[y_train == minority_class_label].std()

    walkers = np.random.normal(mean, std, (NumberOfSamplesToGenerate, X.shape[1]))

    for i in range(NumberOfSamplesToGenerate):
        new_mean = np.random.normal(mean, std)
        likelihood = np.exp(-np.sum((X - new_mean) ** 2) / (2 * std ** 2))
        acceptance_probability = min(1, likelihood.any() / np.exp(-np.sum((X_train - mean) ** 2) / (2 * std ** 2)))
        if acceptance_probability >= np.random.rand():
            mean = new_mean


# Assuming walkers is a list of generated means from MCMC
    
    #burn_in = int(0.2 * len(walkers))  # Discard 20% as burn-in (adjust as needed)
    post_burn_in_means = walkers[burninNeeded:] 
    #synthetic_means = [walker.mean() for walker in walkers][X_train.shape[1]]
    NumberOfSamplesToGenerate = int(NumberOfSamplesToGenerate - burninNeeded)
    synthetic_data = np.random.normal(post_burn_in_means, std, (NumberOfSamplesToGenerate, X.shape[1]))

    concatenatedXs = np.concatenate((X_train , synthetic_data) , axis=0)
    concatenatedYs = np.concatenate((y_train, np.ones(NumberOfSamplesToGenerate)), axis=0 )

    X_MCMC,y_MCMC = shuffle_in_unison(concatenatedXs, concatenatedYs)
    class_counts_A = Counter(y_MCMC)
    minority_class_label_A = min(class_counts_A, key=class_counts.get)
    majority_class_label_A = max(class_counts_A, key=class_counts.get)

    
    return synthetic_data,X_MCMC,y_MCMC

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
    
    precision = precision_score(yt, ypr, average='binary')
    recall = recall_score(yt, ypr, average='binary')
    f1score = f1_score(yt, ypr, average='binary')

    return tr_acc, test_acc, precision, recall, f1score, ypr_predicted

import numpy as np
from sklearn.utils import shuffle

def oversample_mcmc(x_train, y_train, oversample_ratio,num_iterations):
    """
    Perform oversampling of imbalanced binary classification data using MCMC.

    Parameters:
    - x_train: numpy array, feature matrix of shape (n_samples, n_features)
    - y_train: numpy array, binary labels of shape (n_samples,)
    - oversample_ratio: float, desired ratio of minority class to majority class after oversampling
    - num_iterations: int, number of MCMC iterations

    Returns:
    - balanced_x_train: numpy array, balanced feature matrix
    - balanced_y_train: numpy array, balanced binary labels
    """
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

data = pd.read_csv('Data\pageblock.csv', sep=",", header='infer' )

randomstate = 2
t=() 
#gets the dimension of the array also the element in the array
t=data.shape 
X = data.values[:,0:(t[1]-1)].astype(float)
Y = data.values[:,(t[1]-1)].astype(float)

randomstate=1
epochfOne=30
device = 'cpu'

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=randomstate)
#MCMC_Genrated = MCMC(X_train,y_train,X)
class_counts = Counter(y_train)
minority_class_label = min(class_counts, key=class_counts.get)
majority_class_label = max(class_counts, key=class_counts.get)
NumberOfSamplesToGenerate = np.count_nonzero(y_train == majority_class_label) - np.count_nonzero(y_train ==minority_class_label)
MCMC_Genrated = oversample_mcmc(X_train,y_train, 1.0,NumberOfSamplesToGenerate)

train_accuracy_MCMC=[]
test_accuracy_MCMC =[]
f1_score_MCMC =[]
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

with open('PAGEBLOCK_MCMC.pkl', 'wb') as f:
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