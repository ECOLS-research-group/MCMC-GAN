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
    tf.keras.backend.clear_session()
    # Evaluation on test set
    test_loss, test_acc = F1_model.evaluate(xt, yt, verbose=0)
    tr_loss, tr_acc = F1_model.evaluate(xx, yy, verbose=0)

    ypr_predicted = F1_model.predict(xt).flatten()
    ypr = (ypr_predicted > 0.5).astype(int)
    
    
    precision = precision_score(yt, ypr, average='macro')
    recall = recall_score(yt, ypr, average='macro')
    f1score = f1_score(yt, ypr, average='macro')

    return tr_acc, test_acc, precision, recall, f1score, ypr_predicted


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
    concatenatedYs = np.concatenate((y_train, np.zeros(NumberOfSamplesToGenerate)), axis=0 )

    X_MCMC,y_MCMC = shuffle_in_unison(concatenatedXs, concatenatedYs)
    class_counts_A = Counter(y_MCMC)
    minority_class_label_A = min(class_counts_A, key=class_counts.get)
    majority_class_label_A = max(class_counts_A, key=class_counts.get)

    
    return synthetic_data,X_MCMC,y_MCMC

data = pd.read_csv('Data\ionosphere.csv', sep=",", header='infer' )

randomstate = 2
t=() 
#gets the dimension of the array also the element in the array
t=data.shape 
X = data.values[:,0:(t[1]-1)].astype(float)
Y = data.values[:,(t[1]-1)].astype(float)

randomstate=1
epochfOne=30
numberof_epochs=1000
criterion = nn.BCEWithLogitsLoss()
learningRate = 0.0002
batch_size = 128
display_step = 1
device = 'cpu'
NN_ep = 30

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

MCMC_Genrated = MCMC(X_train,y_train,X)
# combined_data_SMOTE =np.concatenate((X_train[:(t2[0])], MCMC_Genrated[1] ), axis=0)
# XSmotified,ySmotified= shuffle_in_unison(combined_data_SMOTE , y_train_res)


train_accuracy_MCMC=[]
test_accuracy_MCMC =[]
f1_score_MCMC =[]
precision_MCMC = []
recall_MCMC = []
ypr_MCMC = []
MCMC = []

for i in range(30):
    
    MCMC = callf1(MCMC_Genrated[1],MCMC_Genrated[2],X_test,y_test,epochfOne)
    
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

with open('IONOSPHERE_MCMC.pkl', 'wb') as f:
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