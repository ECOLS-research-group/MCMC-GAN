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
from sklearn.preprocessing import OneHotEncoder

import pickle
from sklearn.metrics import precision_recall_curve, auc

# from google.colab import drive
#drive.mount('/content/drive')

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

def callf1(xx, yy, xt, yt, ep):

    F1_model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1)
    ])

    F1_model.compile(optimizer='adam',
              loss='mean_absolute_error',
              metrics=['accuracy'])


    F1_model.fit(xx, yy, epochs=ep, verbose=0)
    tf.keras.backend.clear_session()
    # Evaluation on test set
    test_loss, test_acc = F1_model.evaluate(xt, yt, verbose=0)
    tr_loss, tr_acc = F1_model.evaluate(xx, yy, verbose=0)

    ypr_predicted = F1_model.predict(xt).flatten()
    ypr = (ypr_predicted > 0.5).astype(int)
    
    precision = precision_score(yt, ypr)
    recall = recall_score(yt, ypr)
    f1score = f1_score(yt, ypr)

    return tr_acc, test_acc, precision, recall, f1score, ypr_predicted


data = pd.read_csv(r'Data\abalone.csv', sep=",", header='infer' )

randomstate = 2
t=() 
#gets the dimension of the array also the element in the array
t=data.shape 
# X = data.values[:,0:(t[1]-1)].astype(float)
# Y = data.values[:,(t[1]-1)]

category = np.repeat("empty000", data.shape[0])
for i in range(0, data["Class_number_of_rings"].size):
    if(data["Class_number_of_rings"][i] <= 7):
        category[i] = int(1)
    elif(data["Class_number_of_rings"][i] > 7):
        category[i] = int(0)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(data['Sex'])
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
data = data.drop(['Sex'], axis=1)
data['category_size'] = category
data = data.drop(['Class_number_of_rings'], axis=1)
features = data.iloc[:,np.r_[0:7]]
labels = data.iloc[:,7]
X_train, X_test, y_train, y_test, X_gender, X_gender_test = train_test_split(features, labels, onehot_encoded, random_state=10, test_size=0.2)
temp = X_train.values
X_train = np.concatenate((temp, X_gender), axis=1)
temp2 = X_test.values
X_test = np.concatenate((temp2, X_gender_test), axis=1)
test_list = [int(i) for i in y_test.ravel()] 
y_test=np.array(test_list)
train_list = [int(i) for i in y_train.ravel()] 
y_train=np.array(train_list)

randomstate=1
epochfOne=30
device = 'cpu'

# spilt the data between train test
#X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=randomstate)
n_neighbors = 3 # Ensure n_neighbors is <= number of samples

smote = SMOTE(random_state=1, k_neighbors=n_neighbors)
X_train_res,y_train_res = smote.fit_resample(X_train,y_train)
   
train_accuracy_MCMC=[]
test_accuracy_MCMC =[]
f1_score_MCMC =[]
precision_MCMC = []
recall_MCMC = []
ypr_MCMC = []
MCMC = []

for i in range(30):
    
    MCMC = callf1(X_train_res,y_train_res,X_test,y_test,epochfOne)
    
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

with open('ABALONE_SMOTE.pkl', 'wb') as f:
    pickle.dump((precision_MCMC_curve, recall_MCMC_curve, pr_auc_MCMC), f)

train_accuracy=[]
test_accuracy =[]
f1_score_non =[]
precision = []
recall = []
ypr_ = []
NON = []

for i in range(30):
    
    NON = callf1(X_train,y_train,X_test,y_test,epochfOne)
    
    train_accuracy.append(NON[0])
    test_accuracy.append(NON[1])
    precision.append(NON[2])
    recall.append(NON[3])
    f1_score_non.append(NON[4]) 
    ypr_.append(NON[5])

# After collecting all precision and recall values for NON (No augmentation)
precision_NON_avg = np.mean(precision)
recall_NON_avg = np.mean(recall)
precision_NON_curve, recall_NON_curve, _ = precision_recall_curve(y_test, np.mean(ypr_, axis=0))
pr_auc_NON = auc(recall_NON_curve, precision_NON_curve)

with open('ABALONE_NON.pkl', 'wb') as f:
    pickle.dump((precision_NON_curve, recall_NON_curve, pr_auc_NON), f)

print("NON Train - ",np.mean(train_accuracy))
print("NON Test - ",np.mean(test_accuracy))
print("NON F1-score - ",np.mean(f1_score_non))
print("NON precision - ",np.mean(precision))
print("NON recall - ",np.mean(recall))

print("NON Train - ",train_accuracy)
print("NON Test - ",test_accuracy)
print("NON F1-score - ",f1_score_non)
print("NON precision - ",precision)
print("NON recall - ",recall)

print("SMOTE Train - ",np.mean(train_accuracy_MCMC))
print("SMOTE Test - ",np.mean(test_accuracy_MCMC))
print("SMOTE F1-score - ",np.mean(f1_score_MCMC))
print("SMOTE precision - ",np.mean(precision_MCMC))
print("SMOTE recall - ",np.mean(recall_MCMC))

print("SMOTE Train - ",train_accuracy_MCMC)
print("SMOTE Test - ",test_accuracy_MCMC)
print("SMOTE F1-score - ",f1_score_MCMC)
print("SMOTE precision - ",precision_MCMC)
print("SMOTE recall - ",recall_MCMC)
