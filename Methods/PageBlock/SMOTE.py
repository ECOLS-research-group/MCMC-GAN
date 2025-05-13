import numpy as np 
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from imblearn.over_sampling import SMOTE
import pandas as pd 
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


class ModelEvaluator:
    def __init__(self):
        self.model = None

    def callf1(self,xx,yy,xt,yt,ep):#model with 3 layers

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        model.compile(optimizer='adam',
                    loss='mean_absolute_error',
                    metrics=['accuracy'])
        
        model.fit(xx, yy, epochs=ep)

        ls = []

        test_loss, test_acc = model.evaluate(xt,  yt, verbose=2)
        tr_loss, tr_acc = model.evaluate(xx,  yy, verbose=2)

        ypr_predicted = model.predict(xt).flatten()
        ypr = (ypr_predicted > 0.5).astype(int)
        
        precision = precision_score(yt, ypr)
        recall = recall_score(yt, ypr)
        f1score = f1_score(yt, ypr)

        return tr_acc, test_acc, precision, recall, f1score, ypr_predicted



data = pd.read_csv('Data\pageblock.csv', sep=",", header='infer' )

randomstate = 2
t=() 
#gets the dimension of the array also the element in the array
t=data.shape 
X = data.values[:,0:(t[1]-1)].astype(float)
Y = data.values[:,(t[1]-1)]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

randomstate=1
epochfOne=10
device = 'cpu'

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2,stratify = Y, random_state=randomstate)
smote = SMOTE(random_state=2)
X_train_res,y_train_res = smote.fit_resample(X_train,y_train)
 

#NON OVERSAMPLED
train_accuracy=[]
test_accuracy =[]
f1_score_non =[]
precision = []
recall = []
ypr_ = []
NON = []

for i in range(30):
    model_evaluator = ModelEvaluator()
    NON = model_evaluator.callf1(X_train,y_train,X_test,y_test,epochfOne)
    
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

with open('PAGEBLOCK_NON.pkl', 'wb') as f:
    pickle.dump((precision_NON_curve, recall_NON_curve, pr_auc_NON), f)
    
#SMOTE OVERSAMPLED
train_accuracy_MCMC=[]
test_accuracy_MCMC =[]
f1_score_MCMC =[]
precision_MCMC = []
recall_MCMC = []
ypr_MCMC = []
MCMC = []

for i in range(30):
    
    model_evaluator = ModelEvaluator()
    MCMC = model_evaluator.callf1(X_train_res,y_train_res,X_test,y_test,epochfOne)
    
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

with open('PAGEBLOCK_SMOTE.pkl', 'wb') as f:
    pickle.dump((precision_MCMC_curve, recall_MCMC_curve, pr_auc_MCMC), f)

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