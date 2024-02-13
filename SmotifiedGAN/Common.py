import numpy as np 
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE
import torch
from torch import nn
from tqdm.auto import tqdm
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader


def callf1(xx,yy,xt,yt,ep):#model with 3 layers

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

    ls.append(tr_acc)
    ls.append(test_acc)
    ypr = model.predict(xt)
    ypr = (ypr > 0.5)*1
    ypre = np.ravel(ypr)
    #ls.append(f1_score(yt, ypre))
    precision = precision_score(yt, ypre)
    recall = recall_score(yt, ypre)
    ls.append(precision)
    ls.append(recall)
    f1score = f1_score(yt, ypre)
    ls.append(f1score)

    return ls

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

#GAN MODELS

def get_generator_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True),
    )

class Generator(nn.Module):

    def __init__(self, z_dim, im_dim, hidden_dim=128):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            get_generator_block(z_dim, hidden_dim),
            get_generator_block(hidden_dim, hidden_dim * 2),
            get_generator_block(hidden_dim * 2, hidden_dim * 4),
            get_generator_block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid()
        )
    def forward(self, noise):
        return self.gen(noise)
    
    def get_gen(self):
        return self.gen
    
def get_discriminator_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(0.2, inplace=True)        
    )

class Discriminator(nn.Module):
    def __init__(self, im_dim, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            get_discriminator_block(im_dim, hidden_dim * 4),
            get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            get_discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, image):
        return self.disc(image)
    
    def get_disc(self):
        return self.dis
    
def get_noise(n_samples, z_dim, device='cpu'):
    return torch.randn(n_samples,z_dim,device=device) 

def get_disc_loss_smote(gen, disc, criterion, real, noise):

    fake = gen(noise)
    disc_fake_pred = disc(fake.detach())
    disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
    disc_real_pred = disc(real)
    disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
    disc_loss = (disc_fake_loss + disc_real_loss) / 2
    return disc_loss

def get_gen_loss_smote(gen, disc, criterion, noise):

    fake_images = gen(noise)
    disc_fake_pred = disc(fake_images)
    gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
    return gen_loss

#DATA GENERATION METHODS
def trainGAN(numberof_epochs,dataloader,device,disc_opt_SMOTE,genSMOTE,disc_SMOTE,criterion,SmoteNoise,gen_opt_SMOTE,learningRate,
            display_step,X_train_res,y_train_res,X_test,y_test,ep,t2,x_train):

    cur_step = 0
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    test_generator = True 
    gen_loss_SMOTE = False
    error = False

    for epoch in range(numberof_epochs):
        for real, _ in tqdm(dataloader):
            cur_batch_size = len(real)
            real = real.view(cur_batch_size, -1).to(device)

            disc_opt_SMOTE.zero_grad()
            disc_loss_SMOTE = get_disc_loss_smote(genSMOTE, disc_SMOTE, criterion, real, SmoteNoise )
            disc_loss_SMOTE.backward(retain_graph=True)
            disc_opt_SMOTE.step()

            if test_generator:
                old_generator_weights = genSMOTE.gen[0][0].weight.detach().clone()

            gen_opt_SMOTE.zero_grad()
            gen_loss_SMOTE = get_gen_loss_smote(genSMOTE, disc_SMOTE, criterion, SmoteNoise)
            gen_loss_SMOTE.backward()
            gen_opt_SMOTE.step()

            if test_generator:
                try:
                    assert learningRate > 0.0000002 or (
                        genSMOTE.gen[0][0].weight.grad.abs().max() < 0.0005 and epoch == 0)
                    assert torch.any(
                        genSMOTE.gen[0][0].weight.detach().clone() != old_generator_weights)
                except:
                    error = True
                    print("Runtime tests have failed")

            mean_discriminator_loss += disc_loss_SMOTE.item() / display_step
            mean_generator_loss += gen_loss_SMOTE.item() / display_step

            if cur_step % display_step == 0 and cur_step > 0:
                print(
                    f"Epoch {epoch}, step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
                mean_generator_loss = 0
                mean_discriminator_loss = 0
            cur_step += 1
            
    train_accuracy_SMOTEGAN=[]
    test_accuracy_SMOTEGAN =[]
    precision_SMOTEGAN=[]
    recall_SMOTEGAN =[]
    f1_score_SMOTEGAN =[]
    SMOTEGAN = []
    classifier = "NN -, "

    for i in range(30):
        generated_data_SMOTE = genSMOTE(SmoteNoise)
        generated_data_cpu_SMOTE = generated_data_SMOTE.cpu().detach().numpy()
        combined_data_SMOTE =np.concatenate((x_train[:(t2[0])], generated_data_cpu_SMOTE ), axis=0)
        XSmotified,ySmotified= shuffle_in_unison(combined_data_SMOTE , y_train_res)
        
        SMOTEGAN = callf1(XSmotified,ySmotified.ravel(),X_test,y_test.ravel(),ep)
        #SMOTEGAN = RandomForest(XSmotified,ySmotified.ravel(),X_test,y_test.ravel(),ep)
        # SMOTEGAN = SVM(XSmotified,ySmotified.ravel(),X_test,y_test.ravel(),ep)
        # SMOTEGAN = XGBoost(XSmotified,ySmotified.ravel(),X_test,y_test.ravel(),ep)
        # SMOTEGAN = LG(XSmotified,ySmotified.ravel(),X_test,y_test.ravel(),ep)
        
        #SMOTEGAN = callf1(XSmotified,ySmotified.ravel(),X_test,y_test.ravel(),ep)
        train_accuracy_SMOTEGAN.append(SMOTEGAN[0])
        test_accuracy_SMOTEGAN.append(SMOTEGAN[1])
        precision_SMOTEGAN.append(SMOTEGAN[2])
        recall_SMOTEGAN.append(SMOTEGAN[3])
        f1_score_SMOTEGAN.append(SMOTEGAN[4]) 

    
    return train_accuracy_SMOTEGAN,test_accuracy_SMOTEGAN,f1_score_SMOTEGAN,precision_SMOTEGAN,recall_SMOTEGAN

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

def NON_OVERSAMPLED(X_train,y_train,X_test,y_test,epochfOne,NN_ep):

    # for f in range(5):
    #     #f = 2
    
    train_accuracy =[]
    test_accuracy =[]
    Precision =[]
    recall =[]
    f1_score_ =[]
    IMBALANCED = []
    classifier = "NN -, "

    for i in range(30):
        
        IMBALANCED = callf1(X_train,y_train.ravel(),X_test,y_test.ravel(),epochfOne)
        #IMBALANCED = RandomForest(X_train,y_train.ravel(),X_test,y_test.ravel(),epochfOne)
        # IMBALANCED = SVM(X_train,y_train.ravel(),X_test,y_test.ravel(),epochfOne)
        # IMBALANCED = XGBoost(X_train,y_train.ravel(),X_test,y_test.ravel(),epochfOne)
        # IMBALANCED = LG(X_train,y_train.ravel(),X_test,y_test.ravel(),epochfOne)
            
        #IMBALANCED = callf1(X_train,y_train.ravel(),X_test,y_test.ravel(),epochfOne)
        train_accuracy.append(IMBALANCED[0])
        test_accuracy.append(IMBALANCED[1])
        Precision.append(IMBALANCED[2])
        recall.append(IMBALANCED[3])
        f1_score_.append(IMBALANCED[4]) 

    print("NON OVERSAMPLED Train - , ",classifier,np.mean(train_accuracy[0]), file=open("output.txt", "a"))
    print("NON OVERSAMPLED Test - ,",classifier,np.mean(test_accuracy[1]), file=open("output.txt", "a"))
    print("NON OVERSAMPLED F1-score - ,",classifier,np.mean(f1_score_[2]), file=open("output.txt", "a"))
    print("NON OVERSAMPLED precision - ,",classifier,np.mean(Precision[3]), file=open("output.txt", "a"))
    print("NON OVERSAMPLED recall - ,",classifier,np.mean(recall[4]), file=open("output.txt", "a"))

    return train_accuracy,test_accuracy,f1_score_,Precision,recall

def SMOTIfied(X_train,y_train,X_test,y_test,epochfOne,NN_ep):
     
    train_accuracy_SMOTE=[]
    test_accuracy_SMOTE=[]
    Precision_SMOTE=[]
    Recall_SMOTE=[]
    f1_score_SMOTE =[]
    SMOTER = []
    classifier = "NN -, "

    X_train_res,y_train_res = SMOTE().fit_resample(X_train,y_train)

    for i in range(30):
        SMOTER = callf1(X_train_res,y_train_res.ravel(),X_test,y_test.ravel(),epochfOne)
        # SMOTER = RandomForest(X_train_res,y_train_res.ravel(),X_test,y_test.ravel(),epochfOne)
        # SMOTER = SVM(X_train_res,y_train_res.ravel(),X_test,y_test.ravel(),epoc/hfOne)
        # SMOTER = XGBoost(X_train_res,y_train_res.ravel(),X_test,y_test.ravel(),epochfOne)
        # SMOTER = LG(X_train_res,y_train_res.ravel(),X_test,y_test.ravel(),epochfOne)

        #SMOTER = callf1(X_train_res,y_train_res.ravel(),X_test,y_test.ravel(),epochfOne)
        train_accuracy_SMOTE.append(SMOTER[0])
        test_accuracy_SMOTE.append(SMOTER[1])
        Precision_SMOTE.append(SMOTER[2])
        Recall_SMOTE.append(SMOTER[3])
        f1_score_SMOTE.append(SMOTER[4]) 

    print("SMOTE Train  - , ",classifier,np.mean(train_accuracy_SMOTE), file=open("output.txt", "a"))
    print("SMOTE Test - , ",classifier,np.mean(test_accuracy_SMOTE), file=open("output.txt", "a"))
    print("SMOTE precision - , ",classifier,np.mean(Precision_SMOTE), file=open("output.txt", "a"))
    print("SMOTE recall - , ",classifier,np.mean(Recall_SMOTE), file=open("output.txt", "a"))
    print("SMOTE F1-score - , ",classifier,np.mean(f1_score_SMOTE), file=open("output.txt", "a"))


    return train_accuracy_SMOTE,test_accuracy_SMOTE,Precision_SMOTE,Recall_SMOTE,f1_score_SMOTE

