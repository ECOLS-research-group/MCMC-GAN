# DIFFERENT METHODS FOR EACH TYPE OF DATA FOR EXTARCTING DATA AND THEIR PARAMTERS 
# METHOD THAT WILL PROCESS THIS DATA AND PRINT THE RESULTS
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from imblearn.over_sampling import SMOTE
from tqdm.auto import tqdm
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
import Common as common
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from keras.utils import np_utils

class TuningParameters:
    def __init__(self, randomstate,epochfOne,numberof_epochs,criterion,learningRate,batch_size,display_step ,device,NN_ep,X,y,X_train,y_train,X_test,y_test,
                 X_train_res,y_train_res,X_real,y_real,z_dim,t4): 
         
        self.randomstate = randomstate
        self.epochfOne = epochfOne
        self.numberof_epochs = numberof_epochs
        self.criterion = criterion
        self.learningRate = learningRate
        self.batch_size = batch_size
        self.display_step = display_step
        self.device = device
        self.NN_ep = NN_ep
        self.X = X
        self.y = y
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_train_res = X_train_res
        self.y_train_res = y_train_res
        self.X_real = X_real
        self.y_real = y_real
        self.z_dim = z_dim
        self.t4 = t4

def GenerateResults(TuningParameters):
    randomstate = TuningParameters.randomstate
    epochfOne = TuningParameters.epochfOne
    numberof_epochs = TuningParameters.numberof_epochs
    criterion = TuningParameters.criterion
    learningRate = TuningParameters.learningRate
    batch_size = TuningParameters. batch_size
    display_step = TuningParameters.display_step
    device = TuningParameters.device
    NN_ep = TuningParameters.NN_ep
    X = TuningParameters.X
    y = TuningParameters.y
    X_train = TuningParameters.X_train
    y_train = TuningParameters.y_train
    X_test = TuningParameters.X_test
    y_test = TuningParameters.y_test
    X_train_res = TuningParameters.X_train_res
    y_train_res = TuningParameters.y_train_res
    X_real = TuningParameters.X_real
    y_real = TuningParameters.y_real
    z_dim = TuningParameters.z_dim
    t4 = TuningParameters.t4

    
    t2=X_train.shape
    X_oversampled=X_train_res[(t2[0]):]
    epochfOne = 10
     
    #NON OVERSAMPLED SECTION
    NonSampled = common.NON_OVERSAMPLED(X_train,y_train,X_test,y_test,epochfOne,NN_ep)

    #SMOTE SECTION 
    SMOTEE = common.SMOTIfied(X_train,y_train,X_test,y_test,epochfOne,NN_ep)

    # #GAN SECTION
    tensor_x = torch.Tensor(X_real) 
    tensor_y = torch.Tensor(y_real)

    my_dataset = TensorDataset(tensor_x,tensor_y)
    dataloader = DataLoader(
        my_dataset,
        batch_size=batch_size,
        shuffle=True)

    genGAN = common.Generator(z_dim,im_dim = z_dim).to(device)
    gen_opt_GAN = torch.optim.Adam(genGAN.parameters(), lr=learningRate)
    disc_GAN = common.Discriminator(im_dim = z_dim).to(device) 
    disc_opt_GAN = torch.optim.Adam(disc_GAN.parameters(), lr=learningRate)
    Random_Noise = common.get_noise((t4[0]), z_dim, device=device)
    fake_noise = Random_Noise.float().to(device)


    GANResults = common.trainGAN(numberof_epochs,dataloader,device,disc_opt_GAN,genGAN,disc_GAN,criterion,fake_noise,gen_opt_GAN,learningRate,
                display_step,X_train_res,y_train_res,X_test,y_test,NN_ep,t2,X_train)
    
    print("GAN  Train - , ",np.mean(GANResults[0]), file=open("output.txt", "a"))
    print("GAN  Test - ,",np.mean(GANResults[1]), file=open("output.txt", "a"))
    print("GAN  F1-score - ,",np.mean(GANResults[2]), file=open("output.txt", "a"))
    print("GAN  precision - ,",np.mean(GANResults[3]), file=open("output.txt", "a"))
    print("GAN nd recall - ,",np.mean(GANResults[4]), file=open("output.txt", "a"))
    
    # # #SMOTIFIED GAN SECTION
    
    genSMOTE = common.Generator(z_dim,im_dim = z_dim).to(device)
    gen_opt_SMOTE = torch.optim.Adam(genSMOTE.parameters(), lr=learningRate)
    disc_SMOTE = common.Discriminator(im_dim = z_dim).to(device) 
    disc_opt_SMOTE = torch.optim.Adam(disc_SMOTE.parameters(), lr=learningRate)
    smote = torch.from_numpy(X_oversampled) 
    SmoteNoise = smote.float().to(device)

    SMOTIFIEDGANresults = common.trainGAN(numberof_epochs,dataloader,device,disc_opt_SMOTE,
                    genSMOTE,disc_SMOTE,criterion,SmoteNoise,gen_opt_SMOTE,learningRate,
                display_step,X_train_res,y_train_res,X_test,y_test,epochfOne,t2,X_train)
  
    # #MCMC SECTION 
    synthetic_data,X_MCMC,y_MCMC = common.MCMC(X_train,y_train,X)

    train_accuracy_MCMC=[]
    test_accuracy_MCMC =[]
    f1_score_MCMC =[]
    precision_MCMC = []
    recall_MCMC = []
    MCMC = []

    for i in range(30):
        
        MCMC = common.callf1(X_MCMC,y_MCMC,X_test,y_test,epochfOne)
       
        train_accuracy_MCMC.append(MCMC[0])
        test_accuracy_MCMC.append(MCMC[1])
        precision_MCMC.append(MCMC[2])
        recall_MCMC.append(MCMC[3])
        f1_score_MCMC.append(MCMC[4]) 

    print("MCMC Train  - , ",np.mean(train_accuracy_MCMC), file=open("output.txt", "a"))
    print("MCMC Test - , ",np.mean(test_accuracy_MCMC), file=open("output.txt", "a"))
    print("MCMC precision - , ",np.mean(precision_MCMC), file=open("output.txt", "a"))
    print("MCMC recall - , ",np.mean(recall_MCMC), file=open("output.txt", "a"))
    print("MCMC F1-score - , ",np.mean(f1_score_MCMC), file=open("output.txt", "a"))
    

     # MCMC GAN SECTION
    genMCMC = common.Generator(z_dim,im_dim = z_dim).to(device)
    gen_opt_MCMC = torch.optim.Adam(genMCMC.parameters(), lr=learningRate)
    disc_MCMC = common.Discriminator(im_dim = z_dim).to(device) 
    disc_opt_MCMC = torch.optim.Adam(disc_MCMC.parameters(), lr=learningRate)
    MCMCied = torch.from_numpy(synthetic_data)
    MCMC_Noise = MCMCied.float().to(device)
    

    MCMCIEDGANresults = common.trainGAN(numberof_epochs,dataloader,device,disc_opt_MCMC,genMCMC,disc_MCMC,criterion,MCMC_Noise,gen_opt_MCMC,learningRate,
        display_step,X_train_res,y_train_res,X_test,y_test,epochfOne,t2,X_train)

    # #PRINTING SECTION 
    print("NON OVERSAMPLED Train - ",np.mean(NonSampled[0]))
    print("NON OVERSAMPLED Test - ",np.mean(NonSampled[1]))
    print("NON OVERSAMPLED F1-score - ",np.mean(NonSampled[2]))
    print("NON OVERSAMPLED precision - ",np.mean(NonSampled[3]))
    print("NON OVERSAMPLED recall - ",np.mean(NonSampled[4]))

    print("SMOTE Train - ",np.mean(SMOTEE[0]))
    print("SMOTE Test - ",np.mean(SMOTEE[1]))
    print("SMOTE precision - ",np.mean(SMOTEE[2]))
    print("SMOTE recall - ",np.mean(SMOTEE[3]))
    print("SMOTE F1-score - ",np.mean(SMOTEE[4]))

    print("GAN Train - ",np.mean(GANResults[0]))
    print("GAN Test - ",np.mean(GANResults[1]))
    print("GAN F1-score - ",np.mean(GANResults[2]))
    print("GAN precision - ",np.mean(GANResults[3]))
    print("GAN recall - ",np.mean(GANResults[4]))

    print("SMOTEGAN Train - ",np.mean(SMOTIFIEDGANresults[0]))
    print("SMOTEGAN Test - ",np.mean(SMOTIFIEDGANresults[1]))
    print("SMOTEGAN F1-score - ",np.mean(SMOTIFIEDGANresults[2]))
    print("SMOTEGAN Precision - ",np.mean(SMOTIFIEDGANresults[3]))
    print("SMOTEGAN recall - ",np.mean(SMOTIFIEDGANresults[4]))

    print("MCMC Train - ",np.mean(train_accuracy_MCMC))
    print("MCMC Test - ",np.mean(test_accuracy_MCMC))
    print("MCMC F1-score - ",np.mean(f1_score_MCMC))
    print("MCMC precision - ",np.mean(precision_MCMC))
    print("MCMC recall - ",np.mean(recall_MCMC))

    print("MCMC-GAN Train - ",np.mean(MCMCIEDGANresults[0]))
    print("MCMC-GAN Test - ",np.mean(MCMCIEDGANresults[1]))
    print("MCMC-GAN F1-score - ",np.mean(MCMCIEDGANresults[2]))
    print("MCMC-GAN precision - ",np.mean(MCMCIEDGANresults[3]))
    print("MCMC-GAN recall - ",np.mean(MCMCIEDGANresults[4]))

    return t2

def EcoliProcessing():
    url='Data\ecoli.dat'
    data = pd.read_csv(url, sep=",", header='infer' )

    t=() 
    #gets the dimension of the array also the element in the array
    t=data.shape 
    X = data.values[:,0:(t[1]-1)].astype(float)
    Y = data.values[:,(t[1]-1)]

    #encode the label i.e. transform postive negative to 0s and 1s
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)

    yk = []
    for i in encoded_Y:
        if i == 2:
            yk.append(1)
        else:
            yk.append(0)
    encoded_Y = np.asarray(yk, dtype=np.int32)

    randomstate=1
    epochfOne=20
    numberof_epochs=700
    criterion = nn.BCEWithLogitsLoss()
    learningRate = 0.000001
    batch_size = 128
    display_step = 1
    device = 'cpu'
    NN_ep = 30

    # spilt the data between train test
    X_train, X_test, y_train, y_test = train_test_split(X,encoded_Y, test_size=0.2, stratify = encoded_Y , random_state=randomstate)
    X_train_res,y_train_res = SMOTE(random_state=1).fit_resample(X_train,y_train)

    #PREPROCESSING 
    t2=X_train.shape
    y_tr=y_train.ravel()
    li=[]
    for i in range(len(y_tr)):
        if int(y_tr[i])==1:
            li.append(X_train[i])
    X_real=np.array(li)
    t3=X_real.shape
    li2=[1]*(t3[0])
    y_real=np.array(li2)
    X_oversampled=X_train_res[(t2[0]):]
    z_dim = t2[1]
    t4=X_oversampled.shape

    parameter = TuningParameters( randomstate,epochfOne,numberof_epochs,criterion,learningRate,
                                    batch_size,display_step ,device,NN_ep,X,Y,X_train,y_train,X_test,y_test,
                                    X_train_res,y_train_res,X_real,y_real,z_dim,t4)

    GenerateResults(parameter)

def AbaloneProcessing():
    url='https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/abalone_csv.csv'
    data =  pd.read_csv(url, sep=",", header='infer')

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
    features = data.iloc[:, np.r_[0:7]]
    labels = data.iloc[:, 7]

    # X_train, X_test, y_train, y_test, X_gender, X_gender_test = train_test_split(
    #     features, labels, onehot_encoded, random_state=10, test_size=0.2)

    X_train, X_test, y_train, y_test, X_gender, X_gender_test = train_test_split(
        features, labels, onehot_encoded, random_state=10, test_size=0.2)


    temp = X_train.values
    X_train = np.concatenate((temp, X_gender), axis=1)
    temp2 = X_test.values
    X_test = np.concatenate((temp2, X_gender_test), axis=1)
    temp3 = features.values
    features = np.concatenate((temp3, onehot_encoded), axis=1)
    test_list = [int(i) for i in y_test.ravel()]
    y_test = np.array(test_list)
    train_list = [int(i) for i in y_train.ravel()]
    y_train = np.array(train_list)
    #encode the label i.e. transform postive negative to
    randomstate=1
    epochfOne=30
    numberof_epochs=150
    criterion = nn.BCEWithLogitsLoss()
    learningRate = 0.00001
    batch_size = 128
    display_step = 1
    device = 'cpu'
    NN_ep = 30

    # spilt the data between train test
    #X_train, X_test, y_train, y_test = train_test_split(X,encoded_Y, test_size=0.2, random_state=randomstate)
    X_train_res,y_train_res = SMOTE(random_state=2).fit_resample(X_train,y_train)

    class_counts = Counter(y_train)
    # Find the minority class label
    minority_class_label = min(class_counts, key=class_counts.get)
    print(minority_class_label)

    #PREPROCESSING 

    t2=X_train.shape
    y_tr=y_train.ravel()
    li=[]
    for i in range(len(y_tr)):
        if int(y_tr[i])==1:
            li.append(X_train[i])
    X_real=np.array(li)
    t3=X_real.shape
    li2=[1]*(t3[0])
    y_real=np.array(li2)

    X_oversampled=X_train_res[(t2[0]):]
    z_dim = t2[1]

    t4=X_oversampled.shape

    parameter = TuningParameters( randomstate,epochfOne,numberof_epochs,criterion,learningRate,
                                    batch_size,display_step ,device,NN_ep,features,labels,X_train,y_train,X_test,y_test,
                                    X_train_res,y_train_res,X_real,y_real,z_dim,t4)

    GenerateResults(parameter)

def WineQualityProcessing():
    url='Data\wine_quality.dat'
    data = pd.read_csv(url, sep=",", header='infer' )

    t=() 
    #gets the dimension of the array also the element in the array
    t=data.shape 
    X = data.values[:,0:(t[1]-1)].astype(float)
    Y = data.values[:,(t[1]-1)]

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    #encode the label i.e. transform postive negative to 0s and 1s
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)

    # yk = []
    # for i in encoded_Y:
    #     if i == 2:
    #         yk.append(1)
    #     else:
    #         yk.append(0)
    # encoded_Y = np.asarray(yk, dtype=np.int32)

    randomstate=4
    epochfOne=30
    numberof_epochs=1500
    criterion = nn.BCEWithLogitsLoss()
    learningRate = 0.00001
    batch_size = 128
    display_step = 1
    device = 'cpu'
    NN_ep = 30

    # spilt the data between train test
    X_train, X_test, y_train, y_test = train_test_split(X,encoded_Y, test_size=0.2, stratify = encoded_Y ,random_state=randomstate)
    X_train_res,y_train_res = SMOTE(random_state=1).fit_resample(X_train,y_train)

    #PREPROCESSING 
    t2=X_train.shape
    y_tr=y_train.ravel()
    li=[]
    for i in range(len(y_tr)):
        if int(y_tr[i])==1:
            li.append(X_train[i])
    X_real=np.array(li)
    t3=X_real.shape
    li2=[1]*(t3[0])
    y_real=np.array(li2)
    X_oversampled=X_train_res[(t2[0]):]
    z_dim = t2[1]
    t4=X_oversampled.shape

    parameter = TuningParameters( randomstate,epochfOne,numberof_epochs,criterion,learningRate,
                                    batch_size,display_step ,device,NN_ep,X,Y,X_train,y_train,X_test,y_test,
                                    X_train_res,y_train_res,X_real,y_real,z_dim,t4)

    GenerateResults(parameter)

def YeastProcessing():
    
    url='Data\yeast.dat'
    data = pd.read_csv(url, sep=",", header='infer' )
 
    t=() 
    #gets the dimension of the array also the element in the array
    t=data.shape 
    X = data.values[:,0:(t[1]-1)].astype(float)
    Y = data.values[:,(t[1]-1)]

    #encode the label i.e. transform postive negative to 0s and 1s
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    dummy_y = np_utils.to_categorical(encoded_Y)

    randomstate=1
    epochfOne=10
    numberof_epochs=1000
    criterion = nn.BCEWithLogitsLoss()
    learningRate = 0.00001
    batch_size = 128
    display_step = 1
    device = 'cpu'
    NN_ep = 30

    # spilt the data between train test
    X_train, X_test, y_train, y_test = train_test_split(X,encoded_Y, test_size=0.2, stratify = encoded_Y , random_state=randomstate)
    X_train_res,y_train_res = SMOTE().fit_resample(X_train,y_train)

    #PREPROCESSING 
    t2=X_train.shape
    y_tr=y_train.ravel()
    li=[]
    for i in range(len(y_tr)):
        if int(y_tr[i])==1:
            li.append(X_train[i])
    X_real=np.array(li)
    t3=X_real.shape
    li2=[1]*(t3[0])
    y_real=np.array(li2)
    X_oversampled=X_train_res[(t2[0]):]
    z_dim = t2[1]
    t4=X_oversampled.shape

    parameter = TuningParameters( randomstate,epochfOne,numberof_epochs,criterion,learningRate,
                                    batch_size,display_step ,device,NN_ep,X,Y,X_train,y_train,X_test,y_test,
                                    X_train_res,y_train_res,X_real,y_real,z_dim,t4)

    GenerateResults(parameter)

def IonosphereProcessing():
    
    url='Data\ionosphere.csv'
    data = pd.read_csv(url, sep=",", header='infer' )

    t = ()
    t = data.shape
    X = data.values[:, 0:(t[1]-1)].astype(float)
    Y = data.values[:, (t[1]-1)]

    #encode the label i.e. transform postive negative to 0s and 1s
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)

    randomstate=1
    epochfOne=30
    numberof_epochs=1000
    criterion = nn.BCEWithLogitsLoss()
    learningRate = 0.00001
    batch_size = 128
    display_step = 1
    device = 'cpu'
    NN_ep = 30
    # spilt the data between train test
    X_train, X_test, y_train, y_test = train_test_split(X,encoded_Y, test_size=0.2, stratify = encoded_Y , random_state=randomstate)
      
    lst = []
    lst2 = []
    for j in y_train:
        if j == 1:
            lst.append(0)
        else:
            lst.append(1)
    for j2 in y_test:
        if j2 == 1:
            lst2.append(0)
        else:
            lst2.append(1)
    y_train = np.array(lst)
    y_test = np.array(lst2)
    
    X_train_res,y_train_res = SMOTE(random_state=1).fit_resample(X_train,y_train)

    #PREPROCESSING 
    t2=X_train.shape
    y_tr=y_train.ravel()
    li=[]
    for i in range(len(y_tr)):
        if int(y_tr[i])==1:
            li.append(X_train[i])
    X_real=np.array(li)
    t3=X_real.shape
    li2=[1]*(t3[0])
    y_real=np.array(li2)
    X_oversampled=X_train_res[(t2[0]):]
    z_dim = t2[1]
    t4=X_oversampled.shape

    parameter = TuningParameters( randomstate,epochfOne,numberof_epochs,criterion,learningRate,
                                    batch_size,display_step ,device,NN_ep,X,Y,X_train,y_train,X_test,y_test,
                                    X_train_res,y_train_res,X_real,y_real,z_dim,t4)

    GenerateResults(parameter)

def PageBlockProcessing():
    
    url='Data\page_block.dat'
    data = pd.read_csv(url, sep=",", header='infer' )

    t=() 
    #gets the dimension of the array also the element in the array
    t=data.shape 
    X = data.values[:,0:(t[1]-1)].astype(float)
    Y = data.values[:,(t[1]-1)]

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    #encode the label i.e. transform postive negative to 0s and 1s
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)

    randomstate=1
    epochfOne=20
    numberof_epochs=2000
    criterion = nn.BCEWithLogitsLoss()
    learningRate = 0.00001
    batch_size = 128
    display_step = 1
    device = 'cpu'
    NN_ep = 20

    # spilt the data between train test
    X_train, X_test, y_train, y_test = train_test_split(X,encoded_Y, test_size=0.2, stratify = encoded_Y , random_state=randomstate)
    X_train_res,y_train_res = SMOTE(random_state=1).fit_resample(X_train,y_train)

    #PREPROCESSING 
    t2=X_train.shape
    y_tr=y_train.ravel()
    li=[]
    for i in range(len(y_tr)):
        if int(y_tr[i])==1:
            li.append(X_train[i])
    X_real=np.array(li)
    t3=X_real.shape
    li2=[1]*(t3[0])
    y_real=np.array(li2)
    X_oversampled=X_train_res[(t2[0]):]
    z_dim = t2[1]
    t4=X_oversampled.shape

    parameter = TuningParameters( randomstate,epochfOne,numberof_epochs,criterion,learningRate,
                                    batch_size,display_step ,device,NN_ep,X,Y,X_train,y_train,X_test,y_test,
                                    X_train_res,y_train_res,X_real,y_real,z_dim,t4)

    GenerateResults(parameter)

def PokerProcessing():
    
    url='Data\poker.dat'
    data = pd.read_csv(url, sep=",", header='infer' )

    t=() 
    #gets the dimension of the array also the element in the array
    t=data.shape 
    X = data.values[:,0:(t[1]-1)].astype(float)
    Y = data.values[:,(t[1]-1)]

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    #encode the label i.e. transform postive negative to 0s and 1s
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)

    randomstate=1
    epochfOne=20
    numberof_epochs=500
    criterion = nn.BCEWithLogitsLoss()
    learningRate = 0.00001
    batch_size = 128
    display_step = 1
    device = 'cpu'
    NN_ep = 20

    # spilt the data between train test
    X_train, X_test, y_train, y_test = train_test_split(X,encoded_Y, test_size=0.2, stratify = encoded_Y , random_state=randomstate)
    X_train_res,y_train_res = SMOTE(random_state=1).fit_resample(X_train,y_train)

    #PREPROCESSING 
    t2=X_train.shape
    y_tr=y_train.ravel()
    li=[]
    for i in range(len(y_tr)):
        if int(y_tr[i])==1:
            li.append(X_train[i])
    X_real=np.array(li)
    t3=X_real.shape
    li2=[1]*(t3[0])
    y_real=np.array(li2)
    X_oversampled=X_train_res[(t2[0]):]
    z_dim = t2[1]
    t4=X_oversampled.shape

    parameter = TuningParameters( randomstate,epochfOne,numberof_epochs,criterion,learningRate,
                                    batch_size,display_step ,device,NN_ep,X,Y,X_train,y_train,X_test,y_test,
                                    X_train_res,y_train_res,X_real,y_real,z_dim,t4)

    GenerateResults(parameter)

def SpambaseProcessing():
    
    url='Data\spambase.csv'
    data = pd.read_csv(url, sep=",", header='infer' )

    dict_1 = {}
    dict_1 = dict(data.corr()['1'])
    list_features = []
    for key, values in dict_1.items():
        if abs(values) < 0.2:
            list_features.append(key)
    data = data.drop(list_features, axis=1)
    X = data.values[:, 0:19].astype(float)
    Y = data.values[:, 19]


    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    #encode the label i.e. transform postive negative to 0s and 1s
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)

    randomstate=1
    epochfOne=20
    numberof_epochs=500
    criterion = nn.BCEWithLogitsLoss()
    learningRate = 0.00001
    batch_size = 128
    display_step = 1
    device = 'cpu'
    NN_ep = 20

    # spilt the data between train test
    X_train, X_test, y_train, y_test = train_test_split(X,encoded_Y, test_size=0.2, stratify = encoded_Y , random_state=randomstate)
    X_train_res,y_train_res = SMOTE(random_state=1).fit_resample(X_train,y_train)

    #PREPROCESSING 
    t2=X_train.shape
    y_tr=y_train.ravel()
    li=[]
    for i in range(len(y_tr)):
        if int(y_tr[i])==1:
            li.append(X_train[i])
    X_real=np.array(li)
    t3=X_real.shape
    li2=[1]*(t3[0])
    y_real=np.array(li2)
    X_oversampled=X_train_res[(t2[0]):]
    z_dim = t2[1]
    t4=X_oversampled.shape

    parameter = TuningParameters( randomstate,epochfOne,numberof_epochs,criterion,learningRate,
                                    batch_size,display_step ,device,NN_ep,X,Y,X_train,y_train,X_test,y_test,
                                    X_train_res,y_train_res,X_real,y_real,z_dim,t4)

    GenerateResults(parameter)










