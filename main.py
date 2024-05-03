import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from torch import optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cross_decomposition import PLSRegression
import scipy.ndimage
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
#create a model class that inherits nn.Module



class Model(nn.Module):
    # Input layer (1701 features) --> Hidden layer1 (p neurons) --> Hidden layer2 (m neurons) --> Output
    def __init__(self, in_features=1701, h1=100, h2=100, h3=100, out_features=1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1,h2)
        self.fc3 = nn.Linear(h2,h3)       
        self.out = nn.Linear(h3,out_features)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.out(x))

        return x

torch.manual_seed(41)



#'n.tot_usda.a623_w.pct'p.ext_aquaregia_mg.kg', 'p.ext_usda.a1070_mg.kg',
#       'p.ext_usda.a270_mg.kg', 'p.ext_usda.a274_mg.kg',
#       'ph.cacl2_usda.a481_index', 'ph.h2o_usda.a268_index',
 #      's.ext_mel3_mg.kg', 's.tot_usda.a624_w.pct'

#import the global dataset
glob_dat = pd.read_csv('glob_dat.csv')

###dataset reduction###############################################################
#get the preparation methods
prep_types = glob_dat['scan.mir.method.preparation_any_txt'].unique()

#get rid of the samples with unknown prep method
glob_dat_prep = glob_dat[glob_dat['scan.mir.method.preparation_any_txt'].notna()]

#get rid of the samples without oc measurment
glob_dat_prep_oc = glob_dat_prep[glob_dat_prep['oc_usda.c729_w.pct'].notna()]


#get rid of the samples without MIR measurment
glob_dat_prep_oc_mir = glob_dat_prep_oc[glob_dat_prep_oc['scan_mir.628_abs'].notna()]

####################################################################################
###global model training############################################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#get the MIR spectra
glob_MIR = glob_dat_prep_oc_mir.iloc[:,1185:]

scaler = MinMaxScaler()
glob_MIR = scaler.fit_transform(glob_MIR)

import joblib
scaler_filename = "scaler.save"
joblib.dump(scaler, scaler_filename) 


#get the OC
glob_OC = glob_dat_prep_oc_mir['oc_usda.c729_w.pct']

torch.cuda.empty_cache()


X_train, X_test, y_train, y_test = train_test_split(glob_MIR, glob_OC, test_size=0.2, random_state=41)

#convert X features to float tensors
X_train = torch.FloatTensor(glob_MIR)
#X_test = torch.FloatTensor(X_test.to_numpy())
X_train = X_train.to(device)
#X_test = X_test.to(device)


#convert y to tensors
y_train = torch.FloatTensor(glob_OC.to_numpy())
#y_test = torch.FloatTensor(y_test.to_numpy())
y_train = y_train.to(device)
#y_test = y_test.to(device)

model = Model()

model = model.to(device)

#set model criterion
criterion = nn.MSELoss()


#Epochs 
epochs = 500000

#choose ADAM optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)

losses = []
for i in range(epochs):
    #Go forward
    y_pred = model.forward(X_train) #get predicted results
    #measure the loss 
    loss = criterion(y_pred.squeeze(1),y_train)
    losses.append(loss.cpu().detach().numpy())

    #print every 10 epochs
    if i % 10 == 0:
        print(f'Epoch: {i} and loss: {loss}')

    #do back prop

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

wave_test = pd.read_excel('wavenumbers.xlsx',header=None)

glob_dat.columns[1185]
glob_dat.columns[-1]
np.argmax(np.array(wave_test)<4002)

X_test = pd.read_excel("loc_mir.xlsx",header=None)

X_test = X_test.to_numpy()
X_test = X_test[:,2085:]
X_test = np.flip(X_test,axis=1)
np.savetxt('X.csv',X_test,delimiter=',')

X_test = scipy.ndimage.zoom(X_test, (1,1701/X_test.shape[1]),order=5)

X_test = scaler.transform(X_test)

X_test = torch.FloatTensor(X_test)
X_test = X_test.to(device)

y_test =  pd.read_excel("loc_ref.xlsx")
y_test = y_test['OC']


with torch.no_grad():
    y_eval = model(X_test)
    r2 = r2_score(y_test,y_eval.cpu().detach().numpy())
    rmse = np.sqrt(mean_squared_error(y_test,y_eval.cpu().detach().numpy()))

torch.save(model.state_dict(),'state_dict.model')