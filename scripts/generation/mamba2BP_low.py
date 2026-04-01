
# https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/
'''
usually, time seties oreduction is done on a window.
given data from time [t - w,t], you predict t + m where m is any timestep into the future.
w governs how much data you can look at to make a predition, called the look back period.


'''
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import random
import torch.nn.functional as F
import torch.utils.data as data
import torchinfo

from qutils.integrators import myRK4Py, ode87
from qutils.ml.utils import Adam_mini, findDecAcc,printModelParmSize
from qutils.plot import plotOrbitPhasePredictions,plotSolutionErrors, plotStatePredictions,newPlotSolutionErrors

from qutils.orbital import nonDim2Dim4
from qutils.ml.regression import create_datasets, genPlotPrediction, LSTMSelfAttentionNetwork
from qutils.tictoc import timer
from qutils.helper import parse_yaml_config

from qutils.ml.mamba import Mamba, MambaConfig



# seed any random functions
random.seed(123)

# data size set that define amount of data sets we will generate to train the network
DATA_SET_SIZE = 1
TIME_STEP = 0.01
plotOn = False

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

# ------------------------------------------------------------------------
## NUMERICAL SOLUTION

problemDim = 4 

muR = 3.96800e14
DU = 6378.1e3 # radius of earth in km
TU = ((DU)**3 / muR)**0.5

rLEO = np.array([6.611344740000000e+06,0])
vLEO = np.sqrt(muR/np.linalg.norm(rLEO))
vLEO = np.array([0,vLEO])
TLEO = 2*np.pi * np.sqrt(np.linalg.norm(rLEO)**3 / muR)


degreesOfFreedom = problemDim

criterion = F.smooth_l1_loss

# """
# TRANSFER LEARN TO NEW, NONLINEAR SYSTEM ON DIFFERENT INITIAL CONDITIONS AND DIFFERENT TIME PERIOD AND DIFFERENT TIME STEP
# """

TIME_STEP = 0.05

# transfer to different system

# trainableLayer = [True, True, False]
# newModel = transferLSTM(model,newModel,trainableLayer)
hidden_size = 30
config = MambaConfig(d_model=degreesOfFreedom, n_layers=1)

model = Mamba(config).to(device).double()


muR = 396800
DU = 6378.1 # radius of earth in km
TU = ((DU)**3 / muR)**0.5

p = 20410 # km
e = 0.8

a = p/(1-e**2)

rHEO = np.array([(p/(1+e)),0])
vHEO = np.array([0,np.sqrt(muR*((2/rHEO[0])-(1/a)))])
THEO = 2*np.pi*np.sqrt(a**3/muR)
print(THEO)
mu = 1
r = rHEO / DU
v = vHEO * TU / DU
T = THEO / TU
print(TU)
print(T)

J2 = 1.08263e-3

IC = np.concatenate((r,v))
pam = [mu,J2]

m_sat = 1
c_d = 2.1 #shperical model
A_sat = 1.0013 / (DU ** 2)
h_scale = 50 * 1000 / DU
rho_0 = 1.29 * 1000 ** 2 / (DU**2)

def twoBodyPert(t, y, p=pam):
    r = y[0:2]
    R = np.linalg.norm(r)
    v = y[2:4]
    v_norm = np.linalg.norm(v)

    mu = p[0]; J2 = p[1]
    dydt1 = y[2]
    dydt2 = y[3]

    factor =  (- mu) * 1.5 * J2 * (1 / R)**2 / R**3
    j2_accel_x = factor * (1) * r[0]
    j2_accel_y = factor * (3) * r[1]

    rho = rho_0 * np.exp(-R / h_scale)  # Atmospheric density model
    drag_factor = -0.5 * (rho / m_sat) * c_d * A_sat * v_norm
    a_drag_x = drag_factor * y[2]
    a_drag_y = drag_factor *  y[3]

    a_drag_x = 0
    a_drag_y = 0
    # j2_accel_x = 0
    # j2_accel_y = 0
    dydt3 = -mu / R**3 * y[0] + j2_accel_x + a_drag_x
    dydt4 = -mu / R**3 * y[1] + j2_accel_y + a_drag_y

    return np.array([dydt1, dydt2,dydt3,dydt4])

numPeriods = 5

n_epochs = 10
lr = 0.001
input_size = degreesOfFreedom
output_size = degreesOfFreedom
num_layers = 1
p_dropout = 0.0
lookback = 1
p_motion_knowledge = 1/numPeriods

sysfuncptr = twoBodyPert
# sim time
t0, tf = 0, numPeriods * T

t = np.arange(t0, tf, TIME_STEP)

IC = np.concatenate((r,v))

t , numericResult = ode87(sysfuncptr,[t0,tf],IC,t)

output_seq = numericResult

pertNR = numericResult

train_size = int(len(pertNR) * p_motion_knowledge)
train_size = 2
test_size = len(pertNR) - train_size

train_in,train_out,test_in,test_out = create_datasets(pertNR,1,train_size,device)
print(train_in)
print(train_out)

loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=8)


optimizer = torch.optim.Adam(model.parameters(),lr=lr)

trainTime = timer()
for epoch in range(n_epochs):

    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    model.eval()
    with torch.no_grad():
        y_pred_train = model(train_in)
        train_loss = np.sqrt(criterion(y_pred_train, train_out).cpu())
        y_pred_test = model(test_in)
        test_loss = np.sqrt(criterion(y_pred_test, test_out).cpu())

        decAcc, err1 = findDecAcc(train_out,y_pred_train,printOut=False)
        decAcc, err2 = findDecAcc(test_out,y_pred_test)
        err = np.concatenate((err1,err2),axis=0)

    print("Epoch %d: train loss %.4f, test loss %.4f\n" % (epoch, train_loss, test_loss))

timeToTrain=trainTime.tocVal()

networkPrediction,timeToTest = plotStatePredictions(model,t,output_seq,train_in,test_in,train_size,test_size,DU=DU,TU=TU,timeLabel='Periods',outputToc=True)

# plot3dCR3BPPredictions(output_seq,networkPrediction,L=None,earth=False,moon=False)
mambaParams = printModelParmSize(model)

del model
del optimizer
torch.cuda.empty_cache()
import gc
gc.collect()
modelLSTM = LSTMSelfAttentionNetwork(problemDim,hidden_size,problemDim,1, 0).double().to(device)

# optimizer = torch.optim.AdamW(model.parameters(),lr=lr)
optimizer = Adam_mini(modelLSTM,lr=lr)

criterion = F.smooth_l1_loss
# criterion = torch.nn.HuberLoss()
trainTime = timer()
for epoch in range(n_epochs):

    # trajPredition = plotPredition(epoch,model,'target',t=t*TU,output_seq=pertNR)

    modelLSTM.train()
    for X_batch, y_batch in loader:
        y_pred = modelLSTM(X_batch)
        loss = criterion(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    modelLSTM.eval()
    with torch.no_grad():
        y_pred_train = modelLSTM(train_in)
        train_loss = np.sqrt(criterion(y_pred_train, train_out).cpu())
        y_pred_test = modelLSTM(test_in)
        test_loss = np.sqrt(criterion(y_pred_test, test_out).cpu())

        decAcc, err1 = findDecAcc(train_out,y_pred_train,printOut=False)
        decAcc, err2 = findDecAcc(test_out,y_pred_test)
        err = np.concatenate((err1,err2),axis=0)

    print("Epoch %d: train loss %.4f, test loss %.4f\n" % (epoch, train_loss, test_loss))

timeToTrainLSTM=trainTime.tocVal()

networkPredictionLSTM,timeToTestLSTM = plotStatePredictions(modelLSTM,t,output_seq,train_in,test_in,train_size,test_size,DU=DU,TU=TU,timeLabel='Periods',outputToc=True)

t = t / T

pertNR = nonDim2Dim4(pertNR)

plotOrbitPhasePredictions(pertNR,networkPrediction)
plotOrbitPhasePredictions(pertNR,networkPredictionLSTM)


plt.grid()
plt.tight_layout()
# plotSolutionErrors(pertNR,networkPrediction,t)
newPlotSolutionErrors(pertNR,networkPrediction,t,timeLabel='Periods',percentError=True,states = ['x', 'y', '$\dot{x}$', '$\dot{y}$'])


err = nonDim2Dim4(err)
lstmParams = printModelParmSize(modelLSTM)
torchinfo.summary(modelLSTM)
print(numericResult[0,:])
print(numericResult[1,:])
errorAvg = np.nanmean(abs(networkPrediction-pertNR), axis=0)
print("Average values of each dimension:")
for i, avg in enumerate(errorAvg, 1):
    print(f"Dimension {i}: {avg}")

output_seq = nonDim2Dim4(output_seq)

config = parse_yaml_config("vars.yaml")

dataLoc = config["data-folder"]

np.savez(dataLoc+'/2bp_low.npz', trueTraj=output_seq,networkPredictionMamba = networkPrediction,networkPredictionLSTM=networkPredictionLSTM
         ,delT=TIME_STEP,tf=tf,t=t,timeToTrainMamba=timeToTrain,timeToTrainLSTM=timeToTrainLSTM,timeToTestMamba=timeToTest,timeToTestLSTM=timeToTestLSTM
         ,paramsMamba = mambaParams, paramsLSTM = lstmParams,
         d_units = "[km]", t_units = "[days]",
         train_size = train_size, test_size = test_size,)


if plotOn:
    plt.show()
