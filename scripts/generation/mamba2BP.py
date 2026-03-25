import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchinfo

from qutils.integrators import ode87
from qutils.plot import plot3dOrbitPredictions,plotOrbitPhasePredictions, plotSolutionErrors,plotPercentSolutionErrors, plotEnergy,plotStatePredictions,newPlotSolutionErrors
from qutils.orbital import nonDim2Dim6, returnCR3BPIC, readGMATReport, dim2NonDim6, orbitalEnergy
from qutils.ml.mamba import Mamba, MambaConfig
from qutils.ml.utils import printModelParmSize, getDevice, Adam_mini
from qutils.ml.regression import trainModel, create_datasets, genPlotPrediction, LSTMSelfAttentionNetwork
from qutils.tictoc import timer
# from nets import Adam_mini

# from memory_profiler import profile
from qutils.ml.superweight import printoutMaxLayerWeight,getSuperWeight,plotSuperWeight, findMambaSuperActivation, plotSuperActivation

plotOn = False
printoutSuperweight = True
compareLSTM = False
percentRMSE = True

problemDim = 6

device = getDevice()

gmatImport = readGMATReport("gmat/data/reportHEO360Prop.txt")
# gmatImport = readGMATReport("gmat/data/reportLEO5050Prop.txt")
semimajorAxis = 67903.82797675686
tPeriod = 175587.6732104912
# gmat propagation uses 50/70 50/70 JGM-2 with MSISE90 spherical drag model w/ SRP

t = gmatImport[:,-1]

output_seq = gmatImport[:,0:problemDim]

muR = 396800
DU = 6378.1 # radius of earth in km
TU = ((DU)**3 / muR)**0.5

output_seq = dim2NonDim6(output_seq,DU,TU)
print(output_seq[0,:])
# hyperparameters
n_epochs = 5
# lr = 5*(10**-5)
# lr = 0.85
lr = 0.8
lr = 0.08
lr = 0.004
lr = 0.001
lr = 0.01
input_size = problemDim
output_size = problemDim
num_layers = 1
lookback = 1
p_motion_knowledge = 0.5


train_size = int(len(output_seq) * p_motion_knowledge)
# train_size = 2
test_size = len(output_seq) - train_size
print(train_size)
print(test_size)
train_in,train_out,test_in,test_out = create_datasets(output_seq,1,train_size,device)

# initilizing the model, criterion, and optimizer for the data
config = MambaConfig(d_model=problemDim, n_layers=num_layers,d_conv=32)

def returnModel(modelString = 'mamba'):
    if modelString == 'mamba':
        model = Mamba(config).to(device).double()
    elif modelString == 'lstm':
        model = LSTMSelfAttentionNetwork(input_size,30,output_size,num_layers,0).double().to(device)
    return model

model = returnModel()

# optimizer = torch.optim.AdamW(model.parameters(),lr=lr)
optimizer = Adam_mini(model,lr=lr)

criterion = F.smooth_l1_loss
# criterion = torch.nn.HuberLoss()

timeToTrain = trainModel(model,n_epochs,[train_in,train_out,test_in,test_out],criterion,optimizer,printOutAcc = True,printOutToc = True)

networkPrediction, testTime = plotStatePredictions(model,t,output_seq,train_in,test_in,train_size,test_size,DU=DU,TU=TU,outputToc=True)
output_seq = nonDim2Dim6(output_seq,DU,TU)

# networkPrediction = nonDim2Dim6(networkPrediction,DU,TU)

plotOrbitPhasePredictions(output_seq,networkPrediction)
plotOrbitPhasePredictions(output_seq,networkPrediction,plane='xz')
plotOrbitPhasePredictions(output_seq,networkPrediction,plane='yz')


plot3dOrbitPredictions(output_seq,networkPrediction)

print('total prop time',gmatImport[-1,-1])

# networkPrediction = nonDim2Dim6(networkPrediction,DU,TU)
# output_seq = nonDim2Dim6(output_seq,DU,TU)

# plotOrbitPredictions(output_seq,networkPrediction,t=t)
plotSolutionErrors(output_seq,networkPrediction,t/tPeriod)
plotPercentSolutionErrors(output_seq,networkPrediction,t/tPeriod,semimajorAxis,max(np.linalg.norm(gmatImport[:,3:6],axis=1)))
plotEnergy(output_seq,networkPrediction,t/tPeriod,orbitalEnergy,xLabel='Number of Periods (T)',yLabel='Specific Energy')
# plotDecAccs(decAcc,t,problemDim)

from qutils.mlExtras import rmse

rmseMamba = rmse(output_seq,networkPrediction,percentRMSE=True)



errorAvg = np.nanmean(abs(networkPrediction-output_seq), axis=0)
print("Average values of each dimension:")
for i, avg in enumerate(errorAvg, 1):
    print(f"Dimension {i}: {avg}")

printModelParmSize(model)
torchinfo.summary(model)

if printoutSuperweight:
    printoutMaxLayerWeight(model)
    getSuperWeight(model)
    plotSuperWeight(model)

    magnitude, index = findMambaSuperActivation(model,test_in)
    plotSuperActivation(magnitude, index)

del model
del optimizer
torch.cuda.empty_cache()
import gc
gc.collect()
modelLSTM = returnModel('lstm')

# optimizer = torch.optim.AdamW(model.parameters(),lr=lr)
optimizer = Adam_mini(modelLSTM,lr=lr)

criterion = F.smooth_l1_loss
# criterion = torch.nn.HuberLoss()
timeToTrainLSTM = trainModel(modelLSTM,n_epochs,[train_in,train_out,test_in,test_out],criterion,optimizer,printOutAcc = True,printOutToc = True)

output_seq = dim2NonDim6(output_seq,DU,TU)

networkPredictionLSTM, testTimeLSTM = plotStatePredictions(modelLSTM,t,output_seq,train_in,test_in,train_size,test_size,DU=DU,TU=TU,outputToc=True)
output_seq = nonDim2Dim6(output_seq,DU,TU)

plot3dOrbitPredictions(output_seq,networkPrediction,earth=False,networkLabel="Mamba")
plt.plot(networkPredictionLSTM[:, 0], networkPredictionLSTM[:, 1], networkPredictionLSTM[:, 2], label='LSTM',linestyle='dashed')
plt.plot(0,0,0,"ko",label="Earth")
plt.legend(fontsize=10)
plt.tight_layout()

plotOrbitPhasePredictions(output_seq,networkPredictionLSTM)
plotOrbitPhasePredictions(output_seq,networkPredictionLSTM,plane='xz')
plotOrbitPhasePredictions(output_seq,networkPredictionLSTM,plane='yz')

plotSolutionErrors(output_seq,networkPredictionLSTM,t/tPeriod)

fig, axes = newPlotSolutionErrors(output_seq,networkPredictionLSTM,t,timeLabel="Orbit Periods",percentError=True,states = ['x', 'y', 'z', '$\dot{x}$', '$\dot{y}$', '$\dot{z}$'])
newPlotSolutionErrors(output_seq,networkPrediction,t,timeLabel="Orbit Periods",newPlot=axes,networkLabels=["LSTM","Mamba"],percentError=True,states = ['x', 'y', 'z', '$\dot{x}$', '$\dot{y}$', '$\dot{z}$'])
mambaLine = mlines.Line2D([], [], color='b', label='LSTM')
LSTMLine = mlines.Line2D([], [], color='orange', label='Mamba')
fig.legend(handles=[mambaLine,LSTMLine])
# fig.tight_layout()
fig.set_size_inches(12, 8)  # Adjust the figure size here (width, height)

# plotPercentSolutionErrors(output_seq,networkPredictionLSTM,t/tPeriod,semimajorAxis,max(np.linalg.norm(gmatImport[:,3:6],axis=1)))

plotEnergy(output_seq,networkPrediction,t/tPeriod,orbitalEnergy,xLabel='Number of Periods (T)',yLabel='Specific Energy')
plt.plot(t/tPeriod,orbitalEnergy(networkPredictionLSTM),label='LSTM',linestyle='dashed')
plt.legend()

rmseLSTM = rmse(output_seq,networkPredictionLSTM,percentRMSE=True)


errorAvg = np.nanmean(abs(networkPredictionLSTM-output_seq), axis=0)
print("Average values of each dimension:")
for i, avg in enumerate(errorAvg, 1):
    print(f"Dimension {i}: {avg}")

printModelParmSize(modelLSTM)
torchinfo.summary(modelLSTM)


from qutils.helper import parse_yaml_config

config = parse_yaml_config("vars.yaml")

dataLoc = config["data-folder"]


np.savez(dataLoc+"/2bp.npz", trueTraj=output_seq,networkPredictionMamba = networkPrediction,networkPredictionLSTM=networkPredictionLSTM,t=t,tf=t[-1],delT=None)

if plotOn is True:
    plt.show()
