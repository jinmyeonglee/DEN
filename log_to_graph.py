import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

dbe_log = open("dbe_train.log", "r")
smdbe_log = open("smdbe_train.log", "r")

dbe_train_loss = []
dbe_train_rmse = []
dbe_val_loss = []
dbe_val_rmse = []
smdbe_train_loss = []
smdbe_train_rmse = []
smdbe_val_loss = []
smdbe_val_rmse = []
while True:
    line = dbe_log.readline()
    if not line:
        break
    words = line.split(' ')
    if 'INFO:root:Training:' in words :
        for i in range(len(words)):
            if words[i] == 'Loss:':
                dbe_train_loss.append(float(words[i+1]))
            if words[i] == 'RMSE:':
                dbe_train_rmse.append(float(words[i+1]))
    
    elif 'INFO:root:Validation:' in words :
        for i in range(len(words)):
            if words[i] == 'Loss:':
                dbe_val_loss.append(float(words[i+1]))
            if words[i] == 'RMSE:':
                dbe_val_rmse.append(float(words[i+1]))

while True:
    line = smdbe_log.readline()
    if not line:
        break
    words = line.split(' ')
    if 'INFO:root:Training:' in words :
        for i in range(len(words)):
            if words[i] == 'Loss:':
                smdbe_train_loss.append(float(words[i+1]))
            if words[i] == 'RMSE:':
                smdbe_train_rmse.append(float(words[i+1]))
    
    elif 'INFO:root:Validation:' in words :
        for i in range(len(words)):
            if words[i] == 'Loss:':
                smdbe_val_loss.append(float(words[i+1]))
            if words[i] == 'RMSE:':
                smdbe_val_rmse.append(float(words[i+1]))

plt.xlabel("Epoch") 
plt.ylabel("RMSE")

epoch_dbe = [i for i in range(len(dbe_val_rmse))]
epoch_smdbe = [i for i in range(len(smdbe_val_rmse))]

plt.plot(epoch_dbe,dbe_val_rmse, label='DBE')
plt.plot(epoch_smdbe, smdbe_val_rmse, "r-", label = 'DBE + Smoothness')

plt.legend()
plt.show()      