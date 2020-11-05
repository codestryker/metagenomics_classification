import torch, gc
import math
import re
import sys
import numpy as np
import pandas as pd
import pickle as pkl
from app.model import model

use_cuda = torch.cuda.is_available()

if not use_cuda:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

path = 'app/bacteria/'

labels_path = path + 'labels.pkl'
labels_file = open(labels_path, 'rb')
classes = pkl.load(labels_file)

sets= {'A','C','G','T','U','W','S','M','K','R','Y','B','D','H','V','N','Z'}
char=sorted(sets)
char_encode={y:x+1 for x,y in enumerate(char)}

taxa=['Kingdom','Phylum','Class','Order','Family','Genus','Species']

models={}
def load():
    for i in taxa:
       models[i]=model.create(len(classes[i])+1 if i=='Kingdom' else len(classes[i]))
       # move tensors to GPU if CUDA is available
       if use_cuda:
         models[i].cuda()
       models[i].load_state_dict(torch.load(path+f'bacteria_{i.lower()}.pt',map_location=lambda storage, loc: storage))

def get_predict(seq):
    
    # load the read and return the predicted taxa
    inputs=0
    seq=seq.replace('\n','').replace('\r','')[:2500]
    read=np.zeros(2500)
    for x,y in enumerate(seq):
        if y not in sets:
            print("Invalid characters")
        read[x]=char_encode[y]

    read=read.reshape(50,50)
    inputs=read/20

    tensor_r = torch.Tensor(inputs).unsqueeze(0).unsqueeze(0)

    if use_cuda:
        tensor_r = tensor_r.cuda()


    taxas=[]
    for i in taxa:
       models[i].eval()
       output= models[i](tensor_r)

       # convert output probabilities to predicted class
       _, pred_tensor=torch.max(output, 1)
       index = np.squeeze(pred_tensor.numpy()) if not use_cuda else np.squeeze(pred_tensor.cpu().numpy())
       taxas.append(classes[i][index])
    return taxas
    

