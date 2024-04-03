###### In this file I will basically copy the content inside the the notebook hypertune_pll.
###### This way I can run on tmux, without having to be concerned about connection issues. 


################################################################################################
############################### LOADING LIBRARIES ##############################################
################################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys

sys.path.insert(1, "./util")
sys.path.insert(1, "./model")
sys.path.insert(1, "./esm/")
#

import pickle
from encoded_protein_dataset_new import get_embedding, EncodedProteinDataset_new#, collate_fn_new#, dynamic_collate_fn
from dynamic_loader import dynamic_collate_fn, collate_fn_old
from pseudolikelihood import get_npll2, get_npll_indep
from pseudolikelihood import get_npll2, get_npll, get_npll3
import torch, torchvision
from torch.nn.utils import clip_grad_norm_
from potts_decoder import PottsDecoder
from torch.utils.data import DataLoader, RandomSampler
from functools import partial
import biotite.structure
from biotite.structure.io import pdbx, pdb
from biotite.structure.residues import get_residues
from biotite.structure import filter_backbone
from biotite.structure import get_chains
from biotite.sequence import ProteinSequence
from typing import Sequence, Tuple, List
import scipy
from tqdm import tqdm
import csv
import time
from torch.utils.tensorboard import SummaryWriter


import esm
from esm.inverse_folding import util
import esm.pretrained as pretrained
from ioutils import read_fasta, read_encodings
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
from Bio import SeqIO

from dynamic_loader import dynamic_collate_fn, dynamic_cluster
import optuna

####################################################################################################
################################## LOAD THE DATA ###################################################
####################################################################################################
### IDEA: MSAS PROCEDURE CAN GIVE DIFFERENT OUTPUT SHAPES? ASK
max_msas = None
#msa_dir = "/media/luchinoprince/b1715ef3-045d-4bdf-b216-c211472fb5a2/Data/InverseFolding/msas/"
msa_dir = "/home/silval/split2/"
encoding_dir ="/home/silval/structure_encodings/"

train_dataset = EncodedProteinDataset_new(os.path.join(msa_dir, 'train'), encoding_dir, noise=0.02, max_msas=max_msas)          ## Default value of noise used
sequence_test_dataset = EncodedProteinDataset_new(os.path.join(msa_dir, 'test/sequence'), encoding_dir, noise=0.0, max_msas=max_msas)
structure_test_dataset = EncodedProteinDataset_new(os.path.join(msa_dir, 'test/structure'), encoding_dir, noise=0.0, max_msas=max_msas)
superfamily_test_dataset = EncodedProteinDataset_new(os.path.join(msa_dir, 'test/superfamily'), encoding_dir, noise=0.0, max_msas=max_msas)
print(f"I have loaded the train and test datasets: train:{len(train_dataset)}, seq:{len(sequence_test_dataset)}, struc:{len(structure_test_dataset)}, super:{len(superfamily_test_dataset)}")

batch_structure_size_train = 8   ### I think with empty GPU we can go up to 16 easily
batch_structure_size= 8          ### 8 should still be manageable   
perc_subset_test = 1.0     ## During the training, for every dataset available we select a random 10% of its samples
batch_msa_size = 64 
q = 21 


#collate_fn = partial(collate_fn_old, q=q, batch_size=batch_structure_size_train, batch_msa_size=batch_msa_size)
collate_fn = partial(dynamic_collate_fn, q=q, batch_size=batch_structure_size_train, batch_msa_size=batch_msa_size)
train_loader = DataLoader(train_dataset, batch_size=batch_structure_size_train, collate_fn=collate_fn, shuffle=True,
num_workers=1, pin_memory=True)


sequence_test_loader = DataLoader(sequence_test_dataset, batch_size=batch_structure_size, collate_fn=collate_fn, shuffle=False, 
num_workers=4, pin_memory=True, sampler=RandomSampler(sequence_test_dataset, replacement=True, num_samples=int(0.1*len(sequence_test_dataset))))

structure_test_loader = DataLoader(structure_test_dataset, batch_size=batch_structure_size, collate_fn=collate_fn, shuffle=False, 
num_workers=4, pin_memory=True, sampler=RandomSampler(structure_test_dataset, replacement=True, num_samples=int(perc_subset_test*len(structure_test_dataset))))

superfamily_test_loader = DataLoader(superfamily_test_dataset, batch_size=batch_structure_size, collate_fn=collate_fn, shuffle=False, 
num_workers=4, pin_memory=True, sampler=RandomSampler(superfamily_test_dataset, replacement=True, num_samples=int(perc_subset_test*len(superfamily_test_dataset)))) 


#######################################################################################################
######################################  MODEL FIXED PARAMS ############################################
#######################################################################################################

decoder = None
embedding = None
torch.cuda.empty_cache()

seed = 24877
torch.random.manual_seed(seed)
np.random.seed(seed)

bk_iter = False                                                  

input_encoding_dim = 512
param_embed_dim = 512
n_param_heads = 48
d_model = 512 
n_heads = 8 
n_layers = 6
      
dropout = 0.1
### Embeddings function
embedding = get_embedding(q)
device = 2

decoder = PottsDecoder(q, n_layers, d_model, input_encoding_dim, param_embed_dim, n_heads, n_param_heads, dropout=dropout)
decoder.to(device)
embedding = get_embedding(q)
embedding.to(device)


####################################################################################################
##################################### LOSS COMPUTATION AND BATCHER #################################
####################################################################################################

def get_loss(decoder, inputs, eta_J, eta_h):
    """eta is the multiplicative term in front of the penalized negative pseudo-log-likelihood"""
    msas, encodings, padding_mask  = [input.to(device) for input in inputs]
    B, M, N = msas.shape
    #print(f"encodings' shape{encodings.shape}, padding mask:{padding_mask.shape}")
    couplings, fields = decoder.forward_ardca(encodings, padding_mask)
    ### I want to rescale the couplings by position. Very ineffiecient at the moment
    #with torch.no_grad():
    aux1 = torch.tensor(np.arange(N), dtype=torch.float).reshape(N,1)
    aux1[0] = 1.0
    #aux2 = torch.ones(1,q, requires_grad=False)
    aux1 = torch.matmul(aux1, torch.ones(1,q))
    #aux_flat=
    aux1 = torch.matmul(aux1.reshape(N*q,1), torch.ones(1,N*q))
    aux1=torch.einsum('i,jk->ijk', torch.ones(B), aux1).to(device)
    ### AUX1 SHOULD BE [B, Nq, Nq]

    couplings = couplings/aux1

    
    # embed and reshape to (B, M, N*q)
    msas_embedded = embedding(msas).view(B, M, -1)

    # get npll
    npll = get_npll3(msas_embedded, couplings, fields, N, q)
    padding_mask_inv = (~padding_mask)

    # multiply with the padding mask to filter non-existing residues (this is probably not necessary)       
    #npll = npll * padding_mask_inv.unsqueeze(1)
    npll = npll.reshape((B,M,-1)) * padding_mask_inv.unsqueeze(1)
    penalty = eta_J*(torch.sum(couplings**2))/B + eta_h*(torch.sum(fields**2))/B

    # the padding mask does not contain the msa dimension so we need to multiply by M
    npll_mean = torch.sum(npll) / (M * torch.sum(padding_mask_inv))
    loss_penalty = npll_mean + penalty
    return loss_penalty, npll_mean.item() 


def get_loss_loader(decoder, loader, eta_J, eta_h):
    decoder.eval()
    losses = []
    #with torch.no_grad():
    for effective_batch_size, inputs_packed in loader:
        npll_full = 0
        for inputs in inputs_packed:
            mini_batch_size = inputs[0].shape[0]
            #_, npll = get_loss_indep(decoder, inputs, eta_J, eta_h) ## For independent model without couplings
            _, npll = get_loss(decoder, inputs, eta_J, eta_h)
            npll_full += npll*mini_batch_size/effective_batch_size
        losses.append(npll_full)
            #del inputs
    
    return np.mean(losses)

def train(decoder, inputs_packed, eta_J, eta_h, optimizer, scaler):
    effective_batch_size = inputs_packed[0]
    loss_penalty_full = 0
    train_loss_full = 0
    optimizer.zero_grad(set_to_none=True)                           ## set previous gradients to 0
    with torch.cuda.amp.autocast():  ## autocasting mixed precision
        for inputs in inputs_packed[1]:
            mini_batch_size = inputs[0].shape[0]
            #loss_penalty, train_batch_loss = get_loss_indep(decoder, inputs, eta_J, eta_h)    ## get the current loss for the batch this is for the independent training
            loss_penalty, train_batch_loss = get_loss(decoder, inputs, eta_J, eta_h)
            loss_penalty = loss_penalty * mini_batch_size/effective_batch_size
            train_batch_loss = train_batch_loss * mini_batch_size/effective_batch_size
            #loss_penalty.backward()                         ## Get gradients
            scaler.scale(loss_penalty).backward()
            loss_penalty_full += loss_penalty.detach()
            train_loss_full += train_batch_loss
    
    
    scaler.step(optimizer)
    scaler.update()
    #optimizer.step()   

    return loss_penalty_full, train_loss_full

def train_stable(decoder, inputs_packed, eta_J, eta_h, optimizer, scaler):
    """This training does not use an autoscaler/mixed precision."""
    effective_batch_size = inputs_packed[0]
    loss_penalty_full = 0
    train_loss_full = 0
    optimizer.zero_grad(set_to_none=True)                           ## set previous gradients to 0
    #with torch.cuda.amp.autocast():  ## autocasting mixed precision
    for inputs in inputs_packed[1]:
        mini_batch_size = inputs[0].shape[0]
        #loss_penalty, train_batch_loss = get_loss_indep(decoder, inputs, eta_J, eta_h)    ## get the current loss for the batch this is for the independent training
        loss_penalty, train_batch_loss = get_loss(decoder, inputs, eta_J, eta_h)
        loss_penalty = loss_penalty * mini_batch_size/effective_batch_size
        train_batch_loss = train_batch_loss * mini_batch_size/effective_batch_size
        loss_penalty.backward()                         ## Get gradients
        #scaler.scale(loss_penalty).backward()
        loss_penalty_full += loss_penalty.detach()
        train_loss_full += train_batch_loss
    
    
    #scaler.step(optimizer)
    #scaler.update()
    optimizer.step()   

    return loss_penalty_full, train_loss_full


summary_writer = SummaryWriter()
layout = {
    "metrics": {
        "loss": ["Multiline", ["loss/train", "loss/sequence", "loss/structure", "loss/superfamily"]],}
}
summary_writer.add_custom_scalars(layout)

## Let us also save the hyperparameters
import warnings
warnings.filterwarnings("ignore")
lr = 1.27e-4  #0.00012677192803379938 
optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr)
eta_J = 2.15e-6 #hypertuned: 3.15e-6 #1e-4#
eta_h = 6.51e-4 #hypertuned: 5.04e-5 #1e-4#


hyperparams = {'lr':lr, 'eta_J':eta_J, 'eta_h':eta_h, 'batch_size':batch_structure_size, 'batch_msa_size':batch_msa_size, 'n_param_heads':n_param_heads, 'n_layers':n_layers, 
                'dropout':dropout, 'param_embed_dim':param_embed_dim, 'n_heads': n_heads, 'model':'ardca'}

target_epochs = 100
update_steps = target_epochs * (int(len(train_dataset)/batch_structure_size_train)+1)

start = time.time()
scaler = torch.cuda.amp.GradScaler()
with tqdm(total = update_steps) as pbar: ##This is used to have the nice loading bar while training
    train_loss = 0
    update_step = 0
    max_gpu=0
    train_batch_losses = []
    epoch = 0.0
    while update_step < update_steps:
        for inputs_packed in train_loader:
            decoder.train()

            loss_penalty, train_batch_loss = train_stable(decoder, inputs_packed, eta_J, eta_h, optimizer, scaler)
            loss_penalty.detach()
            #optimizer.step()                                ## Do a step of GD
            update_step += 1                                ## Increase update step (the update steps will count also different batches within the same epoch)
            epoch = update_step / len(train_loader)
            
            train_batch_losses.append(train_batch_loss) ## Here we append the lossess in the different batches within the same epoch
            
            if (epoch % 50 == 0):
                bk_dir= '/home/silval/bk_models/'
                fname_par = 'model_22_01_2024_epoch_ardca_scaled_' + str(epoch) + '.pt'

                ##Arguments of the model, could be inferred
                args_run = {}
                args_run['n_layers'] = n_layers
                args_run['input_encoding_dim'] = input_encoding_dim
                args_run['param_embed_dim'] = param_embed_dim
                args_run['n_heads'] = n_heads
                args_run['n_param_heads'] = n_param_heads
                args_run['dropout'] = dropout

                d = {}
                d['epoch'] = epoch
                d['update_step'] = update_step
                d['batch_size'] = batch_structure_size
                d['seed'] = seed
                d['eta_J'] = eta_J
                d['eta_h'] = eta_h
                d['noise'] = 0.02
                d['args_run'] = args_run
                d['model_state_dict'] = decoder.state_dict()
                d['optimizer_state_dict'] = optimizer.state_dict()

                torch.save(d, os.path.join(bk_dir, fname_par))
            
            
            ## We want to keep track of the test loss not at every batch, too costrly otherwise. Usually set to once every 100.
            if (update_step  == 1) or (epoch % 10 == 0):
                train_loss = np.mean(train_batch_losses)
                with torch.no_grad():
                    sequence_test_loss = get_loss_loader(decoder, sequence_test_loader, eta_J, eta_h)
                    structure_test_loss = get_loss_loader(decoder, structure_test_loader, eta_J, eta_h)
                    superfamily_test_loss = get_loss_loader(decoder, superfamily_test_loader, eta_J, eta_h)

                summary_writer.add_scalar('loss/train', train_loss, update_step)
                summary_writer.add_scalar('loss/sequence', sequence_test_loss, update_step)
                summary_writer.add_scalar('loss/structure', structure_test_loss, update_step)
                summary_writer.add_scalar('loss/superfamily', superfamily_test_loss, update_step)

                ## UNCOMMENT THIS!
                train_batch_losses = []

        
            pbar.set_description(f'update_step: {update_step}, epoch: {epoch:.2f}  train: {train_loss:.2f}, sequence: {sequence_test_loss:.2f}, structure: {structure_test_loss:.2f}, superfamily: {superfamily_test_loss:.2f}')
            pbar.update(1)


####################### SAVE FINAL MODEL #############################
bk_dir= '/home/silval/bk_models/'
fname_par = 'model_22_01_2024_epoch_ardca_scaled_final.pt'

##Arguments of the model, could be inferred
args_run = {}
args_run['n_layers'] = n_layers
args_run['input_encoding_dim'] = input_encoding_dim
args_run['param_embed_dim'] = param_embed_dim
args_run['n_heads'] = n_heads
args_run['n_param_heads'] = n_param_heads
args_run['dropout'] = dropout

d = {}
d['epoch'] = epoch
d['update_step'] = update_step
d['batch_size'] = batch_structure_size
d['seed'] = seed
d['eta_J'] = eta_J
d['eta_h'] = eta_h
d['noise'] = 0.02
d['args_run'] = args_run
d['model_state_dict'] = decoder.state_dict()
d['optimizer_state_dict'] = optimizer.state_dict()

torch.save(d, os.path.join(bk_dir, fname_par))
########################################################################

print(f"It took {time.time()-start} seconds")
save_metrics = {'loss/train': train_loss, 'loss/sequence': sequence_test_loss, 
'loss/structure': structure_test_loss, 'loss/superfamily': superfamily_test_loss}
summary_writer.add_hparams(hyperparams, save_metrics)
summary_writer.close()