import os, sys
import numpy as np
sys.path.insert(1, "./../../util")
sys.path.insert(1, "./../../model")
from encoded_protein_dataset_new import get_embedding, EncodedProteinDataset_new
from pseudolikelihood import get_npll2
from dynamic_loader import collate_fn_old
import torch, torchvision

from potts_decoder import PottsDecoder
from torch.utils.data import DataLoader, RandomSampler
from functools import partial
from typing import Sequence, Tuple, List
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter


from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict

## Set maximum number of threads for pytorch if needed
#torch.set_num_threads(4)


###################################### SOME FUNCTIONS WE NEED #######################################
def get_loss_new(decoder, inputs, eta_J, eta_h):
    """eta_J is the multiplicative term in front of the penalized negative pseudo-log-likelihood for the Couplings
        if no penalty is proveded for the fields, we put the equal to one another """
    msas, encodings, padding_mask  = [input.to(device, non_blocking=True) for input in inputs]
    B, M, N = msas.shape
    param_embeddings, fields = decoder.forward_new(encodings, padding_mask)
    msas_embedded = embedding(msas)
    npll = get_npll2(msas_embedded, param_embeddings, fields, N, q)
    padding_mask_inv = (~padding_mask)

    # Multiply with the padding mask to filter non-existing residues        
    npll = npll * padding_mask_inv.unsqueeze(1)
    npll_mean = torch.sum(npll) / (M * torch.sum(padding_mask_inv))
    
    Q = torch.einsum('bkuia, buhia->bkhia', 
                param_embeddings.unsqueeze(2), param_embeddings.unsqueeze(1)).sum(axis=-1)
    penalty = eta_J/B*(torch.sum(torch.sum(Q,axis=-1)**2) - torch.sum(Q**2)) + eta_h/B*torch.sum(fields**2) 
    loss_penalty = npll_mean + penalty
    return loss_penalty, npll_mean.item() 


def get_loss_loader(decoder, loader, eta_J, eta_h):
    decoder.eval()
    losses = []
    for effective_batch_size, inputs_packed in loader:
        npll_full = 0
        for inputs in inputs_packed:
            mini_batch_size = inputs[0].shape[0]
            _, npll = get_loss_new(decoder, inputs, eta_J, eta_h)
            npll_full += npll*mini_batch_size/effective_batch_size
        losses.append(npll_full)
    
    return np.mean(losses)

def train(decoder, inputs_packed, eta_J, eta_h, optimizer, scaler):
    effective_batch_size = inputs_packed[0]
    loss_penalty_full = 0
    train_loss_full = 0
    optimizer.zero_grad(set_to_none=True)                           ## set previous gradients to 0
    with torch.cuda.amp.autocast():  ## autocasting mixed precision to boost speed
        ## We accumulate gradients to allow bigger batch size if under limited GPU memory
        for inputs in inputs_packed[1]:
            mini_batch_size = inputs[0].shape[0]
            loss_penalty, train_batch_loss = get_loss_new(decoder, inputs, eta_J, eta_h)
            loss_penalty = loss_penalty * mini_batch_size/effective_batch_size
            train_batch_loss = train_batch_loss * mini_batch_size/effective_batch_size
            scaler.scale(loss_penalty).backward()
            loss_penalty_full += loss_penalty.detach()
            train_loss_full += train_batch_loss
    
    
    scaler.step(optimizer)
    scaler.update()

    return loss_penalty_full, train_loss_full

###################################################################################################################
##################################################### LOADING DATA ################################################

max_msas = 10
msa_dir = "./../../split2/"
encoding_dir ="./../../structure_encodings/"

train_dataset = EncodedProteinDataset_new(os.path.join(msa_dir, 'train'), encoding_dir, noise=0.02, max_msas=max_msas)          ## Default value of noise used
#sequence_test_dataset = EncodedProteinDataset_new(os.path.join(msa_dir, 'test/sequence'), encoding_dir, noise=0.0, max_msas=max_msas)
sequence_test_dataset = train_dataset
structure_test_dataset = EncodedProteinDataset_new(os.path.join(msa_dir, 'test/structure'), encoding_dir, noise=0.0, max_msas=max_msas)
superfamily_test_dataset = EncodedProteinDataset_new(os.path.join(msa_dir, 'test/superfamily'), encoding_dir, noise=0.0, max_msas=max_msas)

print(f"I have loaded the train and test datasets: train:{len(train_dataset)}, seq:{len(sequence_test_dataset)}, struc:{len(structure_test_dataset)}, super:{len(superfamily_test_dataset)}")

batch_structure_size_train = 4 
batch_structure_size= 4    ## This is the batch size for the testing losses calculation, given that we do not have to calculate gradients this can be set to higher values
perc_subset_test = 1.0     ## This can be set to lower values if one wants a faster training
batch_msa_size = 128       ### Size of subsample of MSA on which the model is trained for every structure in the batch
q = 21 

collate_fn = partial(collate_fn_old, q=q, batch_size=batch_structure_size, batch_msa_size=batch_msa_size)
train_loader = DataLoader(train_dataset, batch_size=batch_structure_size_train, collate_fn=collate_fn, shuffle=True,
num_workers=4, pin_memory=False)


sequence_test_loader = DataLoader(sequence_test_dataset, batch_size=batch_structure_size, collate_fn=collate_fn, shuffle=False, 
num_workers=4, pin_memory=False, sampler=RandomSampler(sequence_test_dataset, replacement=True, num_samples=int(0.1*perc_subset_test*len(sequence_test_dataset))))

structure_test_loader = DataLoader(structure_test_dataset, batch_size=batch_structure_size, collate_fn=collate_fn, shuffle=False, 
num_workers=4, pin_memory=False, sampler=RandomSampler(structure_test_dataset, replacement=True, num_samples=int(perc_subset_test*len(structure_test_dataset))))

superfamily_test_loader = DataLoader(superfamily_test_dataset, batch_size=batch_structure_size, collate_fn=collate_fn, shuffle=False, 
num_workers=4, pin_memory=False, sampler=RandomSampler(superfamily_test_dataset, replacement=True, num_samples=int(perc_subset_test*len(superfamily_test_dataset))))


decoder = None
embedding = None
torch.cuda.empty_cache()

seed = 24877
torch.random.manual_seed(seed)
np.random.seed(seed)



update_steps = 525000    ## Set to match desired target of epochs for training trough n_epochs = update_steps//(len(train_dataset)//batch_structure_size_train)
test_steps = 70000
bk_iter = False         ## Set to true if one wants to save intermediate models                                          


input_encoding_dim = 512
param_embed_dim = 512
n_param_heads = 48  ## Rank approximation for the couplings matrix. 
d_model = 512 
n_heads = 8 
n_layers = 6
device = 0        
dropout = 0.1

decoder = PottsDecoder(q, n_layers, d_model, input_encoding_dim, param_embed_dim, n_heads, n_param_heads, dropout=dropout)
decoder.to(device)
embedding = get_embedding(q)
embedding.to(device)

summary_writer = SummaryWriter(log_dir='./runs')
layout = {
    "metrics": {
        "loss": ["Multiline", ["loss/train", "loss/sequence", "loss/structure", "loss/superfamily"]],}
}
summary_writer.add_custom_scalars(layout)


import warnings
warnings.filterwarnings("ignore")
lr = 3.5e-4
optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr)
eta_J = 2.15e-6
eta_h = 6.51e-4


hyperparams = {'lr':lr, 'eta_J':eta_J, 'eta_h':eta_h, 'batch_size':batch_structure_size, 'batch_msa_size':batch_msa_size, 'n_param_heads':n_param_heads, 'n_layers':n_layers, 
                'dropout':dropout, 'param_embed_dim':param_embed_dim, 'n_heads': n_heads}

start = time.time()
scaler = torch.cuda.amp.GradScaler()

with tqdm(total = update_steps) as pbar: 
    train_loss = 0
    update_step = 0
    max_gpu=0
    train_batch_losses = []
    epoch = 0.0
    while update_step < update_steps:
        for inputs_packed in train_loader:
            decoder.train()

            loss_penalty, train_batch_loss = train(decoder, inputs_packed, eta_J, eta_h, optimizer, scaler)
            loss_penalty.detach()
            update_step += 1                                ## Increase update step (the update steps will count also different batches within the same epoch)
            epoch = update_step / len(train_loader)
            
            train_batch_losses.append(train_batch_loss) ## Here we append the lossess in the different batches within the same epoch
            
            ## We want to keep track of the test loss not at every batch, too costrly otherwise. Set to desired value
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

                train_batch_losses = []

        
            pbar.set_description(f'update_step: {update_step}, epoch: {epoch:.2f}  train: {train_loss:.2f}, sequence: {sequence_test_loss:.2f}, structure: {structure_test_loss:.2f}, superfamily: {superfamily_test_loss:.2f}')
            pbar.update(1)
            
print(f"It took {time.time()-start} seconds")
save_metrics = {'loss/train': train_loss, 'loss/sequence': sequence_test_loss, 
'loss/structure': structure_test_loss, 'loss/superfamily': superfamily_test_loss}
summary_writer.add_hparams(hyperparams, save_metrics)
summary_writer.close()


bk_dir= './../bk_models/'
fname_par = 'model_14_03_202_potts_epoch_' + str(epoch) + '.pt'

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
d['batch_size'] = batch_structure_size_train
d['seed'] = seed
d['eta_h'] = eta_h
d['eta_J'] = eta_J
d['noise'] = 0.02
d['args_run'] = args_run
d['model_state_dict'] = decoder.state_dict()
d['optimizer_state_dict'] = optimizer.state_dict()

torch.save(d, os.path.join(bk_dir, fname_par))