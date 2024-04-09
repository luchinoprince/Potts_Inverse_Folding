########## In this Python file I will streamline the generation mechanism of the data to 

################# IMPORT LIBRARIES #############
import numpy as np
import os, sys, subprocess


sys.path.insert(1, "./../util/")
sys.path.insert(1, "./../model/")
from encoded_protein_dataset_new import get_embedding, EncodedProteinDataset_aux
from pseudolikelihood import get_npll2, get_npll3
import torch, torchvision
from potts_decoder import PottsDecoder
from torch.utils.data import DataLoader, RandomSampler
from functools import partial
from test_utils import load_model, get_samples_potts, compute_distances, select_sequences


from typing import Sequence, Tuple, List
import scipy
import pandas as pd


##Lucas computer
sys.path.insert(1, "./../../esm/")
import esm
#from esm.inverse_folding import util
import esm.pretrained as pretrained
from ioutils import read_fasta, read_encodings
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
from Bio import SeqIO
from dynamic_loader import dynamic_collate_fn, dynamic_cluster
from torch.nn.functional import one_hot

from torch.nn.functional import softmax
from torch.distributions import Categorical

from esm_utils import load_structure, extract_coords_from_structure, get_atom_coords_residuewise
from esm_utils import sample_esm_batch2
from esm_utils import align_esm


################## Function to compute the covariance ####################
def compute_covariance(msa, q):
    """
    Compute covariance matrix of a given MSA having q different amino acids
    """
    M, N = msa.shape

    # One hot encode classes and reshape to create data matrix
    D = torch.flatten(one_hot(msa, num_classes=q), start_dim=1).to(torch.float32)

    # Remove one amino acid
    D = D.view(M, N, q)[:, :, :q-1].flatten(1)

    # Compute bivariate frequencies
    bivariate_freqs = D.T @ D / M
    
    # Compute product of univariate frequencies
    univariate_freqs = torch.diagonal(bivariate_freqs).view(N*(q-1), 1) @ torch.diagonal(bivariate_freqs).view(1, N*(q-1))

    return bivariate_freqs - univariate_freqs


######################## LOADING THE DATA ########################
max_msas = 1
msa_dir = "./../../split2/"
encoding_dir ="./../../structure_encodings/"

####################### CHOOSE TEST DATASET ]]]]]]]]]]]]]]]]]]]]]]]]]
test_dataset = EncodedProteinDataset_aux(os.path.join(msa_dir, 'test/superfamily'), encoding_dir, noise=0.0, max_msas=max_msas)
batch_structure_size_train = 1### I think with empty GPU we can go up to 16 easily
batch_structure_size=1
perc_subset_test = 1.0     ## During the training, for every dataset available we select a random 10% of its samples
batch_msa_size = 128 ### old is 32, original is 16
q = 21 ##isn't always 21
embeddings = get_embedding(q)
device = 2

collate_fn = partial(dynamic_collate_fn, q=q, batch_size=batch_structure_size, batch_msa_size=batch_msa_size)
test_loader = DataLoader(test_dataset, batch_size=batch_structure_size, collate_fn=collate_fn, shuffle=False, 
num_workers=1, pin_memory=True)


############################## FOR ESM ######################################
alphabet='ACDEFGHIKLMNPQRSTVWY-'
default_index = alphabet.index('-')
aa_index = defaultdict(lambda: default_index, {alphabet[i]: i for i in range(len(alphabet))})
aa_index_inv = dict(map(reversed, aa_index.items()))   
model_esm, alphabet_esm = pretrained.esm_if1_gvp4_t16_142M_UR50()
model_esm.eval();



################################ FOR ARDCA ####################################
bk_dir = "./../../bk_models2/"
fname_par_ardca = 'model_11_07_2023_epoch_' + str(94.0) + '_ardca' + '.pt'
model_path_ardca = os.path.join(bk_dir, fname_par_ardca)
decoder_ardca = load_model(model_path_ardca, device=device)

############################## FOR POTTS ########################################
fname_par_potts = 'model_25_06_2023_epoch_' + str(94.0) + '.pt'
model_path_potts = os.path.join(bk_dir, fname_par_potts)
decoder_potts = load_model(model_path_potts, device=device)


############################# ARDCA SAMPLES ################################

Ns = torch.zeros(test_dataset.__len__())
Ms = torch.zeros(test_dataset.__len__())
corr_potts = torch.zeros(test_dataset.__len__())
corr_ardca = torch.zeros(test_dataset.__len__())
decoder_ardca.eval()

idx = 0
import warnings
warnings.filterwarnings("ignore")
with torch.no_grad():
    for inputs_packed in test_loader:
        print(f"I am at index:{idx} out of {test_dataset.__len__()}", end="\r")
        for inputs in inputs_packed[1]:
            msas, encodings, padding_mask  = [input.to(device, non_blocking=True) for input in inputs]
            B, M, N = msas.shape
            couplings, fields = decoder_ardca.forward_ardca(encodings, padding_mask)
            test_msa=torch.load(test_dataset.msas_paths[idx]).to(torch.int)
            M_full = test_msa.shape[0]
            samples_ardca = decoder_ardca.ample_ardca_full_scaled(encodings, padding_mask, device='cpu', n_samples=2000)
            
            samples_ardca=torch.tensor(samples_ardca.to('cpu'), dtype=torch.long)
            test_msa = torch.tensor(test_msa, dtype=torch.long)
            cov_train = compute_covariance(test_msa, q)
            cov_ardca = compute_covariance(samples_ardca, q)
            res = scipy.stats.pearsonr(cov_train.flatten(), cov_ardca.flatten())

            Ns[idx] = N
            Ms[idx] = test_msa.shape[0]
            corr_ardca[idx] = res[0]
            idx+=1

torch.save(corr_ardca, "res_ardca_superfamily.pt")

print("FINISHED ARDCA")
    
###################### POTTS #############

Ns = torch.zeros(test_dataset.__len__())
Ms = torch.zeros(test_dataset.__len__())
corr_potts = torch.zeros(test_dataset.__len__())
corr_ardca = torch.zeros(test_dataset.__len__())
decoder_potts.eval()

alphabet='ACDEFGHIKLMNPQRSTVWY-'
default_index = alphabet.index('-')
aa_index = defaultdict(lambda: default_index, {alphabet[i]: i for i in range(len(alphabet))})
aa_index_inv = dict(map(reversed, aa_index.items()))

idx = 0
import warnings
warnings.filterwarnings("ignore")
with torch.no_grad():
    for inputs_packed in test_loader:
        print(f"I am at index:{idx} out of {test_dataset.__len__()}", end="\r")
        for inputs in inputs_packed[1]:
            msas, encodings, padding_mask  = [input.to(device, non_blocking=True) for input in inputs]
            B, M, N = msas.shape
            couplings, fields = decoder_potts(encodings, padding_mask)
            test_msa=torch.load(test_dataset.msas_paths[idx]).to(torch.int)
            M_full = test_msa.shape[0]

            msa_t = get_samples_potts(couplings, fields, aa_index, aa_index_inv, N, q)
            print(f"msa_t shape is: {msa_t.shape}, test msa shape is: {test_msa.shape}")
            cov_potts = compute_covariance(msa_t, 21)
            test_msa = torch.tensor(test_msa, dtype=torch.long)
            cov_train = compute_covariance(test_msa, q)
            res = scipy.stats.pearsonr(cov_train.flatten(), cov_potts.flatten())


            Ns[idx] = N
            Ms[idx] = test_msa.shape[0]
            corr_potts[idx] = res[0]

            with open("tracker.txt", "w") as f:
                line = "fineshed " + str(idx) + "\n"
                f.write(line)
            idx+=1

    
torch.save(corr_potts, "res_potts_superfamily.pt")



########################## ESM ##########################
import warnings
warnings.filterwarnings("ignore")

Ns = torch.zeros(test_dataset.__len__())
Ms = torch.zeros(test_dataset.__len__())
Ms_sampled = torch.zeros(test_dataset.__len__())
corr_esm = torch.zeros(test_dataset.__len__())
resexp = {}


it = 0
for filename in test_dataset.msas_paths:

    ### LOAD COORDINATES
    pdb_id = test_dataset.msas_paths[it].split('/')[-1][-14:7]
    pdb_dir = './../../dompdb'
    pdb_path = os.path.join(pdb_dir, pdb_id)
    structure =  load_structure(pdb_path)
    coords, native_seq = extract_coords_from_structure(structure)

    ### Get the samples
    samples_esm_str = sample_esm_batch2(model_esm, coords, device=device, n_samples=10000)

    ### ALIGN THE SAMPLES
    ## Here move model to CPU
    msa = torch.tensor(test_dataset[it][0], dtype=torch.long)
    M,N=msa.shape
    samples_esm_aligned = torch.tensor(align_esm(samples_esm_str, msa), dtype=torch.long)
    if len(samples_esm_aligned>0):
        ### Compute covariances and correlations
        cov_esm = compute_covariance(samples_esm_aligned,q)
        cov_true = compute_covariance(msa, q)
        res = scipy.stats.pearsonr(cov_true.flatten(), cov_esm.flatten())
        Ms_sampled[it]=len(samples_esm_aligned)
    else:
        res = [0] ### Later check the zeros
        Ms_sampled[it] = 0

    ## Save results
    Ns[it] = N
    Ms[it] = M
    corr_esm[it] = res[0]

    resexp['Ms'] = Ms
    resexp['Ns'] = Ns
    resexp['Ms_sampled'] = Ms_sampled
    resexp['corr_esm'] = corr_esm
    torch.save(resexp, "res_corr_esm_superfamily.pt")
    with open("tracker_esm.txt", "w") as f:
        line = "fineshed " + str(it) + "\n"
        f.write(line)
    it+=1


#
    

