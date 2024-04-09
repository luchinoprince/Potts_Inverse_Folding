import numpy as np
import os, sys
#

sys.path.insert(1, "./../util")
sys.path.insert(1, "./../model")
#from encoded_protein_dataset_new import get_embedding, EncodedProteinDataset_new, EncodedProteinDataset_aux, collate_fn_new#, dynamic_collate_fn
from potts_decoder import PottsDecoder

from typing import Sequence, Tuple, List
#import scipy
#import pandas as pd
import torch, torchvision
import subprocess


sys.path.insert(1, "./../esm/")
from collections import defaultdict
from torch.nn.functional import one_hot


from esm_utils import load_structure, extract_coords_from_structure, get_atom_coords_residuewise


import pyhmmer

def get_str(seq_num, aa_index_inv):
    """ This function translates a protein expressed in its primary structure from numeric to the standard character format. 
        Args:
        seq_num: vector, tensor or array given the numeric sequence of the protein
        aa_index_inv: dictionary mapping numbers to characters 
    """
    seq_str = ""
    for num in seq_num:
        seq_str += aa_index_inv[num.item()]
    return seq_str

def load_model(model_path, device):
    """ This function loads a trained model which has been saved in the format indicated in the training files
        Args:
        model_path: absolute path to model
        device: where you will want the loaded model to be stored(cpu or gpu)
    """
    checkpoint = torch.load(model_path)
    q=21
    args = checkpoint['args_run']
    n_layers = args['n_layers']
    param_embed_dim = d_model = args['param_embed_dim']
    input_encoding_dim = args['input_encoding_dim']
    n_heads=args['n_heads']
    n_param_heads=args['n_param_heads']
    dropout=args['dropout']

    decoder = PottsDecoder(q, n_layers, d_model, input_encoding_dim, param_embed_dim, n_heads, n_param_heads, dropout=dropout);
    decoder.to(device);

    decoder.load_state_dict(checkpoint['model_state_dict']);
    decoder.eval();   ##to generate data we need just the forward pass of the model!
    return decoder

def compute_distances(samples, idx_bk, test_dataset, pdb_name, aa_index, aa_index_inv):
    """ This function computes the distances of a set of samples(ether sythetic or true) from the native sequence of the structure. 
        Args:
        samples: matrix/tensor of aligned samples of dimension [N_sam x L], where L is the length of the native sequence(after alignment), and N_sam is the number of samples
        idx_bk: index in the training dataset of the msa corresponding to the native sequence
        test_datast: test dataset under consideration
        pdb_name: name of the structure to allow loading of the respective pdb file
        aa_index: dictionary mapping character to integers 
        aa_index_inv: dictionary mapping integers to characters
    """
    ################ THIS FUNCTION WORKS FOR ANY SET OF SAMPLES GENERATED
    pdb_dir = './../../dompdb'
    #### Before computing distances I re-align
    alphabet_hmm = pyhmmer.easel.Alphabet.amino()
    msa_train = test_dataset[idx_bk][0]
    M,N = msa_train.shape
    samples_dig = []

    ### Create digitized data
    for sam in range(msa_train.shape[0]):
        sample_num = msa_train[sam, :]
        sample = get_str(sample_num, aa_index_inv)
        name = f"sample{sam+1}".encode('ASCII')
        sample_dig = pyhmmer.easel.TextSequence(name = name, sequence=sample).digitize(alphabet_hmm)
        samples_dig.append(sample_dig)
        
    digMSA = pyhmmer.easel.DigitalMSA(alphabet=alphabet_hmm, name=b"train", sequences = samples_dig)
    builder = pyhmmer.plan7.Builder(alphabet_hmm, symfrac=0.0)
    background = pyhmmer.plan7.Background(alphabet_hmm)
    hmm, _, _ = builder.build_msa(digMSA, background)
    
    #pdb_name = structure_test_dataset.msas_paths[idx][-14:-7] 
    pdb_path = os.path.join(pdb_dir, pdb_name)

    structure =  load_structure(pdb_path)
    coords, native_seq = extract_coords_from_structure(structure)

    native_seq_num = torch.zeros(len(native_seq), dtype=torch.long)
    idx_char=0
    for char in native_seq:
        native_seq_num[idx_char] = aa_index[char]
        idx_char+=1

    samples_hmm = []
    samples_num = []
    it = 1

    name = f"sample{0}".encode('ASCII')
    sample_dig = pyhmmer.easel.TextSequence(name = name, sequence=native_seq).digitize(alphabet_hmm)
    samples_hmm.append(sample_dig)

    native_seq_aligned = pyhmmer.hmmer.hmmalign(hmm, samples_hmm, trim=True).alignment[0]

    native_seq_num_aligned = []
    for char in native_seq_aligned:
        native_seq_num_aligned.append(aa_index[char])

    native_seq_num_aligned = torch.tensor(native_seq_num_aligned) 
    distances = []
    for j in range(samples.shape[0]):
        distances.append(1 - torch.sum(samples[j,:] == native_seq_num).item()/N)
        
    
    return distances

def select_sequences(distances_full, samples, percs):
    """ This function selects a set of sequences that satisfy some distance requirements defined by percs. 
        Args:
        distances_full: array/vector/list of length N_sam with the distances from the native sequence of a set of sythetic/real sequences
        samples: matrix/tensor of aligned samples of dimension [N_sam x L], where L is the length of the native sequence(after alignment), and N_sam is the number of samples
        percs: vector indicating the bins that define how distances are grouped. By default we select 10 sequences per bin.
    """
    it = 0
    res = {}
    for perc in percs:
        if it==0:
            #print(f"distances shape: {distances_full}")
            mask = (np.array(distances_full) < perc)
            candidate_sequences = samples[mask, :]
            selected_sequences = candidate_sequences[0:10, :]
        else:
            mask_right = (np.array(distances_full) < perc)
            mask_left = (np.array(distances_full) >= percs[it-1])
            mask = mask_left * mask_right
            candidate_sequences = samples[mask, :]
            selected_sequences = candidate_sequences[0:10, :]
        it+=1
        res[str(perc)] = selected_sequences
    return res



###############################################################################
###############################################################################
###############################################################################
###############################################################################

def get_samples_potts(couplings, fields, aa_index, aa_index_inv, N, q=21, nsamples=1000, nchains=10):
    """ This function generates MCMC samples from a Potts model specified by couplings and fields. 
        Args:
        couplings: Tensor of dimension [N*q, N*q], where N is the length of the input sequences, with the predicted couplings for the Potts model
        fiedls: Tensor of dimension [N*q] with the predicted fields for the Potts model
        samples: matrix/tensor of aligned samples of dimension [N_sam x L], where L is the length of the native sequence(after alignment), and N_sam is the number of samples
        percs: vector indicating the bins that define how distances are grouped. By default we select 10 sequences per bin.
        aa_index: dictionary mapping character to integers 
        aa_index_inv: dictionary mapping integers to characters
        N: length of the input sequence
        q: size of dictionary, default is 21
        nsamples: number of samples per MCMC chain, default is 1e3
        nchains: number of parallel MCMC chain to run, default is 10
    """
    auxiliary_model_dir = "./../../Auxiliary_Data_bmdca/"
    ###### SAVE COUPLINGS AND FIELDS TO GENERATE SAMPLES
    with open(os.path.join(auxiliary_model_dir, "potts_couplings_fields.txt"), "w") as f:
        ## write J
        for i in range(N):
            for j in range(i+1, N):
                for aa1 in range(q):
                    for aa2 in range(q):
                        J_el = couplings[0, i*q+aa1, j*q+aa2].detach().to('cpu').item()
                        line = "J " + str(i) + " " + str(j) + " "+ str(aa1) + " " + str(aa2) + " " + str(J_el) +"\n"
                        f.write(line)
        
        ## write h
        for i in range(N):
            for aa in range(q):
                h_el = fields[0, i*q+aa1].detach().to('cpu').item()
                line = "h " + str(i) + " " + str(aa) + " " + str(h_el) + "\n"
                f.write(line)
    ###### SAMPLE
    auxiliary_model_dir = "./../../Auxiliary_Data_bmdca/"
    out_dir = './../../Auxiliary_Data_bmdca/Auxiliary_Samples_Potts/'
    out_file = 'samplesexp.txt'
    samples_path = os.path.join(auxiliary_model_dir, "potts_couplings_fields.txt")
    ## I generate a number of samples equal to the MSA, which we know is filtered to have at least 2k samples
    ## The ! creates a terminal command, to pass variable you need to put square brackets
    bash_command = f"bmdca_sample -p {samples_path} -n {nsamples} -r {nchains} -d {out_dir} -o {out_file} -c bmdca.config"
    subprocess.run(bash_command, shell=True, executable="/bin/bash")
    
    file='samplesexp_numerical.txt'
    with open(os.path.join(out_dir,file), mode='r') as f:
        lines=f.readlines()

    ########################### TRANSLATE FROM THEIR DICTIONARY TO OURS ###########################
    char_seq = []
    for i in range(1, len(lines)):
        line = lines[i][0:-1].split(" ") ## I take out the end of file
        line_char = [aa_index_inv[int(idx)] for idx in line]
        char_seq.append(line_char)
        
    ## Now re-translate
    for prot_idx in range(len(char_seq)):
        for aa in range(len(char_seq[prot_idx])):
            char_seq[prot_idx][aa] = aa_index[char_seq[prot_idx][aa]]
            
    msa_t = torch.tensor(char_seq, dtype=torch.long)
    return msa_t

           

