import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from collections import defaultdict
import sys
import os
import pyhmmer
import numpy as np
import biotite.structure
from biotite.structure.io import pdbx, pdb
from biotite.structure.residues import get_residues
from biotite.structure import filter_backbone
from biotite.structure import get_chains
from biotite.sequence import ProteinSequence
from typing import Sequence, Tuple, List
from Bio import SeqIO


#### This is the alphabet that will be used to translate
alphabet='ACDEFGHIKLMNPQRSTVWY-'
default_index = alphabet.index('-')
aa_index = defaultdict(lambda: default_index, {alphabet[i]: i for i in range(len(alphabet))})
aa_index_inv = dict(map(reversed, aa_index.items()))

def get_str(seq_num):
    seq_str = ""
    #seq_num = msa_train[0,::]
    for num in seq_num:
        #print(num)
        seq_str += aa_index_inv[num.item()]
    return seq_str

def clean_insertions(msa_aligned, L):
    """ This function takes out the insertions after the re-alignment"""
    samples_aligned_num = []
    for it in range(len(msa_aligned.alignment)):
        seq_num = []
        sample = msa_aligned.alignment[it]
        for char in sample:
            if char == '-' or char.isupper():
            #if char != '.' or char.isupper():
                seq_num.append(aa_index[char])
                
        #print(len(seq_num))
        if len(seq_num) == L:
            ### Take out problematic alignments, they are usually very few.
            samples_aligned_num.append(seq_num)
    return samples_aligned_num

def load_structure(fpath, chain=None):
    """
    Args:
        fpath: filepath to either pdb or cif file
        chain: the chain id or list of chain ids to load
    Returns:
        biotite.structure.AtomArray
    """
    with open(fpath) as fin:
        pdbf = pdb.PDBFile.read(fin)
    structure = pdb.get_structure(pdbf, model=1)
    bbmask = filter_backbone(structure)
    structure = structure[bbmask]
    all_chains = get_chains(structure)
    if len(all_chains) == 0:
        raise ValueError('No chains found in the input file.')
    if chain is None:
        chain_ids = all_chains
    elif isinstance(chain, list):
        chain_ids = chain
    else:
        chain_ids = [chain] 
    for chain in chain_ids:
        if chain not in all_chains:
            raise ValueError(f'Chain {chain} not found in input file')
    chain_filter = [a.chain_id in chain_ids for a in structure]
    structure = structure[chain_filter]
    return structure

def extract_coords_from_structure(structure: biotite.structure.AtomArray):
    """
    Args:
        structure: An instance of biotite AtomArray
    Returns:
        Tuple (coords, seq)
            - coords is an L x 3 x 3 array for N, CA, C coordinates
            - seq is the extracted sequence
    """
    coords = get_atom_coords_residuewise(["N", "CA", "C"], structure)
    residue_identities = get_residues(structure)[1]
    seq = ''.join([ProteinSequence.convert_letter_3to1(r) for r in residue_identities])
    return coords, seq

def get_atom_coords_residuewise(atoms: List[str], struct: biotite.structure.AtomArray):
    """
    Example for atoms argument: ["N", "CA", "C"]
    """
    def filterfn(s, axis=None):
        filters = np.stack([s.atom_name == name for name in atoms], axis=1)
        sum = filters.sum(0)
        if not np.all(sum <= np.ones(filters.shape[1])):
            raise RuntimeError("structure has multiple atoms with same name")
        index = filters.argmax(0)
        coords = s[index].coord
        coords[sum == 0] = float("nan")
        return coords

    return biotite.structure.apply_residue_wise(struct, struct, filterfn)

def sample_esm_batch(model, coords, n_samples=1000, temperature=1.0, confidence=None, device='cpu'):
    """ Since I do not want to touch the forward functions which are very entangled, I will try the following trick
    to hopefully speed the sampler enough. When I want :n_samples from a single structure, I will copy
    such a structure :n_samples times across the batch dimension, from which he should be able to sample from. 
    This is clearly highly memory inefficient, yet it is more memory efficient than generating one samples at the time
    for :n_samples structures, as there if :n_samples is high I will have a lot of padding across the batch dimension. """
    """
    Samples sequences based on multinomial sampling (no beam search).

    Args:
        coords: L x 3 x 3 list representing one backbone
        partial_seq: Optional, partial sequence with mask tokens if part of
            the sequence is known
        temperature: sampling temperature, use low temperature for higher
            sequence recovery and high temperature for higher diversity
        confidence: optional length L list of confidence scores for coordinates
    """
    L = len(coords)
    coords = torch.tensor(coords).to(device)
    model=model.to(device)
    

    batch_coords = torch.zeros((1,L+2,3,3)).to(device)
    batch_coords[0, 1:L+1, :, :] = coords
    batch_coords[0,0,:,:] = torch.inf
    batch_coords[0,L+1,:,:] = torch.inf

    padding_mask = torch.isnan(batch_coords[:,:,0,0]).to(device)


    #### If you do not have a confidence, which is the standard one padded with zeros at the beginning and end, esm default to ones!
    #### https://github.com/facebookresearch/esm/blob/main/esm/inverse_folding/util.py at line 240

    confidence = torch.ones((1,L+2)).to(device)
    confidence[0,0] = 0
    confidence[0,L+1] = 0

    # Run encoder only once
    encoder_out = model.encoder(batch_coords, padding_mask, confidence)

    padding_batched = torch.zeros((n_samples, L+2), dtype=torch.bool).to(device)
    encoder_out['encoder_out'][0] = encoder_out['encoder_out'][0].expand(L+2, n_samples, 512)
    encoder_out['encoder_padding_mask'][0] = padding_batched
    ## Now I have to batch some things 

    # Start with prepend token
    mask_idx = model.decoder.dictionary.get_idx('<mask>')
    sampled_tokens = torch.full((n_samples, 1+L), mask_idx, dtype=int).to(device)
    sampled_tokens[:, 0] = model.decoder.dictionary.get_idx('<cath>')


    # Save incremental states for faster sampling
    incremental_state = dict()

    with torch.no_grad():
        # Decode one token at a time
        for i in range(1, L+1):
            print(f"I am position {i} out of {L}", end="\r")
            logits, _ = model.decoder(
                sampled_tokens[:, :i], 
                encoder_out,
                incremental_state=incremental_state,
            )
            logits = logits.squeeze(-1)#.transpose(0,1)
            logits /= temperature
            probs = F.softmax(logits, dim=-1)
            sampled_tokens[:, i] = torch.multinomial(probs, 1).squeeze(-1)
            sampled_seq = sampled_tokens[0, 1:]


    samples_str = []
    for idx in range(n_samples):
        sampled_seq = sampled_tokens[idx, 1:]
        sampled_str = ''.join([model.decoder.dictionary.get_tok(a) for a in sampled_seq])
        samples_str.append(sampled_str)
    torch.cuda.empty_cache()
    return samples_str

def sample_esm_batch2(model, coords, n_samples=1000, temperature=1.0, confidence=None, device='cpu'):
    """ Since I do not want to touch the forward functions which are very entangled, I will try the following trick
    to hopefully speed the sampler enough. When I want :n_samples from a single structure, I will copy
    such a structure :n_samples times across the batch dimension, from which he should be able to sample from. 
    This is clearly highly memory inefficient, yet it is more memory efficient than generating one samples at the time
    for :n_samples structures, as there if :n_samples is high I will have a lot of padding across the batch dimension. 
    With respect to the other function here I will make sure I don't cause CUDA to go OOM. 
    """
    """
    Samples sequences based on multinomial sampling (no beam search).

    Args:
        coords: L x 3 x 3 list representing one backbone
        partial_seq: Optional, partial sequence with mask tokens if part of
            the sequence is known
        temperature: sampling temperature, use low temperature for higher
            sequence recovery and high temperature for higher diversity
        confidence: optional length L list of confidence scores for coordinates
    """

    L = len(coords)
    coords = torch.tensor(coords).to(device)
    model=model.to(device)
    

    batch_coords = torch.zeros((1,L+2,3,3)).to(device)
    batch_coords[0, 1:L+1, :, :] = coords
    batch_coords[0,0,:,:] = torch.inf
    batch_coords[0,L+1,:,:] = torch.inf

    padding_mask = torch.isnan(batch_coords[:,:,0,0]).to(device)


    #### If you do not have a confidence, which is the standard one padded with zeros at the beginning and end, esm default to ones!
    #### https://github.com/facebookresearch/esm/blob/main/esm/inverse_folding/util.py at line 240

    confidence = torch.ones((1,L+2)).to(device)
    confidence[0,0] = 0
    confidence[0,L+1] = 0

    # Run encoder only once
    encoder_out = model.encoder(batch_coords, padding_mask, confidence)
    ### I do 500 at the time if the protein is long, otherwise I go OOM
    if L>200:
        samples_batch = 400
        steps = 5
    else:
        samples_batch=1000
        steps=2
    
    padding_batched = torch.zeros((samples_batch, L+2), dtype=torch.bool).to(device)
    encoder_out['encoder_out'][0] = encoder_out['encoder_out'][0].expand(L+2, samples_batch, 512)
    encoder_out['encoder_padding_mask'][0] = padding_batched
        ## Now I have to batch some things 

    # Start with prepend token
    mask_idx = model.decoder.dictionary.get_idx('<mask>')
    sampled_tokens = torch.full((samples_batch, 1+L), mask_idx, dtype=int).to(device)
    sampled_tokens[:, 0] = model.decoder.dictionary.get_idx('<cath>')
    samples_str = []

    for j in range(steps):
        # Save incremental states for faster sampling
        incremental_state = dict()
        with torch.no_grad():
            # Decode one token at a time
            for i in range(1, L+1):
                print(f"I am position {i} out of {L}, batch {j+1} out of {steps}", end="\r")
                logits, _ = model.decoder(
                    sampled_tokens[:, :i], 
                    encoder_out,
                    incremental_state=incremental_state,
                )
                logits = logits.squeeze(-1)#.transpose(0,1)
                logits /= temperature
                probs = F.softmax(logits, dim=-1)
                #if sampled_tokens[0, i] == mask_idx:
                sampled_tokens[:, i] = torch.multinomial(probs, 1).squeeze(-1)
                sampled_seq = sampled_tokens[0, 1:]


        for idx in range(samples_batch):
            sampled_seq = sampled_tokens[idx, 1:]
            sampled_str = ''.join([model.decoder.dictionary.get_tok(a) for a in sampled_seq])
            if len(sampled_str) == L:
                ## In case we sample off sequences
                samples_str.append(sampled_str)
            else:
                continue
    ### Let us clear up the cache of the GPU
    torch.cuda.empty_cache()
    return samples_str


def align_esm(samples_esm_str, msa):
    """ This function takes a esm sample from a given structure and re-aligns its
    generated sequences using the MSA of the strucutre.

    Args:
    samples_esm_str: list of samples in character form coming from the esm samples which need re-alignment
    msa: msa corresponding to the native structure to which we have to re-align
    
    """
    M, L = msa.size()
    alphabet_hmm = pyhmmer.easel.Alphabet.amino()
    ### DIGITIZE THE MSA SO THAT IT CAN BE USED PY PYHMMER
    samples_dig = []
    for sam in range(msa.shape[0]):
        sample_num = msa[sam, :]
        sample = get_str(sample_num)
        name = f"sample{sam+1}".encode('ASCII')
        sample_dig = pyhmmer.easel.TextSequence(name = name, sequence=sample).digitize(alphabet_hmm)
        samples_dig.append(sample_dig)
    
    digMSA = pyhmmer.easel.DigitalMSA(alphabet=alphabet_hmm, name=b"train", sequences = samples_dig)    

    ###### BUILD THE Hidden Markov Model
    builder = pyhmmer.plan7.Builder(alphabet_hmm, symfrac=0.0)
    background = pyhmmer.plan7.Background(alphabet_hmm)
    hmm, _, _ = builder.build_msa(digMSA, background)

    #### Now Digitize the esm samples and re-align them. 
    samples_esm_dig = []
    it = 0
    for sample in samples_esm_str:
        name = f"sample{it}".encode('ASCII')
        sample_dig = pyhmmer.easel.TextSequence(name = name, sequence=sample).digitize(alphabet_hmm)
        samples_esm_dig.append(sample_dig)
        it+=1
    
    msa_aligned = pyhmmer.hmmer.hmmalign(hmm, samples_esm_dig, trim=True)
    
    ### Now let us take out the padding insertions, which are marked by ".".
    ### Such padding are NOT aligned, hence are useless for our purposes. 
    msa_aligned = clean_insertions(msa_aligned, L)
    return msa_aligned

def get_samples_esm(model, coords, idx_bk, test_dataset, native_seq_num, pdb_id, percs, nfill=25, device=0):
    """This function allows to generate samples from the esm1f model at the desired hamming distances from the native sequence. 
    It does so by starting to generate samples at temperature 1, and then sequentially increasing the temperature of the model by 0.1 to 
    fill all the necessary bins. It reaches a maximum temperature of 4, which can be easily changed.

    Args:
    model: the esm1f model
    coords: coords: N x 3 x 3 list representing one backbone, where N is the length of the sequence
    idx_bk: index in the training dataset of the msa corresponding to the native sequence
    test_datast: test dataset under consideration
    native_seq_num: array with the native sequence in numerical format
    pdb_name: name of the structure to allow loading of the respective pdb file
    percs: vector indicating the bins that define how distances are grouped.
    nfill: number of sequences per bin in 'percs', default is 25.
    device: device where to perform computations, default is 0(i.e GPU). CPU will be significantly slower for this function.

    """

    model.eval();
    model.to(device)

    res = {}

    samples_esm_full = []


    ##############################    GET SAMPLES    ##############################
    ###############################################################################
    filled = False
    temperature = 1.0
    while not filled:
        print(f"I am sampling at temperature:{temperature}")
        filled = True
        ##############################    GET POTENTIAL SAMPLES    ###########################
        ######################################################################################
        samples_esm_str = sample_esm_batch2(model, coords, temperature=temperature, device=device)
        
        ### ALIGN THE SAMPLES
        msa = torch.tensor(test_dataset[idx_bk][0], dtype=torch.long)
        M,N=msa.shape
        if len(native_seq_num)!=N:
            raise Exception("ERROR")
        samples_esm_aligned = torch.tensor(align_esm(samples_esm_str, msa), dtype=torch.long)
        samples_esm_full.append((pdb_id, samples_esm_aligned))
        M_esm = samples_esm_aligned.shape[0]
        
        
        ##############################    COMPUTE DISTANCES    ##############################
        #####################################################################################
        distances = []
        distances_full = []

        for j in range(M_esm):
            distances.append(1 - torch.sum(samples_esm_aligned[j,:] == native_seq_num).item()/N)
            
        distances_full.append((pdb_id, distances))
        
        
        ##############################    FILL RESULT    ###################################
        ####################################################################################
        it = 0
        for perc in percs:
            if str(perc) in res.keys():
                print("I am adding samples\n")
                if res[str(perc)].shape[0] < nfill: ### I have not filled this level yet
                    missing = nfill - res[str(perc)].shape[0] 
                    if it==0:
                        mask = np.array(distances_full[0][1]) < perc
                        candidate_sequences = samples_esm_aligned[mask, :]
                        adding = min(candidate_sequences.shape[0], missing)
                        selected_sequences = candidate_sequences[0:adding, :]
                        res[str(perc)] = np.concatenate((res[str(perc)], selected_sequences), axis=0)

                        if adding != missing:
                            filled = False
                    else:
                        mask_right = np.array(distances_full[0][1]) < perc
                        mask_left = np.array(distances_full[0][1]) >= percs[it-1]
                        mask = mask_left * mask_right
                        candidate_sequences = samples_esm_aligned[mask, :]
                        adding = min(candidate_sequences.shape[0], missing)
                        selected_sequences = candidate_sequences[0:adding, :]
                        res[str(perc)] = np.concatenate((res[str(perc)], selected_sequences), axis=0)
                        
                        if adding != missing:
                            filled = False
                            
            else: ### This means I am at my first loop
                print("I am sampling for the first time\n")
                missing = nfill 
                if it==0:
                    mask = np.array(distances_full[0][1]) < perc
                    candidate_sequences = samples_esm_aligned[mask, :]
                    adding = min(candidate_sequences.shape[0], missing)
                    selected_sequences = candidate_sequences[0:adding, :]
                    res[str(perc)] = selected_sequences
                    if adding != missing:
                        filled = False
                else:
                    mask_right = np.array(distances_full[0][1]) < perc
                    mask_left = np.array(distances_full[0][1]) >= percs[it-1]
                    mask = mask_left * mask_right
                    candidate_sequences = samples_esm_aligned[mask, :]
                    adding = min(candidate_sequences.shape[0], missing)
                    selected_sequences = candidate_sequences[0:adding, :]
                    res[str(perc)] = selected_sequences
                    
                    if adding != missing:
                        filled = False
            it+=1
        temperature += 0.1
        if temperature > 4.0:
            #### I set a maximum temperature
            filled=True
        
    return res
