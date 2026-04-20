import numpy as np
import os
import sys
import torch
import pandas as pd
from collections import Counter, defaultdict
from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
import pickle

# Add necessary paths
sys.path.insert(1, "./../util/")
sys.path.insert(1, "./../model/")
sys.path.insert(1, "./../../esm/")

# Import custom modules
from encoded_protein_dataset_new import EncodedProteinDataset_aux
from dynamic_loader import dynamic_collate_fn
from test_utils import load_model, get_samples_potts, compute_distances, select_sequences
from esm_utils import get_samples_esm, load_structure, extract_coords_from_structure
import esm.pretrained as pretrained

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
MAX_MSAS = 9999
MSA_DIR = "/home/lucasilva/split2/"
ENCODING_DIR = "./../../structure_encodings/"
PDB_DIR = '/home/lucasilva/dompdb/'
BK_DIR = "/home/lucasilva/bk_models2/"
DEVICE = 0
ALPHABET = 'ACDEFGHIKLMNPQRSTVWY-'

def load_dataset():
    test_dataset = EncodedProteinDataset_aux(os.path.join(MSA_DIR, 'test/superfamily'), ENCODING_DIR, noise=0.0, max_msas=MAX_MSAS)
    
    batch_structure_size = 1
    batch_msa_size = 128
    q = 21

    collate_fn = partial(dynamic_collate_fn, q=q, batch_size=batch_structure_size, batch_msa_size=batch_msa_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_structure_size, collate_fn=collate_fn, shuffle=False, num_workers=1, pin_memory=True)
    
    return test_dataset, test_loader

def load_models():
    model_esm, alphabet_esm = pretrained.esm_if1_gvp4_t16_142M_UR50()
    model_esm.eval()

    fname_par_ardca = 'model_11_07_2023_epoch_94.0_ardca.pt'
    model_path_ardca = os.path.join(BK_DIR, fname_par_ardca)
    decoder_ardca = load_model(model_path_ardca, device=DEVICE)

    fname_par_potts = 'model_25_06_2023_epoch_94.0.pt'
    model_path_potts = os.path.join(BK_DIR, fname_par_potts)
    decoder_potts = load_model(model_path_potts, device=DEVICE)

    return model_esm, decoder_ardca, decoder_potts

def process_samples(test_dataset, test_loader, model_esm, decoder_ardca, decoder_potts):
    res_full_esm = {}
    res_full_potts = {}
    res_full_ardca = {}

    default_index = ALPHABET.index('-')
    aa_index = defaultdict(lambda: default_index, {ALPHABET[i]: i for i in range(len(ALPHABET))})
    aa_index_inv = dict(map(reversed, aa_index.items()))

    percs = [2.0]

    for idx, inputs_packed in tqdm(enumerate(test_loader), total=len(test_loader)):
        for inputs in inputs_packed[1]:
            msas, encodings, padding_mask = [input.to(DEVICE, non_blocking=True) for input in inputs]
            B, _, N = msas.shape
            pdb_name = test_dataset.msas_paths[idx][-14:-7]

            # Sample ARDCA
            with torch.no_grad():
                samples_ardca = decoder_ardca.sample_ardca_full(encodings, padding_mask, device=DEVICE, n_samples=10000)
                samples_ardca = torch.tensor(samples_ardca.to('cpu'), dtype=torch.long)

            distances_ardca = compute_distances(samples_ardca, idx, test_dataset, pdb_name, aa_index, aa_index_inv)
            res_ardca = select_sequences(distances_ardca, samples_ardca, percs)

            # Sample ESM
            pdb_path = os.path.join(PDB_DIR, pdb_name)
            structure = load_structure(pdb_path)
            coords, native_seq = extract_coords_from_structure(structure)
            native_seq_num = torch.tensor([aa_index[char] for char in native_seq], dtype=torch.long)
            
            res_esm = get_samples_esm(model_esm, coords, idx, test_dataset, native_seq_num, pdb_name, percs, nfill=100, device=DEVICE)

            # Sample Potts
            couplings_potts, fields_potts = decoder_potts(encodings, padding_mask)
            samples_potts = get_samples_potts(couplings_potts, fields_potts, aa_index, aa_index_inv, N, len(ALPHABET))
            distances_potts = compute_distances(samples_potts, idx, test_dataset, pdb_name, aa_index, aa_index_inv)
            
            res_potts = select_sequences(distances_potts, samples_potts, percs)

            # Store results
            res_full_potts[pdb_name] = res_potts
            res_full_ardca[pdb_name] = res_ardca
            res_full_esm[pdb_name] = res_esm

    return res_full_esm, res_full_potts, res_full_ardca

def save_results(res_full_esm, res_full_potts, res_full_ardca):
    with open("samples_esm_superfamily", mode="wb") as f:
        pickle.dump(res_full_esm, f)
    
    with open("samples_potts_superfamily", mode="wb") as f:
        pickle.dump(res_full_potts, f)
    
    with open("samples_ardca_superfamily", mode="wb") as f:
        pickle.dump(res_full_ardca, f)

def main():
    test_dataset, test_loader = load_dataset()
    model_esm, decoder_ardca, decoder_potts = load_models()
    
    res_full_esm, res_full_potts, res_full_ardca = process_samples(test_dataset, test_loader, model_esm, decoder_ardca, decoder_potts)
    
    save_results(res_full_esm, res_full_potts, res_full_ardca)

if __name__ == "__main__":
    main()