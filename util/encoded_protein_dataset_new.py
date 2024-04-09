### In this file I wrap up what is present in the analogoues _ipynb_ file

import torch
from torch.utils.data import Dataset
import os
from ioutils import read_fasta, read_encodings
from torch.nn.utils.rnn import pad_sequence
import numpy as np


def get_embedding(q):

    embedding = torch.nn.Embedding(q+1, q).requires_grad_(False)
    embedding.weight.data.zero_()

    embedding.weight[:q, :q] = torch.eye(q)

    return embedding


class EncodedProteinDataset_new(Dataset):

    def __init__(self, msa_folder, encodings_folder, noise=0.0, max_msas=None, max_seqs=None):
        #print("I am here")
        self.msa_folder = msa_folder
        self.encodings_folder = encodings_folder
        self.encodings_paths = []
        self.msas_paths = []
        self.q = None
        self.encoding_dim = None
        self.noise = noise

        # read encoding file names
        encoding_files = {s[:7]: s for s in os.listdir(encodings_folder)}
        counter=0
        for numseq_file in filter(lambda file: file.endswith('pt') and (not file.startswith('pi_')), os.listdir(msa_folder)):
            print(f"Counter is:{counter} length data:{len(self.msas_paths)}", end="\r")
            counter+=1
            id = numseq_file[:7]
            if id not in encoding_files:
                print('No encoding file found for MSA file: ', numseq_file)
                continue

            numseq_path = os.path.join(msa_folder, numseq_file)
            encoding_path = os.path.join(encodings_folder, encoding_files[id])

            if not os.path.isfile(encoding_path):
                ## This does not give problems
                print("{} does not exist, skipping {}".format(encoding_path, numseq_path))
                continue

            msa = torch.load(numseq_path).type(torch.int) 
            encodings = torch.tensor(read_encodings(encoding_path, trim=False))
            if msa.shape[1] != encodings.shape[0]:
                print("{} Mismatch in dimension, skipping {}".format(encoding_path, numseq_path))
                continue
            if msa.shape[1]>=512:
                continue

            self.encodings_paths.append(encoding_path)
            self.msas_paths.append(numseq_path)

            if max_msas is not None and len(self.msas_paths) >= max_msas:
                break
        

    def __len__(self):
        return len(self.msas_paths)

    def __getitem__(self, idx):
        encoding_path = self.encodings_paths[idx]
        msa_path = self.msas_paths[idx]

        msa = torch.load(msa_path).type(torch.int)  ## For later calculation, embedding does not work with uint or with Int8!
        encodings = torch.tensor(read_encodings(encoding_path, trim=False))
        if self.noise > 0:
            encodings = encodings + self.noise*torch.randn(encodings.shape)
        if self.encoding_dim is None:
                self.encoding_dim = encodings.shape[1]
        else:
            assert self.encoding_dim == encodings.shape[1], "Inconsistent encoding dimension"
        
        N = msa.shape[1]
        if N != encodings.shape[0]: 
            "Inconsistent encoding and sequence length for numerical sequence file: " + msa_path#numseq_file
            
        #if N < 512:
        return msa, encodings 


########################## To test single training ######################################
class EncodedProteinDataset_aux(Dataset):

    def __init__(self, msa_folder, encodings_folder, noise=0.0, max_msas=None, max_seqs=None):
        #print("I am here")
        self.msa_folder = msa_folder
        self.encodings_folder = encodings_folder
        self.encodings_paths = []
        self.msas_paths = []
        self.q = None
        self.encoding_dim = None
        self.noise = noise

        # read encoding file names
        encoding_files = {s[:7]: s for s in os.listdir(encodings_folder)}
        counter=0
        for numseq_file in filter(lambda file: file.endswith('pt') and (not file.startswith('pi_')), os.listdir(msa_folder)):
            #print(numseq_file)
                #print("Sono qui")
            print(f"Counter is:{counter} length data:{len(self.msas_paths)}", end="\r")
            counter+=1
            id = numseq_file[:7]
            if id not in encoding_files:
                print('No encoding file found for MSA file: ', numseq_file)
                continue

            numseq_path = os.path.join(msa_folder, numseq_file)
            encoding_path = os.path.join(encodings_folder, encoding_files[id])

            if not os.path.isfile(encoding_path):
                ## This does not give problems
                print("{} does not exist, skipping {}".format(encoding_path, numseq_path))
                continue

            msa = torch.load(numseq_path).type(torch.int) 
            encodings = torch.tensor(read_encodings(encoding_path, trim=False))
            if msa.shape[1] != encodings.shape[0]:
                print("{} Mismatch in dimension, skipping {}".format(encoding_path, numseq_path))
                continue
            ### Otherwise transformer modules have problems
            if (msa.shape[1]> 512 or msa.shape[0]< 2000):
                continue

            self.encodings_paths.append(encoding_path)
            self.msas_paths.append(numseq_path)

            if max_msas is not None and len(self.msas_paths) >= max_msas:
                break
        

    def __len__(self):
        return len(self.msas_paths)

    def __getitem__(self, idx):
        encoding_path = self.encodings_paths[idx]
        msa_path = self.msas_paths[idx]

        msa = torch.load(msa_path).type(torch.int)  ## For later calculation, embedding does not work with uint or with Int8!
        encodings = torch.tensor(read_encodings(encoding_path, trim=False))
        if self.noise > 0:
            encodings = encodings + self.noise*torch.randn(encodings.shape)
        if self.encoding_dim is None:
                self.encoding_dim = encodings.shape[1]
        else:
            assert self.encoding_dim == encodings.shape[1], "Inconsistent encoding dimension"
        
        N = msa.shape[1]
        if N != encodings.shape[0]: 
            "Inconsistent encoding and sequence length for numerical sequence file: " + msa_path#numseq_file
            
        return msa, encodings 



class EncodedProteinDataset_old(Dataset):

    def __init__(self, msa_folder, encodings_folder, noise=0.0, max_msas=None, max_seqs=None):
        #print("I am here")
        self.msa_folder = msa_folder
        self.encodings_folder = encodings_folder
        self.data = []
        self.q = None
        self.encoding_dim = None

        # read encoding file names
        encoding_files = {s[:7]: s for s in os.listdir(encodings_folder)}

        # parse data in folder
        counter=0
        counter_fail1=0
        counter_fail2=0
        check=5
        for numseq_file in filter(lambda file: file.endswith('pt') and (not file.startswith('pi_')), os.listdir(msa_folder)):


            print(f"Counter is:{counter}, Counter fail 1:{counter_fail1}, Counter fail 2:{counter_fail2}, length data:{len(self.data)}", end="\r")
            counter+=1
            id = numseq_file[:7]
            if id not in encoding_files:
                raise ValueError('No encoding file found for MSA file: ' + numseq_file)

            numseq_path = os.path.join(msa_folder, numseq_file)
            encoding_path = os.path.join(encodings_folder, encoding_files[id])

            if not os.path.isfile(encoding_path):
                ## This does not give problems
                print("{} does not exist, skipping {}".format(encoding_path, numseq_path))
                continue

            msa = torch.load(numseq_path).type(torch.int)  ## For later calculation, embedding does not work with uint

            encodings = torch.tensor(read_encodings(encoding_path, trim=False))

            if noise>0:
                encodings = encodings + noise*torch.randn(encodings.shape)
            if self.encoding_dim is None:
                self.encoding_dim = encodings.shape[1]
            else:
                assert self.encoding_dim == encodings.shape[1], "Inconsistent encoding dimension"

            if max_seqs is not None and msa.shape[0] > max_seqs:
                msa = msa[:max_seqs, :]

            N = msa.shape[1]

            # note that beginning and end of encodings are trimmed by default in read_encodings

            
            if N != encodings.shape[0]: 
                counter_fail2+=1
                if check>0:
                    check = check-1
                "Inconsistent encoding and sequence length for numerical sequence file: " + numseq_file
                continue
            if N < 512:
            ## Otherwise we have a BUG in pytorch Transformer Encoder Layer, maximum avaiable is 1024, I put 512 for memory reasons.
                self.data.append((msa, encodings))


            if max_msas is not None and len(self.data) >= max_msas:
                break
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


