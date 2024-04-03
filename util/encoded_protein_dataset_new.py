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
        # parse data in folder
        #for numseq_file in filter(lambda file: file.endswith('pt'), os.listdir(msa_folder)):
        ## Now we have also the pi_ files, we have to filter them out also.
        for numseq_file in filter(lambda file: file.endswith('pt') and (not file.startswith('pi_')), os.listdir(msa_folder)):
            #print(numseq_file)
                #print("Sono qui")
            print(f"Counter is:{counter} length data:{len(self.msas_paths)}", end="\r")
            counter+=1
            id = numseq_file[:7]
            if id not in encoding_files:
                print('No encoding file found for MSA file: ', numseq_file)
                #raise ValueError('No encoding file found for MSA file: ' + numseq_file)
                continue

            numseq_path = os.path.join(msa_folder, numseq_file)
            encoding_path = os.path.join(encodings_folder, encoding_files[id])

            if not os.path.isfile(encoding_path):
                ## This does not give problems
                print("{} does not exist, skipping {}".format(encoding_path, numseq_path))
                continue

            msa = torch.load(numseq_path).type(torch.int) 
            encodings = torch.tensor(read_encodings(encoding_path, trim=False))
            #encodings = torch.from_numpy(read_encodings(encoding_path, trim=False)) 
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
        #else:
        #    return ## I return nothing in this case


def collate_fn_new(batch, q, batch_msa_size):
    """ Collate function for data loader
    """
    # subsample msas, here batch_msa_size is referred to the number of MSAS the model sees when training the Potts model.
    msas = [tuple[0][torch.randint(0, tuple[0].shape[0], (batch_msa_size, )), :] for tuple in batch]

    # padding works in the second dimension
    msas = [torch.transpose(msa, 1, 0) for msa in msas]

    encodings = [tuple[1] for tuple in batch]

    msas = pad_sequence(msas, batch_first=True, padding_value=q)
    encodings = pad_sequence(encodings, batch_first=True, padding_value=0.0)

    # permute msa dimension back
    msas = torch.transpose(msas, 2, 1)

    # the padding mask is the same for all sequences in an msa, so we can just take the first one
    padding_mask = msas[:, 0, :] == q

    return msas, encodings, padding_mask



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
        # parse data in folder
        #for numseq_file in filter(lambda file: file.endswith('pt'), os.listdir(msa_folder)):
        ## Now we have also the pi_ files, we have to filter them out also.
        for numseq_file in filter(lambda file: file.endswith('pt') and (not file.startswith('pi_')), os.listdir(msa_folder)):
            #print(numseq_file)
                #print("Sono qui")
            print(f"Counter is:{counter} length data:{len(self.msas_paths)}", end="\r")
            counter+=1
            id = numseq_file[:7]
            if id not in encoding_files:
                print('No encoding file found for MSA file: ', numseq_file)
                #raise ValueError('No encoding file found for MSA file: ' + numseq_file)
                continue

            numseq_path = os.path.join(msa_folder, numseq_file)
            encoding_path = os.path.join(encodings_folder, encoding_files[id])

            if not os.path.isfile(encoding_path):
                ## This does not give problems
                print("{} does not exist, skipping {}".format(encoding_path, numseq_path))
                continue

            msa = torch.load(numseq_path).type(torch.int) 
            encodings = torch.tensor(read_encodings(encoding_path, trim=False))
            #encodings = torch.from_numpy(read_encodings(encoding_path, trim=False)) 
            if msa.shape[1] != encodings.shape[0]:
                print("{} Mismatch in dimension, skipping {}".format(encoding_path, numseq_path))
                continue
            ### Otherwise transformer modules have problems
            #if ((msa.shape[1] <200 or msa.shape[1]>250) or msa.shape[0]<2000):
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
            
        #if N < 512:
        return msa, encodings 
        #else:
        #    return ## I return nothing in this case
        
        
        
class EncodedProteinDataset_aux2(Dataset):

    def __init__(self, msa_folder, encodings_folder, noise=0.0, max_msas=None, max_seqs=None):
        #print("I am here")
        self.msa_folder = msa_folder
        self.encodings_folder = encodings_folder
        self.encodings_paths = []
        self.msas_paths = []
        self.q = None
        self.encoding_dim = None
        self.noise = noise

        mutational_dir = '/media/luchinoprince/b1715ef3-045d-4bdf-b216-c211472fb5a2/Data/InverseFolding/Mutational_Data'
        msas_folder = '/media/luchinoprince/b1715ef3-045d-4bdf-b216-c211472fb5a2/Data/InverseFolding/Mutational_Data/alphafold_results_wildtype'

        protein_original_DMS = 'YAP1_HUMAN_1_b0.5.a2m.wildtype.fasta'
        structure_name = 'YAP1_HUMAN_1_b0.5.a2m_unrelaxed_rank_1_model_5.pdb'

        # read encoding file names
        encoding_files = {s[:7]: s for s in os.listdir(encodings_folder)}
        counter=0
        # parse data in folder
        #for numseq_file in filter(lambda file: file.endswith('pt'), os.listdir(msa_folder)):
        ## Now we have also the pi_ files, we have to filter them out also.
        for numseq_file in filter(lambda file: file.endswith('pt') and (not file.startswith('pi_')), os.listdir(msa_folder)[1:]):
            #print(numseq_file)
                #print("Sono qui")
            print(f"Counter is:{counter} length data:{len(self.msas_paths)}", end="\r")
            counter+=1
            id = numseq_file[:7]
            if id not in encoding_files:
                print('No encoding file found for MSA file: ', numseq_file)
                #raise ValueError('No encoding file found for MSA file: ' + numseq_file)
                continue

            numseq_path = os.path.join(msa_folder, numseq_file)
            encoding_path = os.path.join(encodings_folder, encoding_files[id])

            if not os.path.isfile(encoding_path):
                ## This does not give problems
                print("{} does not exist, skipping {}".format(encoding_path, numseq_path))
                continue

            msa = torch.load(numseq_path).type(torch.int) 
            encodings = torch.tensor(read_encodings(encoding_path, trim=False))
            #encodings = torch.from_numpy(read_encodings(encoding_path, trim=False)) 
            if msa.shape[1] != encodings.shape[0]:
                print("{} Mismatch in dimension, skipping {}".format(encoding_path, numseq_path))
                continue
            if (msa.shape[1]>=70 or msa.shape[0]<1000):
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
        #else:
        #    return ## I return nothing in this case