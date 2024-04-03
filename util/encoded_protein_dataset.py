from torch.utils.data import Dataset
import os
from ioutils import read_fasta, read_encodings
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np


# we define a custom embedding since we want to zero-embed padding tokens
def get_embedding(q):

    embedding = torch.nn.Embedding(q+1, q).requires_grad_(False)
    embedding.weight.data.zero_()

    embedding.weight[:q, :q] = torch.eye(q)

    return embedding


class EncodedProteinDataset(Dataset):

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
        #for numseq_file in filter(lambda file: file.endswith('pt'), os.listdir(msa_folder)):
        ## Now we also have filter the pi_ files
        for numseq_file in filter(lambda file: file.endswith('pt') and (not file.startswith('pi_')), os.listdir(msa_folder)):

            #print(numseq_file)
                #print("Sono qui")
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
            #print(encodings.is_cuda)
            #print(f"Encodings:{encodings}")
            if noise>0:
                encodings = encodings + noise*torch.randn(encodings.shape)
            ####Take out this line after this experiment
            #encodings = torch.rand(encodings.shape)
            if self.encoding_dim is None:
                self.encoding_dim = encodings.shape[1]
            else:
                assert self.encoding_dim == encodings.shape[1], "Inconsistent encoding dimension"

            if max_seqs is not None and msa.shape[0] > max_seqs:
                msa = msa[:max_seqs, :]

            N = msa.shape[1]

            # note that beginning and end of encodings are trimmed by default in read_encodings

            # we're ignoring inconsistent data points here; that is OK for the CATH dataset since these errors were checked by hand
            # all examples seem to be due to mixed up alternate location identifiers, i.e. faulty PDBs.
            
            if N != encodings.shape[0]: 
                #print(encodings.shape, msa.shape)
                #print(msa.shape)
                counter_fail2+=1
                if check>0:
                    check = check-1
                "Inconsistent encoding and sequence length for numerical sequence file: " + numseq_file
                continue
            if N < 512:
            ## Otherwise we have a BUG in pytorch Transformer Encoder Layer, maximum avaiable is 1024, I put 512 for memory reasons.
                self.data.append((msa, encodings))

            #print(f"counter:{counter}, N:{N}, len data:{len(self.data)}")

            if max_msas is not None and len(self.data) >= max_msas:
                break
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch, q, batch_msa_size):
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

def collate_fn2(batch, q, batch_msa_size, device='cpu'):
    """ Collate function for data loader, for the moment we suppose to do things on the CPU. 
    """
    # subsample msas, here batch_msa_size is referred to the number of MSAS the model sees when training the Potts model.
    msas = [tuple[0][torch.randint(0, tuple[0].shape[0], (batch_msa_size, )), :].to(device) for tuple in batch]

    # padding works in the second dimension
    msas = [torch.transpose(msa, 1, 0) for msa in msas]

    encodings = [tuple[1].to(device) for tuple in batch]

    msas = pad_sequence(msas, batch_first=True, padding_value=q)
    encodings = pad_sequence(encodings, batch_first=True, padding_value=0.0)

    # permute msa dimension back
    msas = torch.transpose(msas, 2, 1)

    # the padding mask is the same for all sequences in an msa, so we can just take the first one
    padding_mask = msas[:, 0, :] == q

    return msas, encodings, padding_mask

def dynamic_collate_fn(batch, q, batch_size, batch_msa_size, max_units = 2048):
    """ Dynamic Collate function for data loader, 2048 is 512*4, which was the maximum number of input dimension for batch size equal to 4
    """
    ## Calculate maximum length of the given batch
    #N_max = max([tuple[1].shape[0] for tuple in batch])
    # subsample msas
    #print(batch)
    batch_size = min(len(batch), batch_size)
    msas = [tuple[0][torch.randint(0, tuple[0].shape[0], (batch_msa_size, )), :] for tuple in batch]

    # padding works in the second dimension
    msas = [torch.transpose(msa, 1, 0) for msa in msas]
    encodings = [tuple[1] for tuple in batch]

    #msas_packed, encodings_packed = dynamic_cluster(batch_size, msas, encodings, max_units=max_units)
    inputs_packed = dynamic_cluster(batch_size, q, msas, encodings, max_units=max_units)

    #msas_packed = [pad_sequence(msas, batch_first=True, padding_value=q) for msas in msas_packed]
    #encodings_packed = [pad_sequence(encodings, batch_first=True, padding_value=0.0) for encodings in encodings_packed]

    # permute msa dimension back
    #msas_packed = [torch.transpose(msas, 2, 1) for msas in msas_packed]

    # the padding mask is the same for all sequences in an msa, so we can just take the first one
    #padding_mask_packed = [msas[:, 0, :] == q for msas in msas_packed]

    #return batch_size, msas_packed, encodings_packed, padding_mask_packed
    return batch_size, inputs_packed

def dynamic_cluster(batch_size, q, msas, encodings, max_units):
    ##add sorting, allows for better packing probably (efficient padding). Does not bias since I will use the whole batch at every iter, it is just how fast is it.
    #msas_packed = []
    #encodings_packed = []
    inputs_packed = []
    Ns = np.array([encoding.shape[0] for encoding in encodings])
    order = np.argsort(-Ns)  ## Need to reverse the order
    iterator = 0
    ## Order the encodigns based on batch size, this should also allow to know in advance where to split!!! 
    while iterator < batch_size:
        current_encodings = []
        current_msas = []
        dim = order[iterator]
        mini_batch_size = int(np.floor(max_units/Ns[dim]))
        for _ in range(min(mini_batch_size, batch_size-iterator)):
            current_encodings.append(encodings[order[iterator]])
            current_msas.append(msas[order[iterator]])
            iterator+=1
        
        msas_pad = pad_sequence(current_msas, batch_first=True, padding_value=q)
        encodings_pad = pad_sequence(current_encodings, batch_first=True, padding_value=0.0) 

        msas_pad = torch.transpose(msas_pad, 2, 1)
        padding_mask = msas_pad[:, 0, :] == q

        inputs_packed.append((msas_pad, encodings_pad, padding_mask))
        #encodings_packed.append(current_encodings)
        #msas_packed.append(current_msas)
    return inputs_packed
            



