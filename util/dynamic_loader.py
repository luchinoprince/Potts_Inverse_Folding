import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence


############################################################################################
###################################### DYNAMIC LOADER ######################################   
############################################################################################
def dynamic_collate_fn(batch, q, batch_size, batch_msa_size, max_units = 12228):
    """ Dynamic Collate function for data loader, 2*8192 is 512*(16*2), which was the maximum number of input dimension for batch size equal to 16
    """
    batch_size = min(len(batch), batch_size)
    msas = [tuple[0][torch.randint(0, tuple[0].shape[0], (batch_msa_size, )), :] for tuple in batch]

    # padding works in the second dimension
    msas = [torch.transpose(msa, 1, 0) for msa in msas]
    encodings = [tuple[1] for tuple in batch]

    inputs_packed = dynamic_cluster(batch_size, q, msas, encodings, max_units=max_units)
    return batch_size, inputs_packed

def dynamic_cluster(batch_size, q, msas, encodings, max_units):
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
    return inputs_packed



############################### OLD LOADER ######################################
def collate_fn_old(batch, q, batch_size, batch_msa_size, max_units=None):
    """ Old Collate function for data loader, no permutation. 
    """
    batch_size = len(batch)
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
    ### I return in this weird format to have consistency with the dynamic loader
    return (batch_size, [(msas, encodings, padding_mask)])

