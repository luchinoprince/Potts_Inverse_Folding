from collections import defaultdict
from Bio import SeqIO
import torch


def read_fasta(fasta_path, alphabet='ACDEFGHIKLMNPQRSTVWY-', default_char='-', mutated_exp=False):
    """ Read single fasta file
    """

    default_index = alphabet.index(default_char)
    aa_index = defaultdict(lambda: default_index, {alphabet[i]: i for i in range(len(alphabet))})

    seq_records = list(SeqIO.parse(fasta_path, 'fasta'))
    if mutated_exp==False:
        N = len([s for s in seq_records[0].seq if s.isupper() or s == '-'])
    else:
        N = len(seq_records[0].seq)
    M = len(seq_records)

    msa = torch.zeros((M, N), dtype=torch.long)
    if mutated_exp:
        for (m, seq_record) in enumerate(seq_records):
            msa[m, :] = torch.Tensor([aa_index[c.upper()] for c in str(seq_record.seq)]) 
    else:
        for (m, seq_record) in enumerate(seq_records):
            msa[m, :] = torch.Tensor([aa_index[c] for c in str(seq_record.seq) if c.isupper() or c == '-'])


    return msa, len(alphabet)


def read_encodings(encoding_path, trim=True):
    """ Read single encoding file
    """
    
    #encoding = torch.from_numpy(torch.load(encoding_path)['encodings'])
    encoding = torch.load(encoding_path)['encodings']
    # the ESM encoder adds tokens to beginning and end of structure
    if trim:
        encoding = encoding[1:-1, :]

    return encoding
