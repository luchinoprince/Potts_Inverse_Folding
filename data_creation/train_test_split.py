import argparse
import numpy as np
import os
import shutil
import Bio.SeqIO as SeqIO
from tqdm import tqdm


if __name__ == '__main__':

    

    parser = argparse.ArgumentParser()

    parser.add_argument('--msa_dir', default='msas')
    parser.add_argument('--cath_domain_list', default='cath-domain-list.txt')
    parser.add_argument('--cath_fasta', default='cath-dataset-nonredundant-S40.fa')
    parser.add_argument('--output_dir', default='split')
    parser.add_argument('--superfamily_test_fraction', type=float, default=0.1)
    parser.add_argument('--sequence_test_fraction', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)


    args = parser.parse_args()

    np.random.seed(args.seed)

    structure_test_dir = os.path.join(args.output_dir, 'test/structure')
    superfamily_test_dir = os.path.join(args.output_dir, 'test/superfamily')
    sequence_test_dir = os.path.join(args.output_dir, 'test/sequence')
    train_dir = os.path.join(args.output_dir, 'train')

    out_dirs = [structure_test_dir, superfamily_test_dir, sequence_test_dir, train_dir]

    for dir in out_dirs:
        if not os.path.exists(dir):
            print('Creating directory: {}'.format(dir))
            os.makedirs(dir)
        else:
            raise ValueError('Output directory {} already exists'.format(dir))
    
    summary_fid  = open(os.path.join(args.output_dir, 'summary.txt'), 'w')
    
    summary_fid.write("# domain\tsuperfamily\ttest_type\n")

    # parse cath fasta - this is used for filtering multi-segment domains
    seq_records = list(SeqIO.parse(args.cath_fasta, 'fasta'))

    # read msa file names and parse as domain2msa dictionary
    domain2msa = {}
    msa_files = list(filter(lambda s: s.endswith(".a3m"), os.listdir(args.msa_dir)))
    for msa_file in msa_files:
        domain = msa_file.split('.')[0]

        cath_record = list(filter(lambda s: domain in s.id, seq_records))
        if len(cath_record) != 1:
            raise ValueError('Could not find unique domain {} in cath fasta file'.format(domain))
        cath_record = cath_record[0]

        # filter multi-segment domains
        if len(cath_record.id.split('-')) > 2:
            continue

        domain2msa[domain] = msa_file
    

    # parse CATH domain list
    domain2superfamily = {}
    with open(args.cath_domain_list, 'r') as f:

        for line in f:

            if line.startswith('#'):
                continue

            data = line.split()
            domain = data[0]
            # filter out domains that are made of several segments
            if len(domain.split('-')) > 2:
                continue
            superfamily = '.'.join(data[1:6])
            domain2superfamily[domain] = superfamily

    # restrict to domains appearing in the non-redundant dataset
    domain2superfamily = {k: v for k, v in domain2superfamily.items() if k in domain2msa}

    # invert mapping
    superfamily2domains = {}
    for domain, superfamily in domain2superfamily.items():
        if superfamily not in superfamily2domains:
            superfamily2domains[superfamily] = []
        superfamily2domains[superfamily].append(domain)

    superfamilies = sorted(list(set(domain2superfamily.values())))
    nsuper = len(superfamilies)
    ndomains = len(domain2superfamily)

    print("Number of superfamilies: {}".format(nsuper))
    print("Number of domains: {}".format(ndomains))


    # superfamily split

    superfamily_n_test = int(args.superfamily_test_fraction * nsuper)
    superfamily_n_train = nsuper - superfamily_n_test
    assert superfamily_n_test > 0

    superfamily_perm = np.random.permutation(nsuper)
    superfamily_train_idx = superfamily_perm[:superfamily_n_train]
    superfamily_test_idx = superfamily_perm[superfamily_n_train:]

    superfamily_train = [superfamilies[i] for i in superfamily_train_idx]
    superfamily_test = [superfamilies[i] for i in superfamily_test_idx]

    print("superfamily split...")
    for domain in tqdm(domain2msa):
        if domain2superfamily[domain] in superfamily_test:
            shutil.copy(os.path.join(args.msa_dir, domain2msa[domain]), os.path.join(superfamily_test_dir, domain2msa[domain]))
            summary_fid.write("{}\t{}\t{}\n".format(domain, domain2superfamily[domain], 'superfamily'))

    remaining_domains = [domain for domain in domain2msa if domain2superfamily[domain] in superfamily_train]

    # structure split (from superfamilies with at least 2 domains)
    print("structure split...")
    for superfamily in tqdm(superfamily_train):
        if len(superfamily2domains[superfamily]) > 1:
            domains_superfamily = superfamily2domains[superfamily]
            test_domain = np.random.choice(domains_superfamily)

            shutil.copy(os.path.join(args.msa_dir, domain2msa[test_domain]), os.path.join(structure_test_dir, domain2msa[test_domain]))
            remaining_domains.remove(test_domain)
            summary_fid.write("{}\t{}\t{}\n".format(test_domain, superfamily, 'structure'))
    
    # sequence split
    print("sequence split...")
    for domain in tqdm(remaining_domains):
        msa_path = os.path.join(args.msa_dir, domain2msa[domain])

        seq_records = list(SeqIO.parse(msa_path, 'fasta'))

        nseq = len(seq_records)

        if nseq == 1: 
            train_path = os.path.join(train_dir, domain2msa[domain].replace('.a3m', '_train.a3m'))
            SeqIO.write(seq_records, train_path, 'fasta')
            summary_fid.write("{}\t{}\t{}\n".format(domain, domain2superfamily[domain], 'singleton'))
            continue


        nseq_test = int(args.sequence_test_fraction * nseq)
        if nseq_test == 0:
            nseq_test = 1
        
        nseq_train = nseq - nseq_test

        perm = np.random.permutation(nseq)

        train_idx = perm[:nseq_train]
        test_idx = perm[nseq_train:]

        train_records = [seq_records[i] for i in train_idx]
        test_records = [seq_records[i] for i in test_idx]

        train_path = os.path.join(train_dir, domain2msa[domain].replace('.a3m', '_train.a3m'))
        test_path = os.path.join(sequence_test_dir, domain2msa[domain].replace('.a3m', '_test.a3m'))

        SeqIO.write(train_records, train_path, 'fasta')
        SeqIO.write(test_records, test_path, 'fasta')

        summary_fid.write("{}\t{}\t{}\n".format(domain, domain2superfamily[domain], 'sequence'))






    summary_fid.close()