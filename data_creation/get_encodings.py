import argparse
import numpy as np
import time
from pathlib import Path
import os
import numpy as np
from collections import defaultdict

import esm
import esm.inverse_folding

import torch

from tqdm import tqdm

from tempfile import NamedTemporaryFile


def main():
    parser = argparse.ArgumentParser(
            description='Encode structures'
    )
    parser.add_argument(
            '--pdb_dir',
            default='dompdb'
    )
    parser.add_argument(
            '--out_dir', type=str,
            help='output directory for encoded structures',
            default='encoded_structures',
    )
    parser.add_argument(
            '--device', type=str, default='cuda'
    )

    args = parser.parse_args()

    device = args.device

    np.random.seed(0)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    with torch.no_grad():

        model, _ = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        model = model.eval()
        model.cuda()


        for pdb_file in tqdm(np.random.permutation(os.listdir(args.pdb_dir))):

            outpath = os.path.join(args.out_dir, pdb_file + '.encodings.pt')

            if os.path.exists(outpath) and os.path.getsize(outpath) > 0:
                print(f'Output file {outpath} already exists. Skipping.')
                continue

            print(f'Encoding {pdb_file}...')

            chain = pdb_file[4]
            # chains are parse as uppercase; also, a '0' can indicate no chain or a '0' chain
            if chain != '0':
                chain = chain.upper()
            else:
                with open(os.path.join(args.pdb_dir, pdb_file), 'r') as f:
                    chain = f.readline()[21]
                    if chain == ' ':
                        chain = ''

            # esm.inverse looks at file extensions to determine the format
            with NamedTemporaryFile(suffix='.pdb') as f:
                os.system(f'cat {args.pdb_dir}/{pdb_file} > {f.name}')

                # the try catch here is to handle faulty PDB files which have atoms without and with alternate location indicators
                # this is forbidden by the PDB format, but e.g. 4xhy has it
                try:
                    coords, seq = esm.inverse_folding.util.load_coords(f.name, chain)
                except RuntimeError as e:
                    if str(e) == 'structure has multiple atoms with same name':

                        # the workaround is to add alternative location indicators to all atoms
                        print("WARNING: WORKING AROUND PDB FORMAT VIOLATION")

                        # empty temporary file
                        os.system(f'truncate -s 0 {f.name}')

                        # emulate maximum occupancy altloc choice
                        with open(os.path.join(args.pdb_dir, pdb_file), 'r') as fid:
                            lines = defaultdict(list)
                            for line in fid:
                                if line.startswith('ATOM'):
                                    res = line[22:26]
                                    lines[res].append(line)
                            
                            for res in lines:
                                lines_res = lines[res]
                                altlocs = list(map(lambda line: line[16], lines_res))
                                altloc_most = max(set(altlocs), key=altlocs.count)
                                lines_res = list(filter(lambda line: line[16] == altloc_most, lines_res))
                                for line in lines_res:
                                    os.system(f'echo -n "{line}" >> {f.name}')
                            
                        coords, seq = esm.inverse_folding.util.load_coords(f.name, chain)
                        pass
            
            # Convert to batch format
            batch_converter = esm.inverse_folding.util.CoordBatchConverter(model.decoder.dictionary)
            batch_coords, confidence, _, _, padding_mask = (
                batch_converter([(coords, None, None)], device=device)
            )
            
            # Run encoder only once
            start = time.time()
            encoder_out = model.encoder(batch_coords, padding_mask, confidence, return_all_hiddens=True)
            breakpoint()
            print("encoding took {} seconds".format(time.time() - start))

            d = {'encodings': encoder_out['encoder_out'][0].squeeze().cpu().numpy(), 'seq': seq}

            torch.save(d, outpath)


if __name__ == '__main__':
    main()