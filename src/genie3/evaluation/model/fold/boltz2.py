"""
Boltz2 handler for the Genie pipeline.

Wraps the Boltz2 model for protein structure prediction, supporting
template-based and MSA-based folding modes for binder design.
"""

import os
import yaml
import json
import shutil
import logging
import subprocess
import numpy as np
import pandas as pd
from collections import OrderedDict

from genie3.evaluation.model.fold.base import FoldHandler
from genie3.evaluation.utils.parse import parse_fasta, parse_problem_name_from_fasta


class Boltz2Handler(FoldHandler):
    """FoldHandler implementation using the Boltz2 model."""

    NAME = 'boltz2'

    def setup(self):
        """
        Validate the version and folding mode for Boltz2.
        """
        if self.version == 'binder':
            if self.datadir is None:
                logging.error('Missing data directory')
                exit(0)
            if self.mode not in ['msa']:
                logging.error(f'Invalid boltz2 mode for binder design: {self.mode}')
                exit(0)
        else:
            logging.error(f'Invalid version: {self.version}')
            exit(0)

    def fold(self, structures_dir):
        """
        Run Boltz2 to predict structures for all sequences in the input FASTA.

        Reads sequences from structures_dir/input.fasta, constructs per-sample
        YAML configuration files, runs Boltz2 prediction, and organizes outputs.

        Args:
            structures_dir: Directory containing input.fasta and where output
                            structure files will be written
        """

        ######################################
        ###   Configuration Construction   ###
        ######################################

        # Load problem name
        sequences_filepath = os.path.join(structures_dir, 'input.fasta')
        problem_name = parse_problem_name_from_fasta(sequences_filepath)
        with open(os.path.join(self.datadir, 'problems', f'{problem_name}.json')) as file:
            problem_info = json.load(file)
        
        # Load sequences by name
        sample_name_to_seq = parse_fasta(sequences_filepath)

        # Create input directory
        input_dir = os.path.join(structures_dir, 'input')
        os.makedirs(input_dir)
        
        # Process
        for name in sample_name_to_seq:

            # Initialize
            config = {
                "version": 1,
                "sequences": [],
            }

            # Parse chain sequences
            seqs = sample_name_to_seq[name].split(':')

            # Create unique sequences
            unique_seqs = OrderedDict()
            for i, seq in enumerate(seqs):
                if seq not in unique_seqs:
                    unique_seqs[seq] = {
                        'id': [chr(ord('A') + i)],
                        'msa_filepath': (
                            problem_info['target_msa_filepath_by_chain'][i-1]
                            if i > 0 else None
                        )
                    }
                else:
                    unique_seqs[seq]['id'].append(chr(ord('A') + i))

            # Process
            for i, seq in enumerate(seqs):
                chain = {
                    "protein": {
                        "id": list(unique_seqs[seq]['id']),
                        "sequence": seq,
                        "msa": "empty",
                    }
                }
                if i > 0:
                    if self.mode == 'msa':
                        assert unique_seqs[seq]['msa_filepath'] is not None
                        chain['protein']['msa'] = unique_seqs[seq]['msa_filepath']
                config['sequences'].append(chain)
            
            # Save
            with open(os.path.join(input_dir, f'{name}.yaml'), 'w') as file:
                yaml.dump(config, file, indent=4)

        ###################
        ###   Process   ###
        ###################

        device_id = int(self.device.split(':')[-1])
        cmd = ' '.join([
            f'CUDA_VISIBLE_DEVICES={device_id}',
            'boltz predict',
            input_dir,
            f'--out_dir {structures_dir}',
            # '--recycling_steps 10',
            # '--diffusion_samples 25'
        ])
        subprocess.call(cmd, shell=True)

        #######################
        ###   Postprocess   ###
        #######################

        for name in sample_name_to_seq:
            shutil.copytree(
                os.path.join(structures_dir, 'boltz_results_input', 'predictions', name),
                os.path.join(structures_dir, name)
            )
        shutil.rmtree(os.path.join(structures_dir, 'boltz_results_input'))

    def _compile_filepath(self, design_filepath):
        """
        Parse name, domain, and file paths from a Boltz2 design filepath.

        Args:
            design_filepath: Path to the predicted CIF file

        Returns:
            dict: 'name', 'domain', 'design_filepath', 'info_filepath'
        """
        sample_name = design_filepath.split('/')[-2]
        domain_name = sample_name.split('-')[0]
        resample_index = int(sample_name.split('-')[-1].split('_')[-1])

        elts = design_filepath.split('/')
        elts = elts[:-1] + ['confidence_' + elts[-1].replace('.cif', '.json')]
        info_filepath = '/'.join(elts)

        return {
            'name': sample_name,
            'domain': domain_name,
            'resample_id': resample_index,
            'design_filepath': design_filepath,
            'info_filepath': info_filepath
        }

    def _compile_rank(self, design_filepath):
        """
        Extract the rank of this prediction from the Boltz2 filename.

        Args:
            design_filepath: Path to the predicted structure file

        Returns:
            dict: {'rank': int}
        """
        return {
            'rank': int(design_filepath.split('_')[-1].split('.')[0]) + 1,
            'model_id': 0
        }
    
    def _compile_confidence(self, info_filepath):
        """
        Extract pLDDT, pAE, pTM, ipTM, and per-chain pTM from a Boltz2 JSON file.

        Args:
            info_filepath: Path to the JSON scores file

        Returns:
            dict: 'plddt', 'pae', 'ptm', 'iptm', 'per_chain_ptm'
        """
        with open(info_filepath) as file:
            info = json.load(file)
        plddt_filepath = info_filepath.replace('confidence', 'plddt').replace('json', 'npz')
        pae_filepath = info_filepath.replace('confidence', 'pae').replace('json', 'npz')
        return {
            'plddt': np.load(plddt_filepath)['plddt'],
            'pae': np.load(pae_filepath)['pae'],
            'per_chain_ptm': dict([
                (chr(ord('A') + int(k)), info['chains_ptm'][k])
                for k in info['chains_ptm']
            ]),
            'ptm': info['ptm'],
            'iptm': info['iptm']
        }
    
    def _compile_ipsae(self, info_filepath, design_filepath, pae_cutoff=10, dist_cutoff=10):
        """
        Compute ipSAE interface quality metrics using the IPSAE tool.

        Loads the PAE matrix from the Boltz2 NPZ output and runs the IPSAE
        scoring script to compute interface quality metrics.

        Args:
            info_filepath: Path to the Boltz2 confidence JSON file
            design_filepath: Path to the predicted mmCIF structure file
            pae_cutoff: PAE cutoff for interface residue selection (default: 10)
            dist_cutoff: Distance cutoff for interface residue selection (default: 10)

        Returns:
            dict: 'ipsae', 'ipsae_d0dom', 'pdockq2' interface quality metrics
        """
        pae_filepath = info_filepath.replace('confidence', 'pae').replace('json', 'npz')
        cmd = ' '.join([
            'python',
            'packages/IPSAE/ipsae.py',
            pae_filepath,
            design_filepath,
            str(pae_cutoff),
            str(dist_cutoff)
        ])
        subprocess.call(cmd, shell=True)
        output_filepath = design_filepath.replace('.cif', f'_{pae_cutoff}_{dist_cutoff}.txt')
        df = pd.read_csv(
            output_filepath,
            sep=r"\s+",        # one or more spaces/tabs
            engine="python"    # needed for regex sep
        )
        df = df[(df['Chn1'] == 'A') & (df['Type'] == 'asym')]
        assert len(df) == 1
        return {
            'ipsae': df.iloc[0]['ipSAE'],
            'ipsae_d0dom': df.iloc[0]['ipSAE_d0dom'],
            'pdockq2': df.iloc[0]['pDockQ2'],
        }