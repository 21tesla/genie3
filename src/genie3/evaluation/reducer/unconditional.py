"""
Unconditional reducer for the Genie pipeline.

Compiles and filters results for unconditional protein generation,
computing self-consistency TM-score and RMSD metrics.
"""

import os
import glob
import shutil
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

from genie3.evaluation.reducer.base import Reducer
from genie3.evaluation.utils.metric import compute_allatom_rmsd, compute_multimer_ca_rmsd, compute_tmscore
from genie3.evaluation.utils.parse import parse_fasta
from genie3.evaluation.utils.secondary import assign_secondary_structure


class UnconditionalReducer(Reducer):
    """Reducer for unconditional protein generation evaluation."""

    def _validate(self):
        """
        Validate that the fold model is supported for unconditional evaluation.
        """
        if self.fold_model_name not in ['esmfold']:
            logging.error('[Unconditional Reducer] Available fold model: esmfold')
            exit(0)

    def _compile(self, pdbs_dir, structures_dir, results_dir, verbose):
        """
        Compile per-design metrics for unconditional generation.

        For each folded structure, computes confidence scores, secondary
        structure content, self-consistency TM-score, CA RMSD, and all-atom RMSD.

        Args:
            pdbs_dir: Directory containing original generated PDB files
            structures_dir: Directory containing folded structure outputs
            results_dir: Directory to write compiled results
            verbose: Whether to display a progress bar
        """

        ##################
        ###   Set up   ###
        ##################

        # Load sequences
        sample_name_to_seq = parse_fasta(os.path.join(structures_dir, 'input.fasta'))

        ###################
        ###   Compile   ###
        ###################

        # Iterate
        rows = []
        for dirpath in tqdm(
            glob.glob(os.path.join(structures_dir, '*', '')),
            desc='Compiling', disable=not verbose
        ):

            # Iterate through designed structures
            for design_filepath in glob.glob(os.path.join(dirpath, '*.pdb')):
                
                #############################
                ###   Part 1: Structure   ###
                #############################

                info = self.fold_model.compile(design_filepath)
                info['model'] = self.fold_model_name
                info['generation_filepath'] = os.path.join(pdbs_dir, f'{info["domain"]}.pdb')

                ############################
                ###   Part 2: Sequence   ###
                ############################

                seq = sample_name_to_seq[info['name']]
                length = len(seq)
                info.update({
                    'len': length,
                    'seq': seq
                })

                ##############################
                ###   Part 3: Confidence   ###
                ##############################

                # Sanity check
                assert len(info['plddt']) == length
                assert info['pae'].shape == (length, length)

                # Update
                info.update({
                    'avg_plddt': np.mean(info['plddt']),
                    'avg_pae': np.mean(info['pae'])
                })

                # Clean
                del info['plddt']
                del info['pae']

                #######################################
                ###   Part 4: Secondary Structure   ###
                #######################################

                info.update(assign_secondary_structure(info['design_filepath'], chain='A'))

                ###############################
                ###   Part 5: Consistency   ###
                ###############################

                info.update({
                    'sctm': compute_tmscore(
                        info['generation_filepath'],
                        info['design_filepath']
                    ),
                    'scrmsd': compute_multimer_ca_rmsd(
                        pdb1=info['generation_filepath'], 
                        pdb2=info['design_filepath'], 
                        chain_ids=['A'],
                        anchor='A'
                    )[0],
                    'allatom_scrmsd': compute_allatom_rmsd(
                        pdb1=info['generation_filepath'],
                        pdb2=info['design_filepath']
                    )
                })

                # Update
                rows.append(info)

        # Save
        df = pd.DataFrame(rows)
        df.to_csv(
            os.path.join(results_dir, 'info.csv'),
            index=False
        )
    
    def _filter(self, results_dir):
        """
        Filter designs by self-consistency RMSD and save successful designs.

        Applies scrmsd < 2Å threshold, de-duplicates by domain, and copies
        successful PDB files to the successes directory.

        Args:
            results_dir: Directory containing the compiled info.csv
        """
        
        ##################
        ###   Set up   ###
        ##################

        # Validate input files
        df = pd.read_csv(os.path.join(results_dir, 'info.csv'))

        # Create output directory for successes
        successes_dir = os.path.join(results_dir, 'successful_generations')
        if os.path.exists(successes_dir):
            logging.error(f'Directory existed: {successes_dir}')
            exit(0)
        os.makedirs(successes_dir)
        self._add_successes_dir(successes_dir)

        ##################
        ###   Filter   ###
        ##################

        # Filter
        df_success = df[df['scrmsd'] < 2]

        # De-duplicate predictions by taking samples with the lowest scrmsd
        df_success = df_success.sort_values(by='scrmsd', ascending=True)
        df_success = df_success.drop_duplicates('domain', keep='first')
        df_success.to_csv(os.path.join(results_dir, 'successful_generation_info.csv'), index=False)

        # Extract PDB files
        for _, row in df_success.iterrows():
            shutil.copyfile(
                row['design_filepath'],
                os.path.join(successes_dir, f'{row["domain"]}.pdb')
            )