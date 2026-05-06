"""
Forward (identity) inverse fold handler for the Genie pipeline.

Passes backbone structures directly through without sequence redesign,
using the original sequence from the generated PDB file.
"""

import os
import glob
import json
import logging
from genie3.evaluation.model.inverse_fold.base import InverseFoldHandler
from genie3.evaluation.utils.parse import parse_seq_from_pdb


class ForwardHanlder(InverseFoldHandler):
    """
    InverseFoldHandler that uses the original sequence from the PDB file.

    Supports unconditional, scaffold, and binder pipeline versions.
    """
    
    def inverse_fold(self, pdbs_dir, sequences_dir):
        """
        Extract sequences from PDB files and write them as FASTA files.

        For each PDB file in pdbs_dir, reads the amino acid sequence from
        chain A and writes it to a FASTA file in sequences_dir.

        Args:
            pdbs_dir: Directory containing input PDB files
            sequences_dir: Directory to write output FASTA files
        """
        if self.version in ['unconditional', 'scaffold']:
            self._inverse_fold_basic(pdbs_dir, sequences_dir)
        elif self.version == 'binder':
            self._inverse_fold_binder(pdbs_dir, sequences_dir)
    
    def _inverse_fold_basic(self, pdbs_dir, sequences_dir):
        """
        Extract sequences from PDB files for unconditional or scaffold design.

        Reads chain A sequence from each PDB file and writes it as a FASTA file.

        Args:
            pdbs_dir: Directory containing input PDB files
            sequences_dir: Directory to write output FASTA files
        """

        for pdb_filepath in glob.glob(os.path.join(pdbs_dir, '*.pdb')):
            domain_name = pdb_filepath.split('/')[-1].split('.')[0]
            sequences_filepath = os.path.join(sequences_dir, f'{domain_name}.fasta')

            # Parse
            seq = parse_seq_from_pdb(pdb_filepath, chain='A')

            # Save
            with open(sequences_filepath, 'w') as file:
                file.write(f'>{domain_name}\n{seq}\n')

    def _inverse_fold_binder(self, pdbs_dir, sequences_dir):
        """
        Extract binder and target sequences from PDB files for binder design.

        Reads chain A as the binder sequence, then appends the target sequence
        from the problem configuration to form a complex FASTA entry.

        Args:
            pdbs_dir: Directory containing input PDB files
            sequences_dir: Directory to write output FASTA files
        """
        for pdb_filepath in glob.glob(os.path.join(pdbs_dir, '*.pdb')):
            domain_name = pdb_filepath.split('/')[-1].split('.')[0]
            sequences_filepath = os.path.join(sequences_dir, f'{domain_name}.fasta')

            # Parse binder sequence
            binder_seq = parse_seq_from_pdb(pdb_filepath, chain='A')

            # Parse target sequences from fasta file
            problem_name = domain_name.rsplit('_', 1)[0]
            with open(os.path.join(self.datadir, 'problems', f'{problem_name}.json')) as file:
                info = json.load(file)
            with open(info['target_fasta_filepath']) as file:
                for line in file:
                    if line.startswith('>'):
                        if line.strip()[1:] != problem_name:
                            logging.error(f'Mismatch problem name in fasta file: {line.strip()[1:]}')
                            exit(0)
                    else:
                        target_seq = line.strip()
                        break

            # Save
            with open(sequences_filepath, 'w') as file:
                file.write(f'>{domain_name}\n{binder_seq}:{target_seq}\n')