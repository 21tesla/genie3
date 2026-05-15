"""
ESMFold handler for the Genie pipeline.

Wraps the ESMFold model for single-sequence protein structure prediction,
handling model loading, inference, and output organization.
"""

import os
import json
import torch
import logging
import numpy as np

import esm

from genie3.evaluation.model.fold.base import FoldHandler
from genie3.evaluation.utils.parse import parse_fasta


class ESMFoldHandler(FoldHandler):
    """FoldHandler implementation using the ESMFold model."""

    NAME = 'esmfold'

    def __init__(self, *args, **kwargs):
        """
        Initialize ESMFoldHandler.

        Args:
            *args / **kwargs: Forwarded to FoldHandler.__init__.
        """
        self.num_recycles = kwargs.pop('num_recycles', 3)
        # Pop other kwargs that FoldHandler doesn't accept but might be passed from registry
        kwargs.pop('num_models', None)
        kwargs.pop('backend', None)
        super().__init__(*args, **kwargs)

    def setup(self):
        """
        Load the ESMFold model from the ESM library and move it to the device.
        """
        if self.version not in ['unconditional', 'scaffold', 'binder']:
            logging.error(f'Invalid version: {self.version}')
            exit(0)

        self.model = None
        if self.preload:
            self.model = esm.pretrained.esmfold_v1()
            self.model = self.model.eval().to(self.device)
            try:
                # Use bfloat16 on Blackwell/Hopper/Ampere for faster inference
                self.model = self.model.to(torch.bfloat16)
                logging.info(f"[{self.device}] ESMFold moved to bfloat16")
            except Exception as e:
                logging.warning(f"[{self.device}] Failed to move ESMFold to bfloat16: {e}")

    def fold(self, structures_dir):
        """
        Fold all sequences in the input FASTA and save predicted structures.

        Reads sequences from structures_dir/input.fasta, runs ESMFold inference
        on each, and writes per-sequence PDB files to structures_dir.

        Args:
            structures_dir: Directory containing input.fasta and where output
                            PDB files will be written
        """

        # Sanity check
        sequences_filepath = os.path.join(structures_dir, 'input.fasta')
        if not os.path.exists(sequences_filepath):
            logging.error('Missing sequences filepath')
            exit(0)

        # Load sequences
        seq_by_name = parse_fasta(sequences_filepath)

        # Iterate
        for name in seq_by_name:

            # Set up
            outdir = os.path.join(structures_dir, name)
            os.makedirs(outdir)

            # Predict
            with torch.no_grad():
                output = self.model.infer(seq_by_name[name], num_recycles=self.num_recycles)
                
                # Move floating point tensors to float32 for post-processing (numpy/esm library compatibility)
                output = {
                    k: v.to(torch.float32) if (isinstance(v, torch.Tensor) and torch.is_floating_point(v)) else v 
                    for k, v in output.items()
                }

                pdb_str = self.model.output_to_pdb(output)[0]
                pae = (output['aligned_confidence_probs'].cpu().numpy()[0] * np.arange(64)).mean(-1) * 31
                mask = output['atom37_atom_exists'].cpu().numpy()[0, :, 1] == 1
                pae = pae[mask,:][:,mask]
                plddt = output["plddt"][0, :, 1].cpu().numpy()
                plddt = plddt[mask]
                ptm = output['ptm'].cpu().numpy()[0]
            
            # Renumber residues to start from 1 for each chain
            pdb_str = self._renumber_pdb(pdb_str)

            # Save
            with open(os.path.join(outdir, f'{name}.pdb'), 'w') as file:
                file.write(pdb_str)
            with open(os.path.join(outdir, f'scores_{name}.json'), 'w') as file:
                out = {
                    'plddt': plddt.tolist(),
                    'pae': pae.tolist(),
                    'ptm': float(ptm)
                }
                json.dump(out, file, indent=4)

    def _renumber_pdb(self, pdb_str):
        """
        Renumber residues in a PDB string to start from 1 for each chain.

        ESMFold's multimer output often uses large offsets for additional chains;
        this ensures consistency with downstream analysis tools.

        Args:
            pdb_str: Original PDB string from ESMFold

        Returns:
            str: Renumbered PDB string
        """
        lines = pdb_str.splitlines()
        new_lines = []
        current_chain = None
        res_offset = 0

        for line in lines:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                chain = line[21]
                res_num = int(line[22:26])

                if chain != current_chain:
                    current_chain = chain
                    res_offset = res_num - 1

                new_res_num = res_num - res_offset
                new_line = line[:22] + f"{new_res_num:>4}" + line[26:]
                new_lines.append(new_line)
            elif line.startswith('TER'):
                chain = line[21]
                if chain == current_chain:
                    try:
                        res_num = int(line[22:26])
                        new_res_num = res_num - res_offset
                        new_line = line[:22] + f"{new_res_num:>4}" + line[26:]
                        new_lines.append(new_line)
                    except ValueError:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)

        return "\n".join(new_lines) + "\n"

    def _compile_filepath(self, design_filepath):
        """
        Parse name, domain, and file paths from an ESMFold design filepath.

        Args:
            design_filepath: Path to the predicted PDB file

        Returns:
            dict: 'name', 'domain', 'design_filepath', 'info_filepath'
        """
        sample_name = design_filepath.split('/')[-2]
        domain_name = sample_name.split('-')[0]
        resample_index = int(sample_name.split('-')[-1].split('_')[-1])

        elts = design_filepath.split('/')
        elts = elts[:-1] + ['scores_' + elts[-1].replace('.pdb', '.json')]
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
        Return rank 1 for ESMFold (single-model, no ranking).

        Args:
            design_filepath: Path to the predicted PDB file (unused)

        Returns:
            dict: {'rank': 1}
        """
        return {'rank': 1}
    
    def _compile_confidence(self, info_filepath):
        """
        Extract pLDDT, pAE and pTM from an ESMFold JSON file.

        Args:
            info_filepath: Path to the JSON scores file

        Returns:
            dict: 'plddt', 'pae', 'ptm'
        """
        with open(info_filepath) as file:
            info = json.load(file)
        return {
            'plddt': np.array(info['plddt']),
            'pae': np.array(info['pae']),
            'ptm': info.get('ptm', 0.0)
        }
    
    def _compile_ipsae(self, info_filepath, design_filepath):
        """
        Not implemented for ESMFold (no ipSAE output).

        Args:
            info_filepath: Path to the JSON scores file
            design_filepath: Path to the predicted structure file
        """
        return {}