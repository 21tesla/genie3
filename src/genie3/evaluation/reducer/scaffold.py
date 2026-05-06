"""
Scaffold reducer for the Genie pipeline.

Compiles and filters results for motif-scaffolding protein generation,
computing self-consistency metrics and motif agreement (backbone and all-atom RMSD).
"""

from collections import OrderedDict
import os
import glob
import shutil
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

from genie3.evaluation.np.residue import AA_NAME_TO_ATOM_NAMES
from genie3.evaluation.reducer.base import Reducer
from genie3.evaluation.utils.metric import compute_aligned_rmsd, compute_multimer_ca_rmsd, compute_tmscore
from genie3.evaluation.utils.parse import parse_fasta, parse_motif, parse_problem_name_from_fasta
from genie3.evaluation.utils.secondary import assign_secondary_structure


class ScaffoldReducer(Reducer):
    """Reducer for motif-scaffolding protein generation evaluation."""

    def _validate(self):
        """
        Validate that the fold model is supported for scaffold evaluation.
        """
        if not os.path.join(self.datadir):
            logging.error('[Scaffold Reducer] Data directory not provided')
            exit(0)
        if self.fold_model_name not in ['esmfold']:
            logging.error('[Scaffold Reducer] Available fold model: esmfold')
            exit(0)

    def _compile(self, pdbs_dir, structures_dir, results_dir, verbose):
        """
        Compile per-design metrics for scaffold generation.

        For each folded structure, computes confidence scores, secondary
        structure content, self-consistency metrics, and motif agreement RMSD.

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

        # Parse scaffold information
        df = pd.read_csv(
            os.path.join(
                structures_dir.rsplit('/', 1)[0],
                'scaffold_info.tsv'
            ),
            sep='\t'
        )
        config_str_by_name = df.set_index('name')['config'].to_dict()

        # Parse motif information
        problem_name = parse_problem_name_from_fasta(os.path.join(structures_dir, 'input.fasta'))
        motif_segments_by_id = parse_motif(self.datadir, problem_name)

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
                    )[0]
                })

                ###################################
                ###   Part 6: Motif Agreement   ###
                ###################################

                info.update(
                    self._compute_motif_agreement(
                        domain=info['domain'],
                        design_filepath=info['design_filepath'],
                        config_str_by_name=config_str_by_name,
                        motif_segments_by_id=motif_segments_by_id
                    )
                )

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
        Filter designs by self-consistency RMSD and motif agreement, saving successes.

        Applies three tiers of success criteria:
        - Backbone success: scrmsd < 2Å and motif backbone RMSD < 1Å
        - All-atom success: above + motif all-atom RMSD < 2Å
        - Strict all-atom success: above + motif all-atom RMSD < 1Å

        Args:
            results_dir: Directory containing the compiled info.csv
        """
        
        ##################
        ###   Set up   ###
        ##################

        # Validate input files
        df = pd.read_csv(os.path.join(results_dir, 'info.csv'))

        # Create output directory for backbone successes
        backbnone_successes_dir = os.path.join(results_dir, 'successful_backbone_generations')
        if os.path.exists(backbnone_successes_dir):
            logging.error(f'Directory existed: {backbnone_successes_dir}')
            exit(0)
        os.makedirs(backbnone_successes_dir)
        self._add_successes_dir(backbnone_successes_dir)

        # Create output directory for backbone successes
        allatom_successes_dir = os.path.join(results_dir, 'successful_allatom_generations')
        if os.path.exists(allatom_successes_dir):
            logging.error(f'Directory existed: {allatom_successes_dir}')
            exit(0)
        os.makedirs(allatom_successes_dir)
        self._add_successes_dir(allatom_successes_dir)

        # Create output directory for backbone successes
        allatom_strict_successes_dir = os.path.join(results_dir, 'successful_allatom_strict_generations')
        if os.path.exists(allatom_strict_successes_dir):
            logging.error(f'Directory existed: {allatom_strict_successes_dir}')
            exit(0)
        os.makedirs(allatom_strict_successes_dir)
        self._add_successes_dir(allatom_strict_successes_dir)

        ##################
        ###   Filter   ###
        ##################

        # Filter
        df_success = df[df['scrmsd'] < 2]

        # De-duplicate predictions by taking samples with the lowest motif backbone rmsd
        df_backbone_success = df_success[df_success['motif_backbone_rmsd'] < 1]
        df_backbone_success = df_backbone_success.sort_values(by='motif_backbone_rmsd', ascending=True)
        df_backbone_success = df_backbone_success.drop_duplicates('domain', keep='first')
        df_backbone_success.to_csv(os.path.join(results_dir, 'successful_backbone_generation_info.csv'), index=False)

        # Extract PDB files for backbone successes
        for _, row in df_backbone_success.iterrows():
            shutil.copyfile(
                row['design_filepath'],
                os.path.join(backbnone_successes_dir, f'{row["domain"]}.pdb')
            )
        
        # De-duplicate predictions by taking samples with the lowest motif allatom rmsd
        df_allatom_success = df_success[
            (df_success['motif_backbone_rmsd'] < 1) &
            (df_success['motif_allatom_rmsd'] < 2)
        ]
        df_allatom_success = df_allatom_success.sort_values(by='motif_allatom_rmsd', ascending=True)
        df_allatom_success = df_allatom_success.drop_duplicates('domain', keep='first')
        df_allatom_success.to_csv(os.path.join(results_dir, 'successful_allatom_generation_info.csv'), index=False)

        # Extract PDB files for allatom successes
        for _, row in df_allatom_success.iterrows():
            shutil.copyfile(
                row['design_filepath'],
                os.path.join(allatom_successes_dir, f'{row["domain"]}.pdb')
            )

        # De-duplicate predictions by taking samples with the lowest motif allatom rmsd
        df_allatom_strict_success = df_success[
            (df_success['motif_backbone_rmsd'] < 1) &
            (df_success['motif_allatom_rmsd'] < 1)
        ]
        df_allatom_strict_success = df_allatom_strict_success.sort_values(by='motif_allatom_rmsd', ascending=True)
        df_allatom_strict_success = df_allatom_strict_success.drop_duplicates('domain', keep='first')
        df_allatom_strict_success.to_csv(os.path.join(results_dir, 'successful_allatom_strict_generation_info.csv'), index=False)

        # Extract PDB files for strict allatom successes
        for _, row in df_allatom_strict_success.iterrows():
            shutil.copyfile(
                row['design_filepath'],
                os.path.join(allatom_strict_successes_dir, f'{row["domain"]}.pdb')
            )

    def _compute_motif_agreement(self, domain, design_filepath, config_str_by_name, motif_segments_by_id):
        """
        Compute motif agreement RMSD between designed and reference structures.

        Computes calpha, backbone, and all-atom RMSD between the designed
        structure's motif residues and the reference motif coordinates.

        Args:
            domain: Domain identifier string
            design_filepath: Path to the designed structure PDB file
            config_str_by_name: Dict mapping design name to scaffold config string
            motif_segments_by_id: Dict mapping motif segment ID to residue data

        Returns:
            dict: Per-selection RMSD metrics with keys like 'motif_backbone_rmsd',
                  'motif_allatom_rmsd', and verbose per-motif breakdowns
        """

        ##################
        ###   Define   ###
        ##################

        def _get_gt_motif_atom_positions(name, selection):
            """
            Extract ground-truth motif atom positions from the motif segment data.

            Args:
                name: Design name used to look up the motif configuration string
                selection: Atom selection mode ('calpha', 'backbone', or 'allatom')

            Returns:
                dict: Mapping from motif ID to dict with 'motif_atom_positions'
                      and 'motif_atom_mask' arrays
            """

            def _fetch(motif_segment, selection):
                """
                Collect atom positions and masks for a single motif segment.

                Args:
                    motif_segment: List of per-residue dicts with 'atom_mask'
                                   and 'atom_positions' keys
                    selection: Atom selection mode ('calpha', 'backbone', or 'allatom')

                Returns:
                    tuple: (atom_positions array, atom_mask array)
                """
                atom_mask, atom_positions = [], []
                for residue in motif_segment:
                    if selection == 'calpha':
                        atom_mask.append(residue['atom_mask'][1])
                        atom_positions.append(residue['atom_positions'][1])
                    elif selection == 'backbone':
                        atom_mask.extend(residue['atom_mask'][:4])
                        atom_positions.extend(residue['atom_positions'][:4])
                    elif selection == 'allatom':
                        atom_mask.extend(residue['atom_mask'])
                        atom_positions.extend(residue['atom_positions'])
                    else:
                        logging.error(f'Invalid groundtruth motif atom selection: {selection}')
                        exit(0)
                return np.array(atom_positions), np.array(atom_mask)

            motifs = {}
            for elt in config_str_by_name[name].split(','):
                if not elt.isdigit():
                    if elt[0] not in motifs:
                        motifs[elt[0]] = {
                            'motif_atom_positions': [],
                            'motif_atom_mask': []
                        }
                    atom_positions_, atom_mask_  = _fetch(motif_segments_by_id[elt], selection)
                    motifs[elt[0]]['motif_atom_positions'].append(atom_positions_)
                    motifs[elt[0]]['motif_atom_mask'].append(atom_mask_)

            updated_motifs = {}
            for motif_id in motifs:
                updated_motifs[motif_id] = {
                    'motif_atom_positions': np.concatenate(motifs[motif_id]['motif_atom_positions']),
                    'motif_atom_mask': np.concatenate(motifs[motif_id]['motif_atom_mask']).astype(bool)
                }

            return updated_motifs

        def _get_motif_atom_positions(name, filepath, selection):
            """
            Extract predicted motif atom positions from a folded PDB file.

            Parses the PDB file to find residues at motif positions and returns
            their atom coordinates and masks.

            Args:
                name: Design name used to look up the motif configuration string
                filepath: Path to the folded PDB file
                selection: Atom selection mode ('calpha', 'backbone', or 'allatom')

            Returns:
                dict: Mapping from motif ID to dict with 'motif_atom_positions'
                      and 'motif_atom_mask' arrays
            """
            
            # Parse motif residue indices
            motif_residue_index_to_motif_id = {}
            motif_residue_indices = set()
            motif_residue_indices_unknown_aatype = set()
            cur_residue_index = 1
            for elt in config_str_by_name[name].split(','):
                if elt.isdigit():
                    n_scaffold_residue = int(elt)
                    cur_residue_index += n_scaffold_residue
                else:
                    n_motif_residue = len(motif_segments_by_id[elt])
                    for i in range(cur_residue_index, cur_residue_index + n_motif_residue):
                        motif_residue_indices.add(i)
                        motif_residue_index_to_motif_id[i] = elt[0]
                        motif_residue = motif_segments_by_id[elt][i - cur_residue_index]
                        if motif_residue['residue_name'] == 'UNK':
                            motif_residue_indices_unknown_aatype.add(i)
                    cur_residue_index += n_motif_residue
            
            # Parse pdb file
            residues = OrderedDict()
            with open(filepath) as file:
                for line in file:
                    if line.startswith('ATOM'):
                        atom_name = line[12:16].strip()
                        residue_index = int(line[22:26])
                        if residue_index in motif_residue_indices:
                            if residue_index not in residues:
                                residue_name = line[17:20]
                                residues[residue_index] = {
                                    'name': residue_name,
                                    'atoms': {},
                                    'is_unknown': residue_index in motif_residue_indices_unknown_aatype,
                                }
                            residues[residue_index]['atoms'][atom_name] = [
                                float(line[30:38]),
                                float(line[38:46]),
                                float(line[46:54])
                            ]
            
            # Create atom positions
            motifs = {}
            for residue_index in residues:

                # Create placeholder
                motif_id = motif_residue_index_to_motif_id[residue_index]
                if motif_id not in motifs:
                    motifs[motif_id] = {
                        'motif_atom_positions': []
                    }

                # Arrange in atom14 order
                residue_name = residues[residue_index]['name']
                is_unknown = residues[residue_index]['is_unknown']
                atom_positions_ = [
                    residues[residue_index]['atoms'][atom_name]
                    for atom_name in AA_NAME_TO_ATOM_NAMES[residue_name]
                ]

                # Select
                if selection == 'calpha':
                    motifs[motif_id]['motif_atom_positions'].append(atom_positions_[1])
                elif selection == 'backbone':
                    motifs[motif_id]['motif_atom_positions'].extend(atom_positions_[:4])
                elif selection == 'allatom':
                    if is_unknown:
                        motifs[motif_id]['motif_atom_positions'].extend(atom_positions_[:4])
                    else:
                        motifs[motif_id]['motif_atom_positions'].extend(atom_positions_)
                else:
                    logging.error(f'Invalid motif atom selection: {selection}')
                    exit(0)
            
            # Postprocess
            for motif_id in motifs:
                motifs[motif_id]['motif_atom_positions'] = np.array(motifs[motif_id]['motif_atom_positions'])

            return motifs

        ###################
        ###   Process   ###
        ###################

        info = {}
        for selection in ['calpha', 'backbone', 'allatom']:

            # Parse
            motifs = _get_motif_atom_positions(domain, design_filepath, selection)
            gt_motifs = _get_gt_motif_atom_positions(domain, selection)

            # Sanity check
            assert len(motifs) == len(gt_motifs)
            for motif_id in gt_motifs:
                assert motif_id in gt_motifs, f'Missing motif id: {motif_id}'
                assert motifs[motif_id]['motif_atom_positions'].shape[0] == gt_motifs[motif_id]['motif_atom_mask'].shape[0], 'Mismatch in shape'
                assert motifs[motif_id]['motif_atom_positions'].shape == gt_motifs[motif_id]['motif_atom_positions'].shape, 'Mismatch in shape'
            
            # Process by motif
            motif_ids = []
            rmsds = []
            n_missing_atoms = []
            for motif_id in motifs:
                gt_motif_atom_mask = gt_motifs[motif_id]['motif_atom_mask']
                motif_ids.append(motif_id)
                rmsds.append(
                    compute_aligned_rmsd(
                        motifs[motif_id]['motif_atom_positions'][gt_motif_atom_mask, :],
                        gt_motifs[motif_id]['motif_atom_positions'][gt_motif_atom_mask, :]
                    )
                )
                n_missing_atoms.append(np.sum(gt_motif_atom_mask == 0))

            # Update
            info[f'motif_{selection}_id_verbose'] = '/'.join(motif_ids)
            info[f'motif_{selection}_rmsd_verbose'] = '/'.join([str(rmsd) for rmsd in rmsds])
            info[f'motif_{selection}_n_missing_atom_verbose'] = '/'.join([str(n_missing_atom) for n_missing_atom in n_missing_atoms])
            info[f'motif_{selection}_rmsd'] = np.max(rmsds)
        
        return info
