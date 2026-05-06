"""
Preparation script for Binder Design problems.

This module parses a configuration YAML file defining binder design targets,
extracts and aligns sequences from PDB structures, generates necessary MSA
(Multiple Sequence Alignment) files using ColabFold, and produces JSON
problem definition files compatible with the Genie 3 evaluation pipeline.
"""

import os
import json
import yaml
import shutil
import logging
import argparse
import subprocess
import numpy as np
from ml_collections import ConfigDict
from typing import Tuple, Dict

from genie3.generation.np.protein_constants import PROTEIN_RESTYPES, RESTYPE_3_TO_1
from genie3.generation.utils.interface.extended import compute_extended_interface
from genie3.generation.utils.pdb_utils import parse_pdb, Chain


def get_chain_seq(chain_id: str, chain: Chain, fullseq: str | None = None) -> Tuple[str, str, Dict[str, str]]:
    """
    Extracts and aligns a chain sequence from a PDB structure to an optional full sequence.

    Args:
        chain_id: Identifier to assign to the extracted chain (e.g., 'A', 'B').
        chain: The parsed Chain object containing residue information.
        fullseq: (Optional) The complete expected sequence for the chain. 
                 If provided, the PDB sequence will be aligned against it to correct indexing offsets.

    Returns:
        A tuple containing:
            - The final sequence (fullseq if provided, otherwise the PDB sequence).
            - A string tag representing the chain and its resolved span (e.g., 'A1-150').
            - A dictionary mapping original PDB residue tags (e.g., 'A25') to their updated, aligned tags.
    """

    # Extract pdb sequence. We initialize it with '-' to account for potential gaps
    # or missing residues in the structure.
    start_index = chain.residues[0].index
    end_index = chain.residues[-1].index
    pdbseq = ['-' for _ in range(start_index, end_index + 1)]
    residue_index_map = {}
    for residue in chain.residues:
        i = residue.index - start_index
        pdbseq[i] = RESTYPE_3_TO_1[PROTEIN_RESTYPES[np.argmax(residue.restype)]]
        residue_tag = f'{chain.name}{residue.index}'
        updated_residue_tag = f'{chain_id}{i+1}'
        residue_index_map[residue_tag] = updated_residue_tag
    pdbseq = ''.join(pdbseq)
    chain_tag = f'{chain_id}1-{end_index - start_index + 1}'

    # Return if fullseq is not provided and no missing residues
    if fullseq is None:
        if len(chain.residues) != len(pdbseq):
            logging.error('Missing residues in pdb sequence requires full sequence specified')
            exit(0)
        assert '-' not in pdbseq, 'Implementation error'
        return pdbseq, chain_tag, residue_index_map
    elif len(pdbseq) > len(fullseq):
        logging.error('Specified full sequence is shorter than pdb sequence')
        exit(0)
    
    # Perform sequence alignment to find the offset between the PDB sequence
    # and the provided full sequence. This corrects for missing N-terminal residues.
    pdbseq_residue_index_offset = None
    n_pdbseq, n_fullseq = len(pdbseq), len(fullseq)
    for start in range(0, n_fullseq - n_pdbseq + 1):
        ok = True
        for i in range(n_pdbseq):
            aa_s = pdbseq[i]
            aa_l = fullseq[start + i]
            if aa_s != '-' and aa_s != aa_l:
                ok = False
                break
        if ok:
            pdbseq_residue_index_offset = start
            break
    if pdbseq_residue_index_offset is None:
        logging.error('Mismatch between pdb sequence and full sequence')
        exit(0)

    # Update chain and residue tags
    chain_id, start_residue_index, end_residue_index = chain_tag[0], int(chain_tag[1:].split('-')[0]), int(chain_tag[1:].split('-')[1])
    start_residue_index += pdbseq_residue_index_offset
    end_residue_index += pdbseq_residue_index_offset
    updated_chain_tag = f'{chain_id}{start_residue_index}-{end_residue_index}'
    updated_residue_index_map = {}
    for residue_tag in residue_index_map:
        residue_index = int(residue_index_map[residue_tag][1:])
        residue_index += pdbseq_residue_index_offset
        updated_residue_index_map[residue_tag] = f'{chain_id}{residue_index}'

    return fullseq, updated_chain_tag, updated_residue_index_map

def process(problem: ConfigDict, outdir: str):
    """
    Process a single binder design problem configuration and generate required input files.

    This function executes the complete data preparation pipeline for a given problem:
    1. Parses the target structure (PDB) and any specified sequences or hotspots.
    2. Validates the PDB file to ensure it does not contain insertion codes or alternative
       variable positions, which are not currently supported.
    3. Aligns the extracted sequences from the structure to the actual full-length sequences
       (if provided) to correct indexing offsets and missing residues at the termini.
    4. Generates standard FASTA and formatting-corrected PDB files for both the full complex
       and the individual chains.
    5. Dispatches ColabFold (`colabfold_batch`) to automatically generate Multiple Sequence
       Alignments (MSA) in `.a3m` format for the sequences.
    6. Computes the target interface (hotspot and extended residues) based on distance
       thresholds from the given hotspots.
    7. Dumps all generated paths, metadata, sequence constraints, and binding interface 
       details into a unified JSON problem definition file.

    Args:
        problem: A nested dictionary/ConfigDict containing the properties of a specific 
                 binder design target (e.g. sequence, filepath, hotspot residues, binder length limits).
        outdir (str): The destination directory where the resulting `problems`, `targets/pdb`, 
                      `targets/fasta`, and `targets/msa` directories reside.
    """
    
    # Parse
    filepath = problem.target.filepath
    fullseqs = (
        problem.target.sequence.split()
        if 'sequence' in problem.target and problem.target.sequence
        else None
    )
    hotspots = problem.target.hotspot.split() if 'hotspot' in problem.target else []

    # Sanity check
    # Raise errors if there are alternative positions or insertion codes (TODO: add support for these)
    pdb_lines = []
    missing_hotspots = set(hotspots)
    with open(filepath) as file:
        for line in file:
            if line.startswith('ATOM'):
                if line[16].strip() != '' or line[26].strip() != '':
                    print('Current version does not support insertion code or alternative locations in PDB file')
                    exit(0)
                if line[12:16].strip() == 'CA':
                    chain_id, residue_index = line[21], int(line[22:26])
                    residue_tag = f'{chain_id}{residue_index}'
                    if residue_tag in missing_hotspots:
                        missing_hotspots.remove(residue_tag)
                pdb_lines.append(line)
    if len(missing_hotspots) > 0:
        print(f'Missing hotspots: {", ".join(list(missing_hotspots))}')
        exit(0)

    # Load and process sequences chain-by-chain
    # We parse the PDB and align each chain to its full sequence representation.
    chain_tags = []
    processed_seqs = []
    residue_index_map = {}
    structure = parse_pdb(filepath)
    if fullseqs is not None and len(fullseqs) != len(structure.chains):
        print('Mismatched number of target sequences')
        exit(0)
    for i, chain in enumerate(structure.chains):
        seq, chain_tag, chain_residue_index_map = get_chain_seq(
            chain_id=chr(ord('A') + i + 1), # Start with B for convenience
            chain=chain,
            fullseq=None if fullseqs is None else fullseqs[i]
        )
        chain_tags.append(chain_tag)
        processed_seqs.append(seq)
        residue_index_map.update(chain_residue_index_map)

    # Create FASTA files
    # The merged sequence is saved for full-complex modeling or MSA generation.
    target_fasta_filepath = os.path.join(outdir, 'targets', 'fasta', f'{problem.key}.fasta')
    with open(target_fasta_filepath, 'w') as file:
        file.write(f'>{problem.key}\n{":".join(processed_seqs)}')
    
    # Create individual FASTA files for each chain
    target_fasta_filepath_by_chain = []
    for i, tag in enumerate(chain_tags):
        target_fasta_chain_filepath = os.path.join(outdir, 'targets', 'fasta', f'{problem.key}-chain_{tag[0]}.fasta')
        target_fasta_filepath_by_chain.append(target_fasta_chain_filepath)
        with open(target_fasta_chain_filepath, 'w') as file:
            file.write(f'>{problem.key}-chain_{tag[0]}\n{processed_seqs[i]}')
    
    # Create PDB file with updated residue and chain tags
    # This prepares the target structures formatted appropriately for Genies 3 evaluation.
    target_pdb_filepath = os.path.join(outdir, 'targets', 'pdb', f'{problem.key}.pdb')
    with open(target_pdb_filepath, 'w') as file:
        lines = [
            f'REMARK 999 KEY    {problem.key}\n',
            f'REMARK 999 NAME   {problem.name}\n'
        ]
        lines += [
            f'REMARK 999 TARGET {tag[0]} {tag[1:].split("-")[0].rjust(4)} {tag[1:].split("-")[1].rjust(4)}\n' 
            for tag in chain_tags
        ]
        for line in pdb_lines:
            chain_id, residue_index = line[21], int(line[22:26])
            residue_tag = f'{chain_id}{residue_index}'
            updated_residue_tag = residue_index_map[residue_tag]
            updated_chain_id, updated_residue_index = updated_residue_tag[0], int(updated_residue_tag[1:])
            updated_line = line[:21] + updated_chain_id + str(updated_residue_index).rjust(4) + line[26:]
            lines += updated_line
        file.write(''.join(lines))
    
    # Create pdb file by chain
    target_pdb_filepath_by_chain = []
    for i, tag in enumerate(chain_tags):
        target_pdb_chain_filepath = os.path.join(outdir, 'targets', 'pdb', f'{problem.key}-chain_{tag[0]}.pdb')
        target_pdb_filepath_by_chain.append(target_pdb_chain_filepath)
        with open(target_pdb_chain_filepath, 'w') as file:
            chain_lines = []
            for line in pdb_lines:
                if line.startswith('ATOM') and line[21] == tag[0]:
                    chain_lines.append(line)
            file.write(''.join(chain_lines))

    # Generate Multiple Sequence Alignment (MSA) using ColabFold.
    # We call `colabfold_batch` on the FASTA file and only keep the resulting `.a3m` MSA file.
    target_msa_filepath = os.path.join(outdir, 'targets', 'msa', f'{problem.key}.a3m')
    target_msa_outdir = os.path.join(outdir, 'targets', f'{problem.key}_msa_output')
    cmd = ' '.join([
        'colabfold_batch',
        target_fasta_filepath,
        target_msa_outdir,
        '--msa-only'
    ])
    subprocess.call(cmd, shell=True)
    shutil.copyfile(
        os.path.join(target_msa_outdir, f'{problem.key}.a3m'),
        target_msa_filepath
    )
    shutil.rmtree(target_msa_outdir)

    # Create msa file by chain
    target_msa_filepath_by_chain = []
    for i, tag in enumerate(chain_tags):
        target_msa_chain_filepath = os.path.join(outdir, 'targets', 'msa', f'{problem.key}-chain_{tag[0]}.a3m')
        target_msa_filepath_by_chain.append(target_msa_chain_filepath)
        target_msa_chain_outdir = os.path.join(outdir, 'targets', f'{problem.key}-chain_{tag[0]}_msa_output')
        cmd = ' '.join([
            'colabfold_batch',
            target_fasta_filepath_by_chain[i],
            target_msa_chain_outdir,
            '--msa-only'
        ])
        subprocess.call(cmd, shell=True)
        shutil.copyfile(
            os.path.join(target_msa_chain_outdir, f'{problem.key}-chain_{tag[0]}.a3m'),
            target_msa_chain_filepath
        )
        shutil.rmtree(target_msa_chain_outdir)

    # Create JSON problem configuration file.
    # This dictionary aggregates paths to generated targets, hotpost definitions, 
    # and sequence length constraints to be ingested by the sampling pipeline.
    updated_hotspots = [
        residue_index_map[hotspot]
        for hotspot in hotspots
    ]
    info = {
        'key': problem.key,
        'name': problem.name,
        'target_pdb_filepath': target_pdb_filepath,
        'target_fasta_filepath': target_fasta_filepath,
        'target_msa_filepath': target_msa_filepath,
        'target_pdb_filepath_by_chain': target_pdb_filepath_by_chain,
        'target_fasta_filepath_by_chain': target_fasta_filepath_by_chain,
        'target_msa_filepath_by_chain': target_msa_filepath_by_chain,
        'target_chain_and_residues': chain_tags,
        'target_interface_residues': {
            'hotspot': updated_hotspots,
            'extended': compute_extended_interface(
                target_pdb_filepath=target_pdb_filepath,
                target_hotspot_residues=updated_hotspots,
                version_num=1
            ),
            **({'common': [residue_index_map[r] for r in problem.target.common.split()]}
               if 'common' in problem.target and problem.target.common else {})
        },
        'binder_min_length': problem.binder.min_length,
        'binder_max_length': problem.binder.max_length
    }
    if 'tag' in problem:
        info.update({'tag': problem.tag.split()})
    if 'other' in problem:
        for other_key in problem.other:
            if other_key in info:
                print(f'Conflict key: {other_key}')
                exit(0)
            info[other_key] = problem.other[other_key]
    problem_filepath = os.path.join(outdir, 'problems', f'{problem.key}.json')
    with open(problem_filepath, 'w') as file:
        json.dump(info, file, indent=4)

def main(args):
    """
    Main entry point for problem set preparation.

    Loads the configuration file, sets up the output directory structure,
    and iterates through each problem defined in the configuration to process it.

    Args:
        args: Parsed command-line arguments containing the configuration filepath and output directory.
    """

    # Load configuration file
    if not os.path.exists(args.config):
        print(f'Missing configuration file: {args.config}')
        exit(0)
    with open(args.config) as file:
        config = ConfigDict(yaml.safe_load(file))
    
    # Set up output directory
    outdir = os.path.join(args.outdir, config.name)
    if os.path.exists(outdir):
        print(f'Output directory existed: {outdir}')
        exit(0)
    os.makedirs(os.path.join(outdir, 'problems'))
    os.makedirs(os.path.join(outdir, 'targets', 'pdb'))
    os.makedirs(os.path.join(outdir, 'targets', 'fasta'))
    os.makedirs(os.path.join(outdir, 'targets', 'msa'))

    # Create problems
    for key in config.problem:
        problem = config.problem[key]
        problem['key'] = key
        process(problem, outdir)


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Configuration filepath', required=True)
    parser.add_argument('--outdir', type=str, help='Output directory', required=True)
    args = parser.parse_args()
    
    # Run
    main(args)