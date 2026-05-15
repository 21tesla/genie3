"""
Registry for fold model handlers.

Provides a factory function to instantiate the appropriate FoldHandler
subclass based on the model name.
"""

import logging


def get_fold_model(
    name,
    version,
    mode,
    preload=False,
    device=None,
    datadir=None,
    backend='subprocess',
    verbose=False,
    **kwargs,
):
    """
    Instantiate and optionally preload a fold model handler.

    Args:
        name: Model name; one of 'esmfold', 'colabfold', 'boltz2'
        version: Pipeline version string (e.g. 'binder', 'scaffold')
        mode: Folding mode (e.g. 'single', 'msa', 'template')
        preload: Whether to load the model weights immediately
        device: GPU device index (required when preload=True)
        datadir: Data directory path (used by some models)

    Returns:
        FoldHandler: Configured fold model handler instance
    """
    if name == 'esmfold':
        from genie3.evaluation.model.fold.esmfold import ESMFoldHandler
        return ESMFoldHandler(
            version=version,
            mode=mode,
            device=device,
            datadir=datadir,
            preload=preload,
            verbose=verbose,
            **kwargs,
        )
    elif name == 'colabfold':
        from genie3.evaluation.model.fold.colabfold import ColabFoldHandler
        return ColabFoldHandler(
            version=version,
            mode=mode, 
            device=device, 
            datadir=datadir,
            preload=preload,
            backend=backend,
            verbose=verbose,
            **kwargs,
        )
    elif name == 'boltz2':
        from genie3.evaluation.model.fold.boltz2 import Boltz2Handler
        return Boltz2Handler(
            version=version,
            mode=mode, 
            device=device, 
            datadir=datadir,
            preload=preload,
            verbose=verbose,
        )
    else:
        logging.error(f'Invalid fold model name: {name}')
        exit(0)
