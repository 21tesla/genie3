"""
Registry for pipeline reducer implementations.

Provides a factory function to instantiate the appropriate Reducer
subclass based on the pipeline version.
"""

import logging


def get_reducer(
    *,
    version,
    datadir,
    fold_model_name,
    fold_mode,
    calc_novelty,
):
    """
    Instantiate a reducer for the given pipeline version.

    Args:
        version: Pipeline version string; one of 'unconditional', 'scaffold', 'binder'
        datadir: Data directory path
        fold_model_name: Name of the fold model
        fold_mode: Folding mode
        calc_novelty: Whether to compute novelty metrics

    Returns:
        Reducer: Configured reducer instance
    """
    if version == 'unconditional':
        from genie3.evaluation.reducer.unconditional import UnconditionalReducer
        return UnconditionalReducer(
            version=version,
            datadir=datadir,
            fold_model_name=fold_model_name,
            fold_mode=fold_mode,
            calc_novelty=calc_novelty,
        )
    elif version == 'scaffold':
        from genie3.evaluation.reducer.scaffold import ScaffoldReducer
        return ScaffoldReducer(
            version=version,
            datadir=datadir,
            fold_model_name=fold_model_name,
            fold_mode=fold_mode,
            calc_novelty=calc_novelty,
        )
    elif version == 'binder':
        from genie3.evaluation.reducer.binder import BinderReducer
        return BinderReducer(
            version=version,
            datadir=datadir,
            fold_model_name=fold_model_name,
            fold_mode=fold_mode,
            calc_novelty=calc_novelty,
        )
    else:
        logging.error(f'Invalid reducer version: {version}')
        exit(0)
