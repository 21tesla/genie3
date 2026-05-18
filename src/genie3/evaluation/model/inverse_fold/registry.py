"""
Registry for inverse fold model handlers.

Provides a factory function to instantiate the appropriate InverseFoldHandler
subclass based on the model name.
"""

import logging


def get_inverse_fold_model(name, version, mode, device, datadir=None, num_samples=8, **kwargs):
    """
    Instantiate an inverse fold model handler.

    Args:
        name: Model name; one of 'forward', 'proteinmpnn'
        version: Pipeline version string (e.g. 'binder', 'scaffold')
        mode: Inverse folding mode
        device: GPU device index
        datadir: Data directory path
        **kwargs: Additional parameters for the handler (e.g. sampling_temperature)

    Returns:
        InverseFoldHandler: Configured inverse fold model handler instance
    """
    if name == 'forward':
        from genie3.evaluation.model.inverse_fold.forward import ForwardHanlder
        return ForwardHanlder(
            version=version,
            mode=mode,
            device=device,
            datadir=datadir
        )
    elif name == 'proteinmpnn':
        from genie3.evaluation.model.inverse_fold.proteinmpnn import ProteinMPNNHanlder
        return ProteinMPNNHanlder(
            version=version,
            mode=mode,
            device=device,
            datadir=datadir,
            num_samples=num_samples,
            **kwargs
        )
    else:
        logging.error(f'Invalid inverse fold model name: {name}')
        exit(0)