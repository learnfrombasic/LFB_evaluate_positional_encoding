import torch


def count_parameters(model: torch.nn.Module, verbose: bool = True) -> dict:
    """Count number of parameters in PyTorch model,
    References: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7.

    from utils.utils import count_parameters
    count_parameters(model)
    import sys
    sys.exit(1)
    """
    n_all = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = n_all - n_trainable
    if verbose:
        print(
            "Parameter Count: all {:,d}; trainable {:,d}; frozen {:,d}".format(
                n_all, n_trainable, n_frozen
            )
        )
    return {
        "n_all": n_all,
        "n_trainable": n_trainable,
        "n_frozen": n_frozen,
    }
