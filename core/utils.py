from loguru import logger
import torch
import wandb


def init_logger_and_wandb(project_name, args):
    import sys

    wandb.init(project=project_name)
    wandb.config.update(args)
    config = {
        "handlers": [
            {
                "sink": sys.stdout,
                "format": "<green>{time:YYYY-MM-DD at HH:mm:ss}</green> | {module}.{function} | <level>{message}</level> ",
            },
            {
                "sink": "logging/" + "logger_{time}.log",
                "format": "<green>{time:YYYY-MM-DD at HH:mm:ss}</green> | {module}.{function} | <level>{message}</level> ",
            },
        ],
        "extra": {"user": "usr"},
    }
    logger.configure(**config)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    # num_classes = 1
    if use_cuda:
        device = torch.device("cuda")
        logger.info("Using GPU for training")

    elif use_mps:
        device = torch.device("mps")
        logger.info("Using MPS for training")

    else:
        device = torch.device("cpu")
        logger.info("Using CPU for training")

    return device, use_cuda, use_mps