import wandb
from loguru import logger


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
    return
