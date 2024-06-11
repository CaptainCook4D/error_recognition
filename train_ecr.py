import wandb
from base import fetch_model_name, train_step_test_step_dataset_base, train_sub_step_test_step_dataset_base, \
    train_model_base
from core.config import Config
from core.utils import init_logger_and_wandb
from constants import Constants as const


def train_sub_step_test_step_ecr(config):
    train_loader, val_loader, test_loader = train_sub_step_test_step_dataset_base(config)
    train_model_base(train_loader, val_loader, config)


def train_step_test_step_ecr(config):
    train_loader, val_loader, test_loader = train_step_test_step_dataset_base(config)
    train_model_base(train_loader, val_loader, config, test_loader=test_loader)


def main():
    conf = Config()
    conf.task_name = const.ERROR_CATEGORY_RECOGNITION
    if conf.model_name is None:
        m_name = fetch_model_name(conf)
        conf.model_name = m_name
    conf.print_config()
    init_logger_and_wandb(conf)
    train_step_test_step_ecr(conf)
    wandb.finish()


if __name__ == "__main__":
    main()
