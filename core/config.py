from argparse import ArgumentParser
from constants import Constants as const


class Config(object):
    """Wrapper class for model hyperparameters."""

    def __init__(self):
        """
        Defaults
        """
        self.backbone = 'omnivore'
        self.modality = 'video'
        self.phase = 'train'
        self.segment_length = 1

        self.video_features_directory = "/data/rohith/captain_cook/features/gopro/segments"
        self.audio_features_directory = "/data/rohith/captain_cook/features/gopro/audios"
        self.text_features_directory = "/data/rohith/captain_cook/features/gopro/texts"
        self.depth_features_directory = "/data/rohith/captain_cook/features/gopro/depths"

        self.ckpt_directory = "/data/rohith/captain_cook/checkpoints/"
        self.split = 'recordings'
        self.batch_size = 1
        self.test_batch_size = 1
        self.num_epochs = 50
        self.lr = 1e-3
        self.weight_decay = 1e-3
        self.log_interval = 5
        self.dry_run = False
        self.ckpt = None
        self.seed = 1000
        self.device = 'cuda'

        self.variant = const.MLP_VARIANT
        self.model_name = None
        self.task_name = const.ERROR_RECOGNITION

        self.model_name = None

        self.parser = self.setup_parser()
        self.args = vars(self.parser.parse_args())
        self.save_model = True
        self.__dict__.update(self.args)

        if self.backbone == const.IMAGEBIND:
            if self.modality is None:
                self.modality = [const.VIDEO]
        else:
            self.modality = [const.VIDEO]

    def setup_parser(self):
        """
        Sets up an argument parser
        :return:
        """
        parser = ArgumentParser(description='training code')

        # ----------------------------------------------------------------------------------------------
        # CONFIGURATION PARAMETERS
        # ----------------------------------------------------------------------------------------------

        parser.add_argument('--batch_size', type=int, default=1, help='batch size')
        parser.add_argument('--test-batch-size', type=int, default=1,
                            help='input batch size for testing (default: 1000)')
        parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
        parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay')
        parser.add_argument('--ckpt', type=str, default=None, help='checkpoint path')
        parser.add_argument('--seed', type=int, default=42, help='random seed (default: 1000)')

        parser.add_argument('--backbone', type=str, default=const.IMAGEBIND, help='backbone model')
        parser.add_argument('--features_directory', type=str, default='/data/rohith/captain_cook/features/gopro'
                                                                      '/segments', help='features directory')
        parser.add_argument('--ckpt_directory', type=str, default='/data/rohith/captain_cook/checkpoints',
                            help='checkpoint directory')
        parser.add_argument('--split', type=str, default=const.PERSON_SPLIT, help='split')
        parser.add_argument('--variant', type=str, default=const.TRANSFORMER_VARIANT, help='variant')
        parser.add_argument('--model_name', type=str, default=None, help='model name')
        parser.add_argument('--task_name', type=str, default=const.ERROR_RECOGNITION, help='task name')
        parser.add_argument('--modality', type=str, nargs='+', default=[const.VIDEO], help='audio')

        return parser

    def set_model_name(self, model_name):
        self.model_name = model_name
