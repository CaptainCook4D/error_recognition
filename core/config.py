from argparse import ArgumentParser


# TODO: Finish all the configuration parameters.
# Use this as source for training the model

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
        self.features_directory = "/data/rohith/captain_cook/features/gopro/segments"
        self.ckpt_directory = "/data/rohith/captain_cook/checkpoints/error_recognition"
        self.split = 'environment'
        self.batch_size = 1
        self.test_batch_size = 1
        self.num_epochs = 100
        self.lr = 1e-4
        self.weight_decay = 1e-3
        self.log_interval = 5
        self.dry_run = False
        self.ckpt = None
        self.seed = 1000
        self.device = 'cuda'
        self.parser = self.setup_parser()
        self.args = vars(self.parser.parse_args())
        self.save_model = True
        self.__dict__.update(self.args)

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
        parser.add_argument('--test-batch-size', type=int, default=1, help='input batch size for testing (default: 1000)')
        parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
        parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay')
        parser.add_argument('--ckpt', type=str, default=None, help='checkpoint path')
        parser.add_argument('--seed', type=int, default=1000, help='random seed (default: 1000)')

        parser.add_argument('--backbone', type=str, default='omnivore', help='backbone model')
        parser.add_argument('--modality', type=str, default='video', help='modality')
        parser.add_argument('--features_directory', type=str, default='/data/rohith/captain_cook/features/gopro'
                                                                      '/segments', help='features directory')
        parser.add_argument('--ckpt_directory', type=str, default='/data/rohith/captain_cook/checkpoints'
                                                                  '/error_recognition', help='checkpoint directory')
        parser.add_argument('--split', type=str, default='environment', help='split')
        return parser
