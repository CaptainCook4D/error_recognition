from argparse import ArgumentParser


# TODO: Finish all the configuration parameters.
# Use this as source for training the model

class Config(object):
    """Wrapper class for model hyperparameters."""

    def __init__(self):
        """
        Defaults
        """
        self.backbone = 'resnet50'
        self.modality = 'video'
        self.phase = 'train'
        self.segment_length = 1

        self.ckpt = None

        self.parser = self.setup_parser()
        self.args = vars(self.parser.parse_args())
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

        parser.add_argument('--batch_size', type=int, default=32, help='batch size')
        parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')

        parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
        parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
        parser.add_argument('--ckpt', type=str, default=None, help='checkpoint path')

        return parser
