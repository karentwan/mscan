from trainer.multi_scale_trainer import MultiScaleTrainer
from models.mscan import MSCAN


class TrainerFactory(object):

    def __init__(self, args):
        self._args = args

    def get_trainer(self, key):
        if key == 'MSCAN':
            model = MSCAN()
            return MultiScaleTrainer(self._args, model, infor='MSCAN')
        return None
