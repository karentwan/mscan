from models.multi_scale_loss import MultiScaleLoss
from .base import Base


class MultiScaleTrainer(Base):

    def __init__(self, args, model, infor='multi scale'):
        super(MultiScaleTrainer, self).__init__(args)
        self.model = model
        self.loss_fn = MultiScaleLoss().cuda()
        super(MultiScaleTrainer, self).trainer_initial(self.model, self.loss_fn)
        print('=========================>prepare {} model<===================='.format(infor))

    def get_model_out(self, blur):
        return self.model(blur)[-1]
