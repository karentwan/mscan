from .base import Base


class MultiScaleTrainer(Base):

    def __init__(self, args, model, infor='multi scale'):
        super(MultiScaleTrainer, self).__init__(args)
        self.model = model
        print('=========================>prepare {} model<===================='.format(infor))

    def get_model_out(self, blur):
        return self.model(blur)[-1]
