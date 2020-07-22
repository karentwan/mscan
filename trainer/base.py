from torchvision import transforms
import torchvision
import os
import torch
import util.util as util
import math
import time
import torch.nn.functional as F
from PIL import Image


class Base(object):

    def __init__(self, args):
        super(Base, self).__init__()
        self.args = args
        self.model = None
        # save result
        self.output = None
        # save max value calculated
        self.exp_dir = self.args.exp_dir  # the dir of save model and test result
        self.save_model_name = self.args.save_model_name
        self.best_psnr = 0.
        self.best_epoch = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_model_out(self, blur):
        '''
        returns the result of the model, and when the result of the model has three scales,
        returns the result of the largest scale.
        the function only be invoked during testing
        :param blur: blur image
        :return:
        '''
        return self.model(blur)

    def eval_step(self, input_path, output_path, image_name):
        path = os.path.join(input_path, image_name)
        B1 = transforms.ToTensor()(Image.open(path).convert('RGB'))
        B1 = (B1 - 0.5).unsqueeze(0).to(self.device)
        # make sure the picture is a multiple of 16
        [b, c, h, w] = B1.shape
        new_h = h - h % 16
        new_w = w - w % 16
        B1 = F.interpolate(B1, size=(new_h, new_w), mode='bilinear')
        start = time.time()
        deblur = self.get_model_out(B1).cpu() + 0.5
        duration = time.time() - start
        print('image:{}\ttime:{:.4f}'.format(image_name, duration))
        path = os.path.join(output_path, image_name)
        torchvision.utils.save_image(deblur.data, path)

    def eval(self):
        self.model = self.model.to(self.device)
        util.print_model(self.model)  # calc model size
        epoch = self.restore_model()
        print(' best model in epoch:{}'.format(epoch))
        input_path = self.args.input_path
        images = os.listdir(input_path)
        output_path = self.args.output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for image_name in images:
            with torch.no_grad():
                self.eval_step(input_path, output_path, image_name)

    def restore_model(self, best=False):
        path = os.path.join(self.exp_dir, '{}.pth'.format(self.save_model_name))
        if best:
            path = os.path.join(self.exp_dir, 'best_{}.pth'.format(self.save_model_name))
        if not os.path.exists(path):
            return 0
        if torch.cuda.is_available():
            model_dict = torch.load(path)
        else:
            model_dict = torch.load(path, map_location=torch.device('cpu'))
        epoch = model_dict['epoch']
        self.model.load_state_dict(model_dict['model'])
        return epoch