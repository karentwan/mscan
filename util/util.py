import matplotlib.pyplot as plt
import numpy as np
import os
import json
import torchvision


class PlotPSNR(object):

    def __init__(self,
                 step,
                 exp_dir='.',
                 file_name='psnr_data'):
        self._psnr = np.zeros(shape=(0, ))
        self.step = step
        self._coords = np.zeros(shape=(0, ))
        self.counter = 0
        self.exp_dir = exp_dir
        self.file_name = file_name

    def save(self):
        name = os.path.join(self.exp_dir, self.file_name)
        np.savez(name, coords=self._coords, psnr=self._psnr)

    def load(self):
        name = os.path.join(self.exp_dir, '{}.npz'.format(self.file_name))
        if not os.path.exists(name):
            return
        data = np.load(name)
        self._coords = data['coords']
        self.counter = self._coords[-1]
        self._psnr = data['psnr']

    def write_to_csv(self, name='psnr.csv'):
        path = os.path.join(self.exp_dir, name)
        with open(path, 'w') as f:
            f.write('epoch, psnr\n')
            for epoch, psnr_val in zip(self._coords, self._psnr):
                f.write('{},{}\n'.format(epoch, psnr_val))

    def plot(self, psnr, epoch, name='psnr.png'):
        self._psnr = np.append(self._psnr, psnr)
        # self.counter += self.step
        self._coords = np.append(self._coords, epoch)
        # plt.figure()
        # plt.title('PSNR')
        # plt.plot(self._coords, self._psnr)
        # save_path = os.path.join(self.exp_dir, name)
        # plt.savefig(save_path)
        # plt.close()


class SaveBestPSNR(object):

    def __init__(self, exp_dir='.', name='best_psnr_val.json'):
        self.exp_dir = exp_dir
        self.name = name

    def save(self, psnr, epoch):
        obj = {
            'psnr': psnr,
            'epoch': epoch
        }
        path = os.path.join(self.exp_dir, self.name)
        with open(path, 'w') as f:
            json.dump(obj, f)

    def load(self):
        path = os.path.join(self.exp_dir, self.name)
        if not os.path.exists(path):
            return 0, 0
        with open(path, 'r') as f:
            obj = json.load(f)
            return obj['psnr'], obj['epoch']


def save_feature(tensor, scale, name, path=r'E:\experimental\dccan_share\two_level_plus_with_skip\feature_out'):
    feature_val = tensor.cpu() + 0.5
    path = os.path.join(path, 'scale_{}_name_{}.png'.format(scale, name))
    torchvision.utils.save_image(feature_val, path)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def print_model(*args):
    for item in args:
        print_network(item)
