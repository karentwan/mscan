import argparse
import sys
from trainer.trainer_factory import TrainerFactory


sys.path.append('.')

parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', type=str, default=r'E:\experimental\deblur\dccan_share_pytorch_model\two_level_plus_with_skip\mscan', help='the directory of experiment')
parser.add_argument('--input_path', type=str, default=r'E:\experimental\deblur\code_and_data\Gopro', help='real image path')
parser.add_argument('--output_path', type=str, default=r'E:\experimental\deblur\dccan_share_pytorch_model\two_level_plus_with_skip\mscan', help='test result output path')
parser.add_argument('--save_model_name', type=str, default='mscan', help='the name of the saved model')
parser.add_argument('--trainer_name', type=str, default='MSCAN',
                    choices=['MSCAN'], help='optional model')

args = parser.parse_args()

if __name__ == '__main__':
    trainer_factory = TrainerFactory(args)
    trainer = trainer_factory.get_trainer(args.trainer_name)
    assert trainer is not None, 'please specify the correct name of the trainer'
    print('==============>start testing ...')
    trainer.eval()
