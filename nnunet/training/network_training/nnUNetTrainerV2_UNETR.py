import torch 
from torch import nn
from nnunet.network_architecture.susy.unetr import UNETR
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2

class nnUNetTrainerV2_UNETR(nnUNetTrainerV2):

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)

    def initialize_network(self):     
        if self.threeD:
            conv_op = nn.Conv3d
        else:
            conv_op = nn.Conv2d
        self.network = UNETR(self.num_input_channels, self.num_classes, self.patch_size, self.net_pool_per_axis,
                                len(self.net_num_pool_op_kernel_sizes), self.net_num_pool_op_kernel_sizes,
                                self.net_conv_kernel_sizes, conv_op)
        if torch.cuda.is_available():
            self.network.cuda()
