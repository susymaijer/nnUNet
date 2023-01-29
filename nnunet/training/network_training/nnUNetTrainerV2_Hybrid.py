# COPYRIGHT NOTICE THESIS SMAIJER
# Adapted from standard nnU-Net nnUnetTrainerV2 class 

import torch
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.susy.hybrid import Hybrid
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn

class nnUNetTrainerV2_Hybrid(nnUNetTrainerV2):

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 1500

    def initialize_network(self):
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d
        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}

        # create hybrid network
        self.network = Hybrid(self.num_input_channels, 
                                self.num_classes, 
                                self.patch_size, 
                                self.net_pool_per_axis,
                                len(self.net_num_pool_op_kernel_sizes), 
                                feature_size = self.base_num_features, 
                                num_conv_per_stage=self.conv_per_stage, 
                                conv_op=conv_op, 
                                norm_op=norm_op, 
                                norm_op_kwargs=norm_op_kwargs,
                                dropout_op=dropout_op,
                                dropout_op_kwargs=dropout_op_kwargs,
                                nonlin=net_nonlin,
                                nonlin_kwargs=net_nonlin_kwargs,
                                deep_supervision=True, 
                                dropout_in_localization=False, 
                                final_nonlin=lambda x: x, 
                                weightInitializer=InitWeights_He(1e-2),
                                conv_kernel_sizes=self.net_conv_kernel_sizes,
                                pool_op_kernel_sizes=self.net_num_pool_op_kernel_sizes,
                                upscale_logits=False, 
                                convolutional_upsampling=True)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper