#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
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

    def initialize_network(self):
        # u-net variables
        self.conv_op = nn.Conv3d
        dropout_op = nn.Dropout3d
        norm_op = nn.InstanceNorm3d
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
                                feature_size = self.base_num_features, ## till here its same as unetr argumetns
                                num_conv_per_stage=self.conv_per_stage, 
                                conv_op=self.conv_op, 
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
                                upscale_logits=False, 
                                convolutional_upsampling=True,
                                do_print=False # till here its u-net
                                    )
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper