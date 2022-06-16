# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# TODO edit this copyright when handing in code
# UNETR based on: "Hatamizadeh et al.,
# UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"

from nnunet.network_architecture.generic_UNet import Generic_UNETDecoder, ConvDropoutNormNonlin
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.susy.unetr import UNETREncoder
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
from typing import Tuple, Union

class Hybrid(SegmentationNetwork):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int, int],
        num_pool_per_axis: Tuple[int, int, int], # smaijer
        num_pool,
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        deep_supervision=True,
        upscale_logits=False, ### till here its pure unetr

        num_conv_per_stage=2,
        conv_op=nn.Conv2d,
        norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
        dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU, nonlin_kwargs=None,  
        dropout_in_localization=False,
        final_nonlin=softmax_helper,
        weightInitializer=InitWeights_He(1e-2),
        conv_kernel_sizes=None,
        convolutional_upsampling=False,
        basic_block=ConvDropoutNormNonlin,
        seg_output_use_bias=False, ## till here it's u-net decoder
        do_print=True
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.

        Examples::

            # for single channel input 4-channel output with patch size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

            # for 4-channel input 3-channel output with patch size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), pos_embed='conv', norm_name='instance')

        """
        super(Hybrid, self).__init__()

        # create encoder
        self.encoder = UNETREncoder(in_channels, img_size, num_pool_per_axis, feature_size, hidden_size, mlp_dim, num_heads, 
                                    pos_embed, norm_name, conv_block, res_block, dropout_rate, do_print)
        skip_features=[feature_size, feature_size*2, feature_size*4, feature_size*8, hidden_size]

        # create decoder 
        num_pool = 4
        print(f'num pool: {num_pool}')
        print(f'conv_kernel_sizes: {conv_kernel_sizes}')
        print(f'convolutional_upsampling: {convolutional_upsampling}')
        self.decoder = Generic_UNETDecoder(out_channels, num_pool, skip_features, num_conv_per_stage, conv_op, norm_op, norm_op_kwargs,
                                            dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, deep_supervision, 
                                            dropout_in_localization, final_nonlin, None, conv_kernel_sizes, 
                                            upscale_logits, convolutional_upsampling, basic_block, seg_output_use_bias, do_print)

        # Necessary for nnU-net
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

        # register all modules properly
        self.conv_blocks_localization = nn.ModuleList(self.decoder.conv_blocks_localization)
        self.tu = nn.ModuleList(self.decoder.tu)
        self.seg_outputs = nn.ModuleList(self.decoder.seg_outputs)
        if upscale_logits:
            self.upscale_logits_ops = nn.ModuleList(
                self.decoder.upscale_logits_ops)  # lambda x:x is not a Module so we need to distinguish here

        if weightInitializer is not None:
            self.apply(weightInitializer)
        
    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)