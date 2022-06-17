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
#

from typing import Tuple, Union

import torch.nn as nn
from nnunet.network_architecture.generic_UNet import Generic_UNETDecoder
from nnunet.network_architecture.neural_network import SegmentationNetwork

from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.nets import ViT

def find_patch_size(num_pool_ops):
    # smaijer
    # we know the spatial dimension is divisible by 2^X, where X is the amount of pooling operations in that dimension
    # since our ideal patch size is 16, we use patch size = 16 when the amount of pooling operations is high 4 or 5 (2^4=16 and 2^5 = 25)
    # when there are less, we choose patch size = 2^X
    # TODO for computational effiency we can also take patch size = 2^X (so 32) when there are 5 pooling ops. however for now we choose 16 for
    #        optional performance gain

    if num_pool_ops >= 4:
        return 2**4
    else:
        return 2**num_pool_ops

def proj_feat(x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

class UNETREncoder(nn.Module):

    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
            self,
            in_channels: int,
            img_size: Tuple[int, int, int],
            num_pool_per_axis: Tuple[int, int, int], #  smaijer
            feature_size: int = 16,
            hidden_size: int = 768,
            mlp_dim: int = 3072,
            num_heads: int = 12,
            pos_embed: str = "perceptron",
            norm_name: Union[Tuple, str] = "instance",
            conv_block: bool = False,
            res_block: bool = True,
            dropout_rate: float = 0.0,
            do_print: bool = False
        ):
        super(UNETREncoder, self).__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.num_layers = 12
        self.do_print = do_print

        #### smaijer
        print(f"Img size: {img_size}")
        self.patch_size = (
            find_patch_size(num_pool_per_axis[0]),
            find_patch_size(num_pool_per_axis[1]),
            find_patch_size(num_pool_per_axis[2])
        )
        print(f"Patch size: {self.patch_size}")
        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
            img_size[2] // self.patch_size[2],
        )
        print(f"Feature size: {self.feat_size}")
        #### smaijer

        self.hidden_size = hidden_size
        self.classification = False
        self.vit = ViT(
            in_channels=in_channels,  # 1
            img_size=img_size,  # 80, 192, 160
            patch_size=self.patch_size, # 10, 24, 20
            hidden_size=hidden_size, # 768
            mlp_dim=mlp_dim, # 3072
            num_layers=self.num_layers, # 12
            num_heads=num_heads, # 12
            pos_embed=pos_embed, # perceptron
            classification=self.classification, # false
            dropout_rate=dropout_rate,
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=in_channels, # 1
            out_channels=feature_size, # 16
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )

    def forward(self, x_in):
        # x = [2, 512, 768]
        # hidden_states_out 12x [2, 512, 768]
        x, hidden_states_out = self.vit(x_in)
        if self.do_print:
            print(f"x_in.shape: {x_in.shape}")
            print(f"hidden_states_out.shape: {len(hidden_states_out)}")

        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(proj_feat(x2, self.hidden_size, self.feat_size))
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(proj_feat(x3, self.hidden_size, self.feat_size))
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(proj_feat(x4, self.hidden_size, self.feat_size))
        bottleneck = proj_feat(x, self.hidden_size, self.feat_size)
        if self.do_print:
            print(f"x: {x.shape}, x2: {x2.shape}, x3: {x3.shape}, x4: {x4.shape}")
            print(f"enc1: {enc1.shape}, enc2: {enc2.shape}, enc3: {enc3.shape}, enc4: {enc4.shape}, bottleneck: {bottleneck.shape}")
        return bottleneck, [enc1, enc2, enc3, enc4]

class UNETRDecoder(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(self, hidden_size, feat_size, feature_size, num_pool_per_axis, num_pool, pool_op_kernel_sizes,
                norm_name, res_block, out_channels, deep_supervision, upscale_logits, upsample_mode, do_print=False):
        super(UNETRDecoder, self).__init__()


        self.hidden_size = hidden_size
        self.feat_size = feat_size
        self.num_pool_per_axis = num_pool_per_axis
        self.num_pool = num_pool
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.upscale_logits = upscale_logits
        self.upsample_mode = upsample_mode
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.do_print=do_print

        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)  # type: ignore

    def forward(self, input):
        dec4, [enc1, enc2, enc3, enc4] = input # dec4=bottleneck

        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2) 
        out = self.decoder2(dec1, enc1)
        if self.do_print:
            print(f"dec4: {dec4.shape}, dec3: {dec3.shape}, dec2: {dec2.shape}, dec1: {dec1.shape}, out: {out.shape}")
        
        logits = self.out(out)
        if self.do_print:
            print(f"logits: {logits.shape}")

        if self._deep_supervision and self.do_ds:
            if self.do_print:
                print("Suus 12c We doen deep supervision dingen")
            seg_outputs = [dec4, dec3, dec2, dec1, logits]
            Generic_UNETDecoder.set_upscale_logits_ops(self)
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return logits

class UNETR(SegmentationNetwork):

    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int, int],
        num_pool_per_axis: Tuple[int, int, int], # smaijer
        num_pool,
        pool_op_kernel_sizes,
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
        upscale_logits=False,
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
        super(UNETR, self).__init__()

        upsample_mode = 'trilinear' # hardcoded because we only do 3d. see generic_UNet for nice code for 2d and stuff.

        self.encoder = UNETREncoder(in_channels, img_size, num_pool_per_axis, feature_size, hidden_size, mlp_dim, num_heads, 
                                    pos_embed, norm_name, conv_block, res_block, dropout_rate, do_print)

        self.decoder = UNETRDecoder(hidden_size, self.encoder.feat_size, feature_size, num_pool_per_axis, num_pool, pool_op_kernel_sizes, norm_name, 
                                    res_block, out_channels, deep_supervision, upscale_logits, upsample_mode, do_print)

        # Necessary for nnU-net
        self.conv_op = nn.Conv3d
        self.num_classes = out_channels
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        
    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)