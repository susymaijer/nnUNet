# Code below is mostly the UNETR code. 
# 
# COPYRIGHT UNETR
# https://github.com/Project-MONAI/research-contributions/blob/main/UNETR/BTCV/networks/unetr.py 
# UNETR based on: "Hatamizadeh et al.
# UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"

from typing import Tuple, Union

from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.nets import ViT

from nnunet.network_architecture.generic_UNet import Generic_UNETDecoder
from nnunet.network_architecture.neural_network import SegmentationNetwork
import numpy as np

import torch.nn as nn

def find_patch_size(num_pool_ops, net_conv_kernel_sizes, dim):
    '''
        thesis function by smaijer
    '''
    min_conv = np.min(np.array(net_conv_kernel_sizes), axis=(0))[dim]

    # in this case the encoder needs to do less convolutions in this dimension. 
    # this is not possible in the current architecture (transformer.encoder3 has 3 blocks) 
    # so I fix this by taking a smaller patch size in this dimension (= bigger feature size)
    # such that the resolution gets reduced as normal but we end up with a resolution corresponding 
    # to when we would've convoluted 1 times less
    if min_conv == 1 and num_pool_ops >= 4: # NB only when 4 cause we cap pool ops at 4 (see next line)
        num_pool_ops -= 1
    # we know the spatial dimension is divisible by 2^X, where X is the amount of pooling operations in that dimension
    # since our ideal patch size is 16, we use patch size = 16 when the amount of pooling operations is high 4 or 5 (2^4=16 and 2^5 = 25)
    # when there are less, we choose patch size = 2^X
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
    See https://github.com/Project-MONAI/research-contributions/blob/main/UNETR/BTCV/networks/unetr.py 
    """

    def __init__(
            self,
            in_channels: int,
            img_size: Tuple[int, int, int],
            num_pool_per_axis: Tuple[int, int, int], #  smaijer
            conv_kernel_sizes,
            feature_size: int = 16,
            hidden_size: int = 768,
            mlp_dim: int = 3072,
            num_heads: int = 12,
            pos_embed: str = "perceptron",
            norm_name: Union[Tuple, str] = "instance",
            conv_block: bool = False,
            res_block: bool = True,
            dropout_rate: float = 0.0
        ):
        super(UNETREncoder, self).__init__()

        # START unetr code
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.num_layers = 12
         # END unetr code

        # START thesis code smaijer
        print(f"Img size: {img_size}")
        self.patch_size = (
            find_patch_size(num_pool_per_axis[0], conv_kernel_sizes, 0),
            find_patch_size(num_pool_per_axis[1], conv_kernel_sizes, 1),
            find_patch_size(num_pool_per_axis[2], conv_kernel_sizes, 2)
        )
        print(f"Patch size: {self.patch_size}")
        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
            img_size[2] // self.patch_size[2],
        )
        print(f"Feature size: {self.feat_size}")
        self.max_num_pool = max(num_pool_per_axis)
        # END thesis code smaijer

        # START unetr code
        self.hidden_size = hidden_size
        self.classification = False
        self.vit = ViT(
            in_channels=in_channels,  
            img_size=img_size,  
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim, 
            num_layers=self.num_layers, 
            num_heads=num_heads, 
            pos_embed=pos_embed, 
            classification=self.classification, 
            dropout_rate=dropout_rate,
        )
        if self.max_num_pool >= 1:
            self.encoder1 = UnetrBasicBlock(
                spatial_dims=3,
                in_channels=in_channels, 
                out_channels=feature_size, 
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
            )
        if self.max_num_pool >= 2:
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
        if self.max_num_pool >= 3:
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
        if self.max_num_pool >= 4:
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
        # END unetr code

    def forward(self, x_in):
        # START adapted unetr code / thesis smaijer code
        x, hidden_states_out = self.vit(x_in)

        if self.max_num_pool >= 1:
            enc1 = self.encoder1(x_in)
        
        if self.max_num_pool >= 2:
            x2 = hidden_states_out[3]
            enc2 = self.encoder2(proj_feat(x2, self.hidden_size, self.feat_size))
        
        if self.max_num_pool >= 3:
            x3 = hidden_states_out[6]
            enc3 = self.encoder3(proj_feat(x3, self.hidden_size, self.feat_size))
        
        if self.max_num_pool >= 4:
            x4 = hidden_states_out[9]
            enc4 = self.encoder4(proj_feat(x4, self.hidden_size, self.feat_size))
        
        bottleneck = proj_feat(x, self.hidden_size, self.feat_size)
        # END adapted unetr code / thesis smaijer code
        return bottleneck, [enc1, enc2, enc3, enc4]

class UNETRDecoder(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(self, hidden_size, feat_size, feature_size, num_pool_per_axis, num_pool, pool_op_kernel_sizes,
                norm_name, res_block, out_channels, deep_supervision, upscale_logits, upsample_mode):
        super(UNETRDecoder, self).__init__()

        # START thesis smaijer code (necessary for nnU-Net)
        self.hidden_size = hidden_size
        self.feat_size = feat_size
        self.num_pool_per_axis = num_pool_per_axis
        self.num_pool = num_pool
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.upscale_logits = upscale_logits
        self.upsample_mode = upsample_mode
        self._deep_supervision = deep_supervision
        # END thesis smaijer code (necessary for nnU-Net)

        # START unetr code
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
        # END unetr code

    def forward(self, input):
        # START mix nnU-net and unetr code 
        dec4, [enc1, enc2, enc3, enc4] = input # dec4=bottleneck

        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2) 
        out = self.decoder2(dec1, enc1)
        logits = self.out(out)
        if self._deep_supervision and self.do_ds:
            seg_outputs = [dec4, dec3, dec2, dec1, logits]
            Generic_UNETDecoder.set_upscale_logits_ops(self)
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return logits

        # END mix nnU-net and unetr code

class UNETR(SegmentationNetwork):

    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    See https://github.com/Project-MONAI/research-contributions/blob/main/UNETR/BTCV/networks/unetr.py 
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int, int],
        num_pool_per_axis: Tuple[int, int, int], # START thesis smaijer 
        num_pool,
        pool_op_kernel_sizes,
        conv_op_kernel_sizes,
        conv_op,                                
        deep_supervision=True,
        upscale_logits=False,                     # END thesis smaijer 
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = True,
        dropout_rate: float = 0.0
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

        # START thesis smaijer code (necessary nnU-Net)
        if conv_op == nn.Conv2d:
            upsample_mode = 'bilinear'
        elif conv_op == nn.Conv3d:
            upsample_mode = 'trilinear'
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))
        # END thesis smaijer code (necessary nnU-Net)

        self.encoder = UNETREncoder(in_channels, img_size, num_pool_per_axis, conv_op_kernel_sizes, feature_size, hidden_size, 
                                    mlp_dim, num_heads, pos_embed, norm_name, conv_block, res_block, dropout_rate)

        self.decoder = UNETRDecoder(hidden_size, self.encoder.feat_size, feature_size, num_pool_per_axis, num_pool, pool_op_kernel_sizes, norm_name, 
                                    res_block, out_channels, deep_supervision, upscale_logits, upsample_mode)

        # START thesis smaijer code (necessary nnU-Net)
        self.conv_op = conv_op
        self.num_classes = out_channels
        self._deep_supervision = deep_supervision
        self.set_do_ds(deep_supervision)
        # END thesis smaijer code (necessary nnU-Net)

    def set_do_ds(self, do_ds):
        '''
            thesis smaijer function (necessary nnU-Net)
        '''
        self.do_ds = do_ds 
        self.decoder.do_ds = do_ds
        
    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)