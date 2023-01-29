# COPYRIGHT NOTICE THESIS SMAIJER
# Adapted from standard nnU-Net nnUnetTrainerV2 class 

import torch
from nnunet.training.network_training.nnUNetTrainerV2_Hybrid import nnUNetTrainerV2_Hybrid

class nnUNetTrainerV2_Hybrid2(nnUNetTrainerV2_Hybrid):

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 500
        self.initial_lr = 1e-4
        self.weight_decay = 1e-5

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay)
        self.lr_scheduler = None