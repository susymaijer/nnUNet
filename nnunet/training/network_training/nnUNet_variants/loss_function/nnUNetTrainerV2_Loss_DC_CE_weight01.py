import torch
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.training.loss_functions.dice_loss import weighted_DC_and_CE_loss


class nnUNetTrainerV2_Loss_DC_CE_weight01(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.lower_weight = 0.1

    def process_plans(self, plans):
        super().process_plans(plans)
        weights = torch.empty(self.num_classes).fill_(self.lower_weight)
        weights[1] = 1 # pancreas label is always 1
        if torch.cuda.is_available():
            weights = weights.to(device="cuda:0")
        self.loss = weighted_DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {}, label_weights = weights)