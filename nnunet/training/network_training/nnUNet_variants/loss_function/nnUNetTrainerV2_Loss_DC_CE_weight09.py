from nnunet.training.network_training.nnUNet_variants.loss_function.nnUNetTrainerV2_Loss_DC_CE_weight01 import nnUNetTrainerV2_Loss_DC_CE_weight01

class nnUNetTrainerV2_Loss_DC_CE_weight09(nnUNetTrainerV2_Loss_DC_CE_weight01):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.lower_weight = 0.9