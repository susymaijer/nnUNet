from nnunet.training.network_training.nnUNetTrainerV2_Hybrid2LR import nnUNetTrainerV2_Hybrid2LR

class nnUNetTrainerV2_Hybrid2LR_1500(nnUNetTrainerV2_Hybrid2LR):

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 1500