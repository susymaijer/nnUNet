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
from nnunet.network_architecture.unetr.unetr import UNETR
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn

from time import time

class nnUNetTrainerV2_UNETR(nnUNetTrainerV2):

    def initialize_network(self):
        self.print_to_log_file("UNETR initialising network")
        self.network = UNETR(self.num_input_channels, self.num_classes, self.patch_size)
        if torch.cuda.is_available():
            self.network.cuda()

    def load_checkpoint(self, fname, train=True):
        self.print_to_log_file("UNETR loading checkpoint", fname, "train=", train)
        if not self.was_initialized:
            self.initialize(train)
        # saved_model = torch.load(fname, map_location=torch.device('cuda', torch.cuda.current_device()))
        saved_model = torch.load(fname, map_location=torch.device('cpu'))
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in saved_model["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        self.load_checkpoint_ram(new_state_dict, train)