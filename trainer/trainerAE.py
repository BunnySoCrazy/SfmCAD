import torch
from .base import BaseTrainer
from .loss import reconLoss
from .acc_recall import acc_recall
from collections import OrderedDict
import random
from tensorboardX import SummaryWriter
import os

class TrainerAE(BaseTrainer):
    def __init__(self, specs):
        super().__init__(specs)
        self.set_optimizer(lr=specs["LearningRate"],betas=specs["betas"])
        self.train_tb = SummaryWriter(os.path.join(self.log_dir, 'train.events'))

    def build_net(self):
        arch = __import__("model." + self.specs["NetworkArch"], fromlist=["Encoder", "Decoder", "Generator"])

        self.encoder = arch.Encoder().cuda()
        decoder_kwargs = {"num_primitives": self.specs["NumPrimitives"]}
        if 'num_layers' in arch.Decoder.__dict__:
            decoder_kwargs['num_layers'] = self.specs["DecoderLayers"]
        self.decoder = arch.Decoder(**decoder_kwargs).cuda()
        self.generator = arch.Generator(num_primitives=self.specs["NumPrimitives"],
                                   segment=self.specs["Segments"],
                                   sharpness=self.specs["Sharpness"],
                                   soft_sharp=self.specs["SoftSharp"]).cuda()

    def set_optimizer(self, lr, betas):
        params_to_optimize = [
            {"params": self.encoder.parameters(), "lr": lr, "betas": (betas[0], betas[1])},
            {"params": self.decoder.parameters(), "lr": lr, "betas": (betas[0], betas[1])},
            {"params": self.generator.parameters(), "lr": lr, "betas": (betas[0], betas[1])},
        ]
        if hasattr(self.generator, 'sharpness') and isinstance(self.generator.sharpness, torch.nn.parameter.Parameter):
            params_to_optimize.append({"params": self.generator.sharpness, "lr": 1, "betas": (0.5, betas[1])})

        if hasattr(self.generator, 'soft_sharp') and isinstance(self.generator.soft_sharp, torch.nn.parameter.Parameter):
            params_to_optimize.append({"params": self.generator.soft_sharp, "lr": 1, "betas": (0.5, betas[1])})

        self.optimizer = torch.optim.Adam(params_to_optimize)

    def set_loss_function(self):
        self.loss_func = reconLoss(self.specs["LossWeightTrain"],
                                   self.specs["CBLoss"],self.specs["SMLoss"],
                                   self.specs["HWLoss"] if "HWLoss" in self.specs else None,self.specs["Strategy"]).cuda()

    def set_accuracy_function(self):
        self.acc_func = acc_recall().cuda()

    def forward(self, data, phase):
        voxels = data['voxels'].cuda()
        occ_data = data['occ_data'].cuda()
        load_point_batch_size = occ_data.shape[1]
        point_batch_size = 16*16*16

        point_batch_num = int(load_point_batch_size/point_batch_size)
        which_batch = torch.randint(point_batch_num+1, (1,))
        if which_batch == point_batch_num:
            xyz = occ_data[:,-point_batch_size:, :3]
            gt_3d_occ = occ_data[:,-point_batch_size:, 3]
        else:
            xyz = occ_data[:,which_batch*point_batch_size:(which_batch+1)*point_batch_size, :3]
            gt_3d_occ = occ_data[:,which_batch*point_batch_size:(which_batch+1)*point_batch_size, 3]

        shape_code = self.encoder(voxels)
        shape_3d = self.decoder(shape_code)
        outputs = self.generator(xyz, shape_3d, shape_code, phase=phase)

        loss_dict = self.loss_func(outputs, gt_3d_occ, voxels)
        acc_dict = self.acc_func(outputs, gt_3d_occ, voxels)

        return outputs, loss_dict, acc_dict

    def set_soft_sharp(self):
        self.generator.soft_sharp = min(5 + (self.clock.epoch // 20) * 5, 50)

    def train_func(self, data, phase='box'):
        self.encoder.train()
        self.decoder.train()
        self.generator.train()
        self.optimizer.zero_grad()

        if self.specs["SetSoftSharp"] and not isinstance(self.generator.soft_sharp, torch.nn.parameter.Parameter):
            self.set_soft_sharp()

        outputs, losses, acc_recall = self.forward(data,phase=phase)
        self.update_network(losses)

        self.updata_epoch_info(losses, acc_recall)
        if self.clock.step % 10 == 0:
            self.record_to_tb(losses, acc_recall)

        loss_info = OrderedDict({k: "{:.5f}".format(v.item()/(self.clock.minibatch+1))
                                for k, v in self.epoch_loss.items()})
        acc_info = OrderedDict({k: "{:.2f}".format(v.item()/(self.clock.minibatch+1))
                                for k, v in self.epoch_acc.items()})

        out_info = loss_info.copy()
        out_info.update(acc_info)

        if 'sharpness' in outputs:
            if isinstance(self.generator.sharpness, torch.Tensor):
                self.generator.sharpness.data.clamp_(max=250)
            sharp_info = OrderedDict({'sharp':"{:.2f}".format(outputs['sharpness'])})
            out_info.update(sharp_info)

        if 'soft_sharp' in outputs:
            if isinstance(self.generator.soft_sharp, torch.Tensor):
                self.generator.soft_sharp.data.clamp_(max=150)
            soft_info = OrderedDict({'soft':"{:.2f}".format(outputs['soft_sharp'])})
            out_info.update(soft_info)

        return outputs, out_info
