import os
import torch
from collections import OrderedDict
from .base import BaseTrainer
from .loss import reconLoss
from .acc_recall import acc_recall
from utils.workspace import get_model_params_dir,get_model_params_dir_shapename

class FineTunerAE(BaseTrainer):
    def __init__(self, specs):
        super().__init__(specs)
        self.set_optimizer(lr=specs["ft_LearningRate"],betas=specs["ft_betas"])

    def build_net(self):
        arch = __import__("model." + self.specs["NetworkArch"], fromlist=["Encoder", "Decoder", "Generator"])

        self.encoder = arch.Encoder().cuda()
        decoder_kwargs = {"num_primitives": self.specs["NumPrimitives"]}
        if 'drop_rate' in arch.Decoder.__dict__:
            decoder_kwargs['drop_rate'] = 0
        if 'num_layers' in arch.Decoder.__dict__:
            decoder_kwargs['num_layers'] = self.specs["DecoderLayers"]
        self.decoder = arch.Decoder(**decoder_kwargs).cuda()
        self.generator = arch.Generator(num_primitives=self.specs["NumPrimitives"],
                                   segment=self.specs["Segments"],
                                   sharpness=self.specs["ft_Sharpness"],
                                   sharpness_IM=self.specs["ft_Sharpness_IM"] if "ft_Sharpness_IM" in self.specs else 50,
                                   soft_sharp=self.specs["ft_SoftSharp"]).cuda()

    def set_optimizer(self, lr, betas):
        params_to_optimize = [
            {"params": self.decoder.parameters(), "lr": lr, "betas": (betas[0], betas[1])},
            {"params": self.generator.parameters(), "lr": lr, "betas": (betas[0], betas[1])}
        ]
        if hasattr(self.generator, 'sharpness') and isinstance(self.generator.sharpness, torch.nn.Parameter):
            params_to_optimize.append({"params": self.generator.sharpness, "lr": 0.01, "betas": (0.5, betas[1])})

        if hasattr(self.generator, 'soft_sharp') and isinstance(self.generator.soft_sharp, torch.nn.Parameter):
            params_to_optimize.append({"params": self.generator.soft_sharp, "lr": 0.01, "betas": (0.5, betas[1])})

        self.optimizer = torch.optim.Adam(params_to_optimize)

    def set_loss_function(self):
        self.loss_func = reconLoss(self.specs["LossWeightTrain"],
                                   self.specs["CBLoss"],self.specs["SMLoss"],
                                   self.specs["HWLoss"] if "HWLoss" in self.specs else None,self.specs["Strategy"]).cuda()

    def set_accuracy_function(self):
        self.acc_func = acc_recall().cuda()

    def save_model_if_best_per_shape(self, shapename, grid_sample):
        epoch_loss_value = sum(self.epoch_loss.values()).item()/(self.clock.minibatch+1)
        if epoch_loss_value < self.best_loss:
            model_params_dir = get_model_params_dir(self.experiment_directory)
            model_params_dir = get_model_params_dir_shapename(model_params_dir, shapename)
            torch.save(
                {"epoch": self.clock.epoch,
                "shape_code_state_dict": self.shape_code,
                "decoder_state_dict": self.decoder.state_dict(),
                "generator_state_dict": self.generator.state_dict(),
                "opt_state_dict": self.optimizer.state_dict()},
                os.path.join(model_params_dir,  f"best_{grid_sample}.pth")
            )
            self.best_loss = epoch_loss_value

    def forward(self, data, voxels=None, phase='box'):
        load_point_batch_size = data.shape[1]
        point_batch_size = 16*16*16

        point_batch_num = int(load_point_batch_size/point_batch_size)
        which_batch = torch.randint(point_batch_num+1, (1,))
        if which_batch == point_batch_num:
            xyz = data[:,-point_batch_size:, :3]
            gt_3d_occ = data[:,-point_batch_size:, 3]
        else:
            xyz = data[:,which_batch*point_batch_size:(which_batch+1)*point_batch_size, :3]
            gt_3d_occ = data[:,which_batch*point_batch_size:(which_batch+1)*point_batch_size, 3]

        shape_3d = self.decoder(self.shape_code.cuda())

        outputs = self.generator(xyz, shape_3d, self.shape_code.cuda(), phase=phase)

        loss_dict = self.loss_func(outputs, gt_3d_occ, voxels)
        acc_dict = self.acc_func(outputs, gt_3d_occ, voxels)

        return outputs, loss_dict, acc_dict

    def load_shape_code(self, phase, voxels, shapename, checkpoint, grid_sample=64, load_ckp_para_per_shape=False):
        print('Phase , ', phase)
        if not load_ckp_para_per_shape:
            continue_from = checkpoint
            print('Continuing from "{}"'.format(continue_from))
            model_epoch = super().load_model_parameters(continue_from, opt=False)
            shape_code = self.encoder(voxels)
            shape_code = shape_code.detach().cpu().numpy()

            shape_code = torch.from_numpy(shape_code)
            print('shape_code loaded, ', shape_code.shape)
            start_epoch = model_epoch +1
        else:
            continue_from = f"best_{int(grid_sample/2)}"
            continue_from = checkpoint

            print('Continuing from "{}"'.format(continue_from))
            model_epoch, shape_code = super().load_model_parameters_per_shape(
                shapename, continue_from
            )
            print('shape_code loaded, ', shape_code.shape)
            start_epoch = model_epoch + 1

        self.shape_code = shape_code

        self.optimizer_code = torch.optim.Adam(
            [
                {
                    "params": self.shape_code,
                    "lr": self.specs["ft_LearningRate"],
                    "betas": (0.5, 0.999),
                },
            ]
        )
        print("Starting from epoch {}".format(start_epoch))
        return start_epoch

    def train_func(self, data,voxels=None,phase='box'):
        self.decoder.train()
        self.generator.train()
        self.optimizer.zero_grad()
        self.optimizer_code.zero_grad()

        outputs, losses, acc_recall = self.forward(data, voxels, phase)
        self.update_network(losses)
        self.updata_epoch_info(losses, acc_recall)

        loss_info = OrderedDict({k: "{:.3f}".format(v.item()/(self.clock.minibatch+1))
                                for k, v in self.epoch_loss.items()})
        acc_info = OrderedDict({k: "{:.2f}".format(v.item()/(self.clock.minibatch+1))
                                for k, v in self.epoch_acc.items()})

        out_info = loss_info.copy()
        out_info.update(acc_info)

        if 'sharpness' in outputs:
            sharp_info = OrderedDict({'sharp':"{:.2f}".format(outputs['sharpness'])})
            out_info.update(sharp_info)

        return outputs, out_info

    def update_network(self, loss_dict):
        loss = sum(loss_dict.values())
        loss.backward()
        self.optimizer.step()
        self.optimizer_code.step()

    def evaluate(self, shapename, checkpoint):
        saved_model_epoch, shape_code = super().load_model_parameters_per_shape(
                shapename, checkpoint,
            )

        print('Loaded epoch: %d'%(saved_model_epoch))
        self.decoder.eval()
        self.generator.eval()

        shape_3d = self.decoder(shape_code.cuda())
        return shape_code, shape_3d
