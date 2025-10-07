import os
import torch
from abc import abstractmethod
from utils.workspace import get_model_params_dir,get_model_params_dir_shapename


class BaseTrainer(object):
    def __init__(self, specs):
        self.specs = specs
        self.experiment_directory = specs['experiment_directory']
        self.log_dir = os.path.join(specs['experiment_directory'], 'log/')

        self.best_loss = float('inf')
        self.epoch_loss = None
        self.epoch_acc = None
        self.clock = TrainClock()

        self.build_net()
        self.set_loss_function()
        self.set_accuracy_function()

    @abstractmethod
    def build_net(self):
        raise NotImplementedError

    @abstractmethod
    def set_optimizer(self):
        raise NotImplementedError

    def load_shape_code(self):
        pass

    def set_loss_function(self):
        pass

    def set_accuracy_function(self):
        pass

    @abstractmethod
    def forward(self, data):
        raise NotImplementedError

    @abstractmethod
    def train_func(self, data):
        raise NotImplementedError
    
    def update_network(self, loss_dict):
        loss = sum(loss_dict.values())
        loss.backward()
        self.optimizer.step()

    def record_to_tb(self, loss_dict, acc_dict):
        losses_acc_values = {k: v.item() for k, v in loss_dict.items()|acc_dict.items()}
        tb = self.train_tb
        for k, v in losses_acc_values.items():
            tb.add_scalar(k, v, self.clock.step)

    def updata_epoch_info(self, loss_dict, acc_dict):
        if self.clock.minibatch == 0:
            self.epoch_loss = None
            self.epoch_acc = None

        self.epoch_loss = {key: self.epoch_loss.get(key, 0) + loss_dict[key] for key in loss_dict} \
                                                                if self.epoch_loss else loss_dict
        self.epoch_acc = {key: self.epoch_acc.get(key, 0) + acc_dict[key] for key in acc_dict} \
                                                                if self.epoch_acc else acc_dict

    def save_model_parameters(self, filename):
        model_params_dir = get_model_params_dir(self.experiment_directory)
        torch.save(
            {"epoch": self.clock.epoch,
            "encoder_state_dict": self.encoder.state_dict(),
            "decoder_state_dict": self.decoder.state_dict(),
            "generator_state_dict": self.generator.state_dict(),
            "opt_state_dict": self.optimizer.state_dict()},
            os.path.join(model_params_dir, filename),
        )

    def save_model_if_best(self, grid_sample):
        epoch_loss_value = sum(self.epoch_loss.values()).item()/(self.clock.minibatch+1)
        if epoch_loss_value < self.best_loss:
            print('saving best model')
            model_params_dir = get_model_params_dir(self.experiment_directory)
            torch.save(
                {"epoch": self.clock.epoch,
                "encoder_state_dict": self.encoder.state_dict(),
                "decoder_state_dict": self.decoder.state_dict(),
                "generator_state_dict": self.generator.state_dict(),
                "opt_state_dict": self.optimizer.state_dict()},
                os.path.join(model_params_dir, f"best_{grid_sample}.pth"),
            )
            self.best_loss = epoch_loss_value

    def save_model_parameters_per_shape(self, shapename, filename):
        model_params_dir = get_model_params_dir(self.experiment_directory)
        model_params_dir = get_model_params_dir_shapename(model_params_dir, shapename)

        torch.save(
            {"epoch": self.clock.epoch,
            "shape_code_state_dict": self.shape_code,
            "decoder_state_dict": self.decoder.state_dict(),
            "generator_state_dict": self.generator.state_dict(),
            "opt_state_dict": self.optimizer.state_dict()},
            os.path.join(model_params_dir, filename)
        )

    def load_encoder(self, filename):
        if not os.path.isfile(filename):
            raise Exception('model state dict "{}" does not exist'.format(filename))

        data = torch.load(filename)
        self.encoder.load_state_dict(data["encoder_state_dict"])
        return 0

    def load_ckpt(self, checkpoint, opt=False):
        filename = os.path.join(
            self.experiment_directory, "ModelParameters", checkpoint + ".pth"
        )

        if not os.path.isfile(filename):
            raise Exception('model state dict "{}" does not exist'.format(filename))

        data = torch.load(filename)
        self.encoder.load_state_dict(data["encoder_state_dict"])
        self.decoder.load_state_dict(data["decoder_state_dict"])

        try:
            self.generator.load_state_dict(data["generator_state_dict"],strict=False)
        except:
            print('pass generator load_state_dict')
        if opt:
            self.optimizer.load_state_dict(data["opt_state_dict"])
        return 0

    def load_model_parameters(self, checkpoint, opt=False):
        filename = os.path.join(
            self.experiment_directory, "ModelParameters", checkpoint + ".pth"
        )

        if not os.path.isfile(filename):
            raise Exception('model state dict "{}" does not exist'.format(filename))

        data = torch.load(filename)
        self.encoder.load_state_dict(data["encoder_state_dict"])
        self.decoder.load_state_dict(data["decoder_state_dict"])

        try:
            self.generator.load_state_dict(data["generator_state_dict"])
        except:
            print('pass generator load_state_dict')

        if opt:
            self.optimizer.load_state_dict(data["opt_state_dict"])
        return data["epoch"]

    def load_model_parameters_per_shape(self, shapename, checkpoint):
        filename = os.path.join(
            self.experiment_directory, "ModelParameters", shapename, checkpoint + ".pth"
        )

        if not os.path.isfile(filename):
            raise Exception('model state dict "{}" does not exist'.format(filename))

        data = torch.load(filename)

        try:
            self.optimizer.load_state_dict(data["opt_state_dict"])
        except:
            pass
        self.decoder.load_state_dict(data["decoder_state_dict"])
        try:
            self.generator.load_state_dict(data["generator_state_dict"],strict=False)
        except:
            print('pass generator load_state_dict')

        return data["epoch"], data["shape_code_state_dict"]


class TrainClock(object):
    def __init__(self):
        self.epoch = 1
        self.minibatch = 0
        self.step = 0

    def tick(self):
        self.minibatch += 1
        self.step += 1

    def tock(self):
        self.epoch += 1
        self.minibatch = 0

    def make_checkpoint(self):
        return {
            'epoch': self.epoch,
            'minibatch': self.minibatch,
            'step': self.step
        }

    def restore_checkpoint(self, clock_dict):
        self.epoch = clock_dict['epoch']
        self.minibatch = clock_dict['minibatch']
        self.step = clock_dict['step']
