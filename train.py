import os
import argparse
import torch.utils.data as data_utils
from tqdm import tqdm

from utils import init_seeds
from utils.workspace import load_experiment_specifications
from dataset import dataloader
from trainer import TrainerAE

def main(args):
    init_seeds(42)
    experiment_directory = os.path.join('./exp_log', args.experiment_directory)
    specs = load_experiment_specifications(experiment_directory)

    occ_dataset = dataloader.BSP_GTSamples(specs["DataSource"], test_flag=args.test_data, grid_sample=args.grid_sample)

    data_loader = data_utils.DataLoader(
        occ_dataset,
        batch_size=specs["BatchSize"],
        shuffle=True,
        num_workers=4
    )

    specs["experiment_directory"] = experiment_directory
    tr_agent = TrainerAE(specs)

    if specs["Continue"]:
        tr_agent.load_ckpt(specs["Checkpoint"], opt=False)

    if args.cont:
        tr_agent.load_ckpt(args.ckpt, opt=False)

    clock = tr_agent.clock

    if int(args.epoch) > 0:
        specs["NumEpochs"] = int(args.epoch)

    for epoch in range(specs["NumEpochs"]):
        pbar = tqdm(data_loader)
        for b, data in enumerate(pbar):
            outputs, out_info = tr_agent.train_func(data, phase=args.phase)
            pbar.set_description("[{}][{}][{}]".format(args.experiment_directory, epoch, b))
            pbar.set_postfix(out_info)
            clock.tick()

        if epoch % specs["SaveFrequency"] == 0:
            tr_agent.save_model_parameters(f"{epoch}.pth")
        tr_agent.save_model_if_best(args.grid_sample)

        clock.tock()

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--experiment", "-e", dest="experiment_directory", required=True)
    arg_parser.add_argument("--gpu", "-g", dest="gpu", default=0)
    arg_parser.add_argument("--grid", dest="grid_sample", default=64)
    arg_parser.add_argument("--ph", dest="phase", default='box')
    arg_parser.add_argument("--epoch", dest="epoch", default=-1)
    arg_parser.add_argument('--test_data', dest='test_data', default=False, action='store_true')
    arg_parser.add_argument('--continue', dest='cont', default=False, action='store_true')
    arg_parser.add_argument('--ckpt', type=str, default='best', required=False)

    args = arg_parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d"%int(args.gpu)

    main(args)