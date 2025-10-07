import argparse
import os
import utils
from utils.workspace import load_experiment_specifications
from trainer import FineTunerAE
from dataset import dataloader
import torch
import torch.nn.functional as F

def main(args):
	experiment_directory = os.path.join('./exp_log', args.experiment_directory)
	specs = load_experiment_specifications(experiment_directory)
	occ_dataset = dataloader.BSP_GTSamples(specs["DataSource"], test_flag=True)	
	reconstruction_dir = os.path.join(experiment_directory, "Reconstructions")
	
	if not os.path.isdir(reconstruction_dir):
		os.makedirs(reconstruction_dir)
  
	shape_indexes = list(range(int(args.start), int(args.end)))
	print('Shape indexes all: ', shape_indexes)
	specs["experiment_directory"] = experiment_directory
	ft_agent = FineTunerAE(specs)
	
	for index in shape_indexes:
		shapename = occ_dataset.data_names[index]
  
		try:
			shape_code, shape_3d = ft_agent.evaluate(shapename, args.checkpoint)
		except:
			continue
  
		if args.sk:
			sk_dir = os.path.join(reconstruction_dir, 'sk/')
			if not os.path.isdir(sk_dir):
				os.makedirs(sk_dir)
			sk_filepath = os.path.join(sk_dir, shapename)
			local_codes = [F.leaky_relu(linear_layer(shape_code.cuda()), negative_slope=0.01) for linear_layer in ft_agent.generator.linear_layers]
			local_codes = torch.stack(local_codes, dim=-1)
			for part_idx in range(16):
				sk_filepath = os.path.join(sk_dir, shapename+str(part_idx))
				sketch_list = utils.get_sketch_list(ft_agent.generator, local_codes[...,part_idx].cuda(), 500, shape_3d, part_idx)
				utils.create_frenet_sketch(shape_3d, sk_filepath, sketch_list, part_idx)
		
		elif args.each_part:
			MC_dir = os.path.join(reconstruction_dir, 'MC/')
			if "Shape_Part" in specs["DataSource"]:	
				mesh_filename = os.path.join(MC_dir, shapename.split('_')[0],shapename.split('_')[1])
				if not os.path.isdir(os.path.dirname(mesh_filename)):
					os.makedirs(os.path.dirname(mesh_filename))
			else:
				mesh_filename = os.path.join(MC_dir, shapename)
				if not os.path.isdir(MC_dir):
					os.makedirs(MC_dir)
			for part_idx in range(16):
				utils.create_mesh_mc(
					ft_agent.generator, shape_3d.cuda(), shape_code.cuda(), mesh_filename+str(part_idx), max_batch=16**3, N=int(args.grid_sample), 
					threshold=float(args.mc_threshold),
					part_idx = part_idx
				)
		elif args.bezier_curve:
			curve_dir = os.path.join(reconstruction_dir, 'Curve/')
			if "Shape_Part" in specs["DataSource"]:
				curve_filename = os.path.join(curve_dir, shapename.split('_')[0],shapename.split('_')[1])
				if not os.path.isdir(os.path.dirname(curve_filename)):
					os.makedirs(os.path.dirname(curve_filename))
			else:
				curve_filename = os.path.join(curve_dir, shapename)
				if not os.path.isdir(curve_dir):
					os.makedirs(curve_dir)
			utils.create_curves(shape_3d, curve_filename, Part_mode="Shape_Part" in specs["DataSource"])
		elif args.frenet_boxes:
			box_dir = os.path.join(reconstruction_dir, 'Boxes/')
			box_filename = os.path.join(box_dir, shapename.split('_')[0],shapename.split('_')[1])
			if not os.path.isdir(os.path.dirname(box_filename)):
				os.makedirs(os.path.dirname(box_filename))
			utils.create_frenet_boxes(shape_3d, box_filename)
		else:
			MC_dir = os.path.join(reconstruction_dir, 'MC/')
			if "Shape_Part" in specs["DataSource"]:	
				mesh_filename = os.path.join(MC_dir, shapename.split('_')[0],shapename.split('_')[1])
				if not os.path.isdir(os.path.dirname(mesh_filename)):
					os.makedirs(os.path.dirname(mesh_filename))
			else:
				mesh_filename = os.path.join(MC_dir, shapename)
				if not os.path.isdir(MC_dir):
					os.makedirs(MC_dir)
			utils.create_mesh_mc(
			ft_agent.generator, shape_3d.cuda(), shape_code.cuda(), mesh_filename, max_batch=16**3, N=int(args.grid_sample),
			threshold=float(args.mc_threshold),phase='IM'
			)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="test trained model")
    arg_parser.add_argument("--experiment", "-e", dest="experiment_directory", required=True)
    arg_parser.add_argument("--checkpoint", "-c", dest="checkpoint", default="latest")
    arg_parser.add_argument("--start", dest="start", default=0)
    arg_parser.add_argument("--end", dest="end", default=1)
    arg_parser.add_argument("--mc_threshold", dest="mc_threshold", default=0.9)
    arg_parser.add_argument("--gpu", "-g", dest="gpu", required=True)
    arg_parser.add_argument("--grid_sample", dest="grid_sample", default=128)
    arg_parser.add_argument("--each_part", dest="each_part", default=False)
    arg_parser.add_argument('--test_data', dest='test_data', default=False, action='store_true')
    arg_parser.add_argument('--bezier_curve', dest='bezier_curve', default=False, action='store_true')
    arg_parser.add_argument('--frenet_boxes', dest='frenet_boxes', default=False, action='store_true')
    arg_parser.add_argument('--sk', dest='sk', default=False, action='store_true')

    args = arg_parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d"%int(args.gpu)

    main(args)