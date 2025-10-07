import torch
import torch.nn as nn
import sys
sys.path.append('./')
from utils.sdfs import (sample_cubic_bezier, points_to_boxes_3d,
                        sdfBox_roatate2d, transform_points,sdfExtrusion,
                        quaternion_apply,quaternion_invert)

from utils.utils import add_latent
from .sketchHead import SketchHead
import torch.nn.functional as F
import math

class Generator(nn.Module):
    def __init__(self, num_primitives=4, sharpness=150, sharpness_IM=50, test=False, segment=8, soft_sharp=20):
        super(Generator, self).__init__()
        self.sharpness = sharpness
        self.soft_sharp = soft_sharp
        self.test = test
        self.segment = segment
        self.res = (32, 32, 3)
        D_IN = 2
        G_LATENT_SIZE = 256
        L_LATENT_SIZE = 256
        self.num_primitives = num_primitives

        self.linear_layers = nn.ModuleList([nn.Linear(G_LATENT_SIZE, L_LATENT_SIZE) for _ in range(num_primitives)])
        self.sketch_head = SketchHead(d_in=D_IN+L_LATENT_SIZE, dims=[512, 512, 512])

    def forward(self, sample_point_coordinates, primitive_parameters, code, part_idx=-1, phase='box'):
        '''
        Input
            sample_point_coordinates: [B,N,3]
            primitive_parameters: [B,P,4x3] (4 control points, 3 coords)
            code: [B, 256]
        Output
            occ3d: [B, N]
        '''
        B, N = sample_point_coordinates.shape[:2]
        B, _ = primitive_parameters.shape
        P,S = self.num_primitives,self.segment

        local_codes = [F.leaky_relu(linear_layer(code), negative_slope=0.01) for linear_layer in self.linear_layers]
        local_codes = torch.stack(local_codes, dim=-1)

        bezier_para = primitive_parameters[...,:self.num_primitives*(4*3)].view(-1, self.num_primitives, 4, 3)
        outputs = {}

        hw = torch.abs(primitive_parameters[...,self.num_primitives*(4*3):self.num_primitives*(4*3+2)].view(-1, self.num_primitives, 2))
        outputs['hw'] = hw

        top_dirs = primitive_parameters[..., self.num_primitives*(4*3+2):self.num_primitives*(4*3+3)].view(-1, self.num_primitives, 1) + 1
        bottom_dirs = primitive_parameters[..., self.num_primitives*(4*3+3):].view(-1, self.num_primitives, 1) + 1

        top_dirs *= math.pi
        bottom_dirs *= math.pi

        interp = torch.linspace(0, 1, steps=self.segment).to(top_dirs.device)
        interp = interp.view(1, 1, self.segment).expand(B, P, self.segment)

        up_dirs = (1 - interp) * top_dirs + interp * bottom_dirs
        up_dirs = up_dirs.unsqueeze(3).expand(B,P,self.segment,1)
        up_dirs = up_dirs.reshape(B,P*self.segment,1)

        t_values = torch.linspace(0, 1, self.segment+1).to(primitive_parameters.device)
        curve_points = sample_cubic_bezier(bezier_para, t_values)

        hw = hw.unsqueeze(2).expand(B,P,self.segment,2)
        scale = hw/2
        scale = scale.unsqueeze(1)
        hw_ = hw.clone()
        hw_ = hw_.reshape(B,P*self.segment,2)

        boxes_params = points_to_boxes_3d(curve_points, up_dirs)
        boxes = boxes_params.view(B,P*self.segment,3+4+1)

        lwh = torch.cat((hw_/2, boxes[..., 7].unsqueeze(-1)*1.2),dim=-1)
        boxes_sdf = sdfBox_roatate2d(boxes[..., :4], boxes[..., 4:7], lwh, sample_point_coordinates, up_dirs).squeeze(-1)
        boxes_occ = torch.tanh(-1*boxes_sdf*self.sharpness)/2+.5

        if phase == 'IM_sample_intime':
            sample_3dpoint_coord_occ = self.sample_points_within_boxes(boxes_params,local_codes, lwh.view(B,P,S,-1),res=self.res, up_dir=up_dirs)
            sample_3dpoint_coord_occ = sample_3dpoint_coord_occ.reshape(B,-1,4)
            outputs["sample_3d_occ"] = sample_3dpoint_coord_occ
            occ = boxes_occ
        elif phase == 'IM':
            bezier_sdf = self.bezier_SDF(boxes, code, local_codes, sample_point_coordinates, scale=None, angle=up_dirs)
            bezier_occ = torch.tanh(-1*bezier_sdf*self.sharpness)/2+.5
            occ = boxes_occ*bezier_occ
        else:
            occ = boxes_occ

        union_occ,_ = torch.max(occ, dim=-1)

        outputs["output_3d_occ"]= union_occ
        outputs["sharpness"]= self.sharpness.item() if isinstance(self.sharpness, torch.Tensor) else self.sharpness
        outputs["soft_sharp"]= self.soft_sharp.item() if isinstance(self.soft_sharp, torch.Tensor) else self.soft_sharp
        outputs["bezier_para"]= bezier_para
        outputs["local_codes"]= local_codes

        return outputs

    def get_boxes_params(self, primitive_parameters):
        B = primitive_parameters.shape[0]
        P = self.num_primitives

        bezier_para = primitive_parameters[...,:self.num_primitives*(4*3)].view(-1, self.num_primitives, 4, 3)
        up_dirs = primitive_parameters[...,self.num_primitives*(4*3+2):].view(-1, self.num_primitives, 1) + 1
        up_dirs *= math.pi
        up_dirs = up_dirs.unsqueeze(2).expand(B,P,self.segment,1)
        up_dirs = up_dirs.reshape(B,P*self.segment,1)
        hw = torch.abs(primitive_parameters[...,self.num_primitives*(4*3):self.num_primitives*(4*3+2)].view(-1, self.num_primitives, 2))

        t_values = torch.linspace(0, 1, self.segment+1).to(primitive_parameters.device)
        curve_points = sample_cubic_bezier(bezier_para, t_values)

        hw = hw.unsqueeze(2).expand(B,P,self.segment,2)
        scale = hw/2
        scale = scale.unsqueeze(1)
        hw_ = hw.clone()
        hw_ = hw_.reshape(B,P*self.segment,2)

        boxes_params = points_to_boxes_3d(curve_points, up_dirs)

        return boxes_params

    def bezier_SDF(self, boxes, code, local_codes, sample_point_coordinates, scale, angle):
        P = self.num_primitives
        B, N = sample_point_coordinates.shape[:2]

        transformed_points = transform_points(boxes[..., :4], boxes[..., 4:7], sample_point_coordinates)

        cos_angle = torch.cos(angle).squeeze(-1)
        sin_angle = torch.sin(angle).squeeze(-1)

        x = transformed_points[..., 0]
        y = transformed_points[..., 1]
        z = transformed_points[..., 2]

        rotated_x = x * cos_angle.unsqueeze(1) - y * sin_angle.unsqueeze(1)
        rotated_y = x * sin_angle.unsqueeze(1) + y * cos_angle.unsqueeze(1)

        transformed_points = torch.stack([rotated_x, rotated_y, z], dim=-1)

        latent_points = []
        for i in range(self.num_primitives):
            input_points = transformed_points[..., i*self.segment:(i+1)*self.segment, :2]/scale[..., i, :, :] \
                    if scale is not None  else transformed_points[..., i*self.segment:(i+1)*self.segment, :2]

            input_points = input_points.reshape(B, N*self.segment, 2)
            input_code = local_codes[...,i]

            latent_point = add_latent(input_points, input_code).float()
            latent_points.append(latent_point)

        sdfs_2d = [self.sketch_head(latent_points[i]).reshape(B, N, self.segment) for i in range(self.num_primitives)]
        sdfs_2d = torch.cat(sdfs_2d, dim=-1)

        box_ext = sdfExtrusion(sdfs_2d, boxes[..., 7]*1.2, transformed_points).squeeze(-1)

        return box_ext

    def sketch_SDFs(self, boxes, code, local_codes, res=(8, 8, 3)):
        B = code.shape[0]
        P = self.num_primitives
        x_linspace = torch.linspace(-1, 1, res[0], device=boxes.device)
        y_linspace = torch.linspace(-1, 1, res[1], device=boxes.device)

        x_grid, y_grid, = torch.meshgrid(x_linspace, y_linspace)
        grid_shape = [B] + list(res[:2])
        x_grid = x_grid.expand(*grid_shape)
        y_grid = y_grid.expand(*grid_shape)
        sample_2dpoint_coord = torch.stack([x_grid, y_grid], dim=-1)

        latent_points = []
        for i in range(self.num_primitives):
            input_points = sample_2dpoint_coord.view(B,-1,2)
            input_code = local_codes[...,i]

            latent_point = add_latent(input_points, input_code).float()
            latent_points.append(latent_point)

        sdfs_2d = [self.sketch_head(latent_points[i]) for i in range(self.num_primitives)]
        sdfs_2d = torch.cat(sdfs_2d, dim=-1)
        return sdfs_2d

    def sample_points_within_boxes(self, boxes, local_codes, hw, res=(12, 12, 5), up_dir=None):
        B, P, S, _ = boxes.shape
        angle, translation, h = torch.split(boxes, [4, 3, 1], dim=-1)

        w, d = torch.split(hw[...,:2], [1,1], dim=-1)

        x_linspace = torch.linspace(-0.5, 0.5, res[0], device=boxes.device)
        y_linspace = torch.linspace(-0.5, 0.5, res[1], device=boxes.device)
        z_linspace = torch.linspace(-1, 1, res[2], device=boxes.device)

        x_grid_base, y_grid_base, z_grid_base = torch.meshgrid(x_linspace, y_linspace, z_linspace)

        sample_2dpoint_coord = torch.stack([x_grid_base, y_grid_base], dim=-1)
        sample_2dpoint_coord = sample_2dpoint_coord.unsqueeze(0).repeat(B,1,1,1,1)
        sample_2dpoint_coord = sample_2dpoint_coord[...,0,:]

        latent_points = []
        for i in range(P):
            input_points = sample_2dpoint_coord.view(B,-1,2)
            input_code = local_codes[...,i]
            latent_point = add_latent(input_points, input_code).float()
            latent_points.append(latent_point)

        sdfs_2d = [self.sketch_head(latent_points[i]) for i in range(self.num_primitives)]
        sdfs_2d = torch.cat(sdfs_2d, dim=-1)
        sdfs_2d = torch.tanh(-1*sdfs_2d*self.sharpness/4)/2+.5

        sdfs_2d = sdfs_2d.reshape(B,res[0],res[1],P)
        sdfs_2d = sdfs_2d.unsqueeze(-2).repeat(1,1,1,res[2],1)
        sdfs_2d = sdfs_2d.permute(0,4,1,2,3)
        sdfs_2d = sdfs_2d.unsqueeze(2).repeat(1,1,S,1,1,1)
        sdfs_2d = sdfs_2d.unsqueeze(-1)

        grid_shape = [B, P, S] + list(res)

        x_grid_base = x_grid_base.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        y_grid_base = y_grid_base.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        z_grid_base = z_grid_base.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        x_grid_base = x_grid_base.expand(*grid_shape)
        y_grid_base = y_grid_base.expand(*grid_shape)
        z_grid_base = z_grid_base.expand(*grid_shape)

        x_grid = x_grid_base * w.unsqueeze(-1).unsqueeze(-1) * 2
        y_grid = y_grid_base * d.unsqueeze(-1).unsqueeze(-1) * 2
        z_grid = z_grid_base * h.unsqueeze(-1).unsqueeze(-1)

        samples = torch.stack([x_grid, y_grid, z_grid], dim=-1)

        samples_rotated = quaternion_apply(quaternion_invert(angle.unsqueeze(-2).unsqueeze(-2).unsqueeze(-2)), samples)
        samples_translated = samples_rotated + translation.unsqueeze(-2).unsqueeze(-2).unsqueeze(-2)

        return torch.cat((samples_translated, sdfs_2d),dim=-1)

    def get_profiles(self, local_codes, res=(100, 100)):
        B = local_codes.shape[0]
        P = self.num_primitives
        x_linspace = torch.linspace(-1, 1, res[0], device=local_codes.device)
        y_linspace = torch.linspace(-1, 1, res[1], device=local_codes.device)

        x_grid, y_grid, = torch.meshgrid(x_linspace, y_linspace)
        grid_shape = [B] + list(res[:2])
        x_grid = x_grid.expand(*grid_shape)
        y_grid = y_grid.expand(*grid_shape)
        sample_2dpoint_coord = torch.stack([x_grid, y_grid], dim=-1)

        latent_points = []
        for i in range(self.num_primitives):
            input_points = sample_2dpoint_coord.view(B,-1,2)
            input_code = local_codes[...,i]

            latent_point = add_latent(input_points, input_code).float()
            latent_points.append(latent_point)

        sdfs_2d = [self.sketch_head(latent_points[i]) for i in range(self.num_primitives)]
        sdfs_2d = torch.cat(sdfs_2d, dim=-1)
        return sdfs_2d

if __name__ =="__main__":
    B = 3
    N = 1000
    P = 6
    res = 64
    generator = Generator(num_primitives=P)

    input = torch.randn(B,1,res,res,res)
    xyz = torch.randn(B,N,3)
    z = torch.randn(B,256)
    decode = torch.randn(B,P*(4*3+2))

    output_3d_occ = generator(xyz, decode, z)
    print('output_3d_occ', output_3d_occ)
