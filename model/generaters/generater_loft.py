import torch
import torch.nn as nn
import sys
sys.path.append('./')
from utils.sdfs import (sample_cubic_bezier, points_to_boxes_3d,
                        occ_between_2planes, transform_points,sdfLoft)
from utils.utils import add_latent_
from .sketchHead import SketchHead

class Generator(nn.Module):
    def __init__(self, num_primitives=4, sharpness=150, sharpness_IM=50, test=False, segment=8, soft_sharp=20):
        super(Generator, self).__init__()
        self.sharpness = sharpness
        self.soft_sharp = soft_sharp
        self.test = test
        self.segment = segment
        self.res = (24, 24, 20)
        D_IN = 2
        G_LATENT_SIZE = 256
        L_LATENT_SIZE = 256
        self.num_primitives = num_primitives

        self.linear1 = nn.Sequential(nn.Linear(G_LATENT_SIZE, G_LATENT_SIZE),
                                        nn.LeakyReLU(True),
                                        nn.Linear(G_LATENT_SIZE, G_LATENT_SIZE),
                                        )
        self.sketch_head_top = SketchHead(d_in=D_IN+L_LATENT_SIZE+G_LATENT_SIZE+1, dims=[512, 512, 512])
        self.sketch_head_bottom = SketchHead(d_in=D_IN+L_LATENT_SIZE+G_LATENT_SIZE+1, dims=[512, 512,512])

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
        B, _ = primitive_parameters.shape[:2]
        P,S = self.num_primitives,self.segment

        local_codes = primitive_parameters[...,:256].permute(0,2,1)
        boxes = primitive_parameters[...,-8:]
        bezier_sdf,output_2d = self.bezier_SDF(boxes, code, local_codes, sample_point_coordinates, scale=None, angle=None)
        bezier_occ = torch.tanh(-1*bezier_sdf*self.sharpness)/2+.5
        occ = bezier_occ

        if part_idx >= 0:
            union_occ = occ[...,part_idx]
        else:
            with torch.no_grad():
                weights = torch.softmax(occ * self.soft_sharp, dim=-1)
            union_occ = torch.sum(weights * occ, dim=-1)

        outputs = {}
        outputs["output_3d_occ"]= union_occ
        outputs["sharpness"]= self.sharpness.item() if isinstance(self.sharpness, torch.Tensor) else self.sharpness
        outputs["soft_sharp"]= self.soft_sharp.item() if isinstance(self.soft_sharp, torch.Tensor) else self.soft_sharp
        outputs.update(output_2d)

        return outputs

    def get_boxes_params(self, primitive_parameters):
        B = primitive_parameters.shape[0]
        P = self.num_primitives

        bezier_para = primitive_parameters[...,:self.num_primitives*(4*3)].view(-1, self.num_primitives, 4, 3)
        up_dirs = primitive_parameters[...,self.num_primitives*(4*3+2):].view(-1, self.num_primitives, 1) + 1
        up_dirs *= torch.pi
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
        plane_2_occ = occ_between_2planes(transformed_points, boxes[:,:,7])
        plane_2_occ = plane_2_occ>=0.5
        latent_points = []
        for i in range(self.num_primitives):
            input_points = transformed_points[..., i*self.segment:(i+1)*self.segment, :2]/scale[..., i, :, :] \
                    if scale is not None  else transformed_points[..., i*self.segment:(i+1)*self.segment, :2]

            input_points = input_points.reshape(B, N*self.segment, 2)

            input_code = torch.cat((code, local_codes[...,i]),dim=-1)
            latent_point = add_latent_(input_points, plane_2_occ[:,:,i], input_code).float()

            latent_points.append(latent_point)
        sdfs_2d_top = [self.sketch_head_top(latent_points[i]).reshape(B, N, self.segment) for i in range(self.num_primitives)]
        sdfs_2d_bottom = [self.sketch_head_bottom(latent_points[i]).reshape(B, N, self.segment) for i in range(self.num_primitives)]

        sdfs_2d_top = torch.cat(sdfs_2d_top, dim=-1)
        sdfs_2d_bottom = torch.cat(sdfs_2d_bottom, dim=-1)
        box_ext = sdfLoft(sdfs_2d_top, sdfs_2d_bottom, boxes[..., 7], transformed_points).squeeze(-1)
        h = torch.abs(boxes[...,7].unsqueeze(1))
        msk_top = (transformed_points[...,2]>=0).int() & (transformed_points[...,2]<=h).int()
        msk_top = msk_top.int()
        msk_bottom = (transformed_points[...,2]>=-h).int() & (transformed_points[...,2]<=0).int()
        msk_bottom = msk_bottom.int()
        occ_2d_top = torch.sigmoid(-1*sdfs_2d_top*150)
        occ_2d_bottom = torch.sigmoid(-1*sdfs_2d_bottom*150)
        output_2d = {"occ_2d_top":occ_2d_top,"occ_2d_bottom":occ_2d_bottom,"msk_top":msk_top,"msk_bottom":msk_bottom}

        return box_ext,output_2d

    def get_2d_sdf(self, boxes, code, local_codes, sample_point_coordinates, h_ratio=torch.tensor([1])):
        B, N = sample_point_coordinates.shape[:2]

        transformed_points = sample_point_coordinates
        latent_points = []
        for i in range(self.num_primitives):

            input_points = transformed_points[...,:2]

            h = torch.abs(boxes[...,7].unsqueeze(1))
            h_ratio = 1-torch.abs(h_ratio-0.5)*2

            h_ratio = h_ratio.to(input_points.device)
            zs = h*h_ratio
            plane_2_occ = torch.sigmoid((h - zs)*20)
            plane_2_occ = plane_2_occ.repeat(1,N,1)
            input_code = torch.cat((code, local_codes[...,i]),dim=-1)
            latent_point = add_latent_(input_points, plane_2_occ[...,i], input_code).float()
            latent_points.append(latent_point)

        sdfs_2d_top = [self.sketch_head_top(latent_points[i]).reshape(B, N, self.segment) for i in range(self.num_primitives)]
        sdfs_2d_bottom = [self.sketch_head_bottom(latent_points[i]).reshape(B, N, self.segment) for i in range(self.num_primitives)]

        sdfs_2d_top = torch.cat(sdfs_2d_top, dim=-1)
        sdfs_2d_bottom = torch.cat(sdfs_2d_bottom, dim=-1)

        return sdfs_2d_top, sdfs_2d_bottom

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
