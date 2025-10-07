import torch
import torch.nn as nn
from utils.sdfs import sample_cubic_bezier
import torch.nn.functional as F
import os

class HWLoss(nn.Module):
    def __init__(self, max_size=0.1, weight=0.05):
        super(HWLoss, self).__init__()
        self.max_size = max_size
        self.weight = weight

    def forward(self, hw, p0123):
        diff = hw - self.max_size
        diff = torch.relu(diff)
        hw_loss = diff.abs().mean()

        start = sample_cubic_bezier(p0123, t=torch.Tensor([0.001]).to(hw.device))
        end = sample_cubic_bezier(p0123, t=torch.Tensor([0.999]).to(hw.device))

        D = torch.dist(start, end)
        HW_product = torch.prod(hw, dim=-1)
        D_diff = torch.sqrt(HW_product) - D
        D_diff = torch.relu(D_diff)
        D_loss = D_diff.abs().mean()

        return self.weight * (hw_loss + D_loss)

class reconLoss(nn.Module):
    def __init__(self, weights, CB_loss=False, SM_loss=False, HW_loss=False, strategy='grow'):
        super().__init__()
        self.weights = weights
        self.CB_loss = CB_loss
        self.SM_loss = SM_loss
        self.HW_loss = HW_loss
        self.strategy = strategy

    def calculate_sm_loss(self, vectors):
        cosines = torch.sum(vectors[..., :-1, :] * vectors[..., 1:, :], dim=-1) / (torch.norm(vectors[..., :-1, :], dim=-1) * torch.norm(vectors[..., 1:, :], dim=-1))
        loss = 1-cosines.min()
        return loss

    def curve_inside_loss(self, curve_3d_occ):
        target = torch.ones_like(curve_3d_occ).to(curve_3d_occ.device)
        loss_func = nn.MSELoss()
        loss = loss_func(curve_3d_occ, target)
        return loss

    def forward(self, outputs, gt_3d_occ, voxels=None):
        output_3d_occ = outputs["output_3d_occ"]

        if torch.min(gt_3d_occ).item()<0:
            gt_3d_occ = -gt_3d_occ
            positive_part = (gt_3d_occ > 0).float()
            negative_part = gt_3d_occ * (gt_3d_occ <= 0).float()
            gt_3d_occ = positive_part + negative_part

        if self.CB_loss:
            pos_weight = gt_3d_occ.numel() / gt_3d_occ.sum()
            neg_weight = gt_3d_occ.numel() / (gt_3d_occ.numel() - gt_3d_occ.sum())

            weights = gt_3d_occ.clone().detach()
            weights[gt_3d_occ == 1] = pos_weight
            weights[gt_3d_occ == 0] = neg_weight

            loss_recon = nn.MSELoss(reduction='none')(gt_3d_occ, output_3d_occ)
            loss_recon = (loss_recon * weights).mean()
        else:
            if self.strategy == 'balance':
                loss_recon = nn.MSELoss()(output_3d_occ, gt_3d_occ)
            elif self.strategy == 'grow':
                weights = gt_3d_occ.clone().detach()
                weights[gt_3d_occ == 1] = 1.5
                weights[gt_3d_occ == 0] = 1
                loss_recon = nn.MSELoss(reduction='none')(gt_3d_occ, output_3d_occ)
                loss_recon = (loss_recon * weights).mean()
            elif self.strategy == 'shrink':
                weights = gt_3d_occ.clone().detach()
                weights[gt_3d_occ == 1] = 1
                weights[gt_3d_occ == 0] = 1.5
                loss_recon = nn.MSELoss(reduction='none')(gt_3d_occ, output_3d_occ)
                loss_recon = (loss_recon * weights).mean()

        res = {"L_recon": loss_recon}

        if self.HW_loss:
            hw = outputs['hw']
            L_curve_sm = HWLoss()(hw,outputs['bezier_para'])
            res["L_hw"] = L_curve_sm

        if 'bezier_para' in outputs and self.SM_loss:
            bezier_para = outputs['bezier_para']
            freq = 30
            t_values = torch.linspace(0, 1, freq+1).to(output_3d_occ.device)
            curve_points = sample_cubic_bezier(bezier_para, t_values)
            vectors = curve_points[..., 1:, :]-curve_points[..., :freq, :]
            L_curve_sm = self.calculate_sm_loss(vectors)
            res["L_cur_sm"]= L_curve_sm*0.05

        if voxels is not None and "sample_3d_occ" in outputs:
            sample_3d = outputs["sample_3d_occ"]
            sample_3d_coord, sample_3d_occ = torch.split(sample_3d, [3,1], dim=-1)
            sample_3d_coord = torch.floor((sample_3d_coord+0.5)*64).long()
            sample_3d_coord = torch.clamp(sample_3d_coord, min=0, max=63)
            B, N = sample_3d_coord.shape[:2]
            sample_3d_occ_gt = voxels[
                torch.arange(B).unsqueeze(1).repeat(1, N).view(-1),
                0,
                sample_3d_coord[:, :, 0].view(-1),
                sample_3d_coord[:, :, 1].view(-1),
                sample_3d_coord[:, :, 2].view(-1)
            ].view(B, N, 1)

            loss_recon_sample = nn.MSELoss()(sample_3d_occ, sample_3d_occ_gt)
            res["L_recon_smp"]= loss_recon_sample

        return res

class seperate_Loss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def forward(self, outputs, gt_3d_occ):
        box_3d_occ = outputs["box_3d_occ"]
        bezier_3d_occ = outputs["bezier_3d_occ"]
        loss_box = nn.MSELoss()(gt_3d_occ, box_3d_occ)
        loss_bezier = nn.MSELoss()(gt_3d_occ, bezier_3d_occ)

        res = {"L_recon": loss_box,
               "L_recon": loss_bezier
               }

        return res
