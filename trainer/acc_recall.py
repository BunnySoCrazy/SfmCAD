import torch
import torch.nn as nn

class acc_recall(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, gt_3d_occ, voxels=None):
        output_3d_occ = outputs["output_3d_occ"]

        if torch.min(gt_3d_occ).item() < 0:
            predict_occupancies = (output_3d_occ > 0).float()
            target_occupancies = (gt_3d_occ < 0).float()
        else:
            predict_occupancies = (output_3d_occ > 0.5).float()
            target_occupancies = (gt_3d_occ > 0.5).float()

        accuracy = torch.sum(predict_occupancies*target_occupancies)/(torch.sum(target_occupancies)+1e-9)
        recall = torch.sum(predict_occupancies*target_occupancies)/(torch.sum(predict_occupancies)+1e-9)
        f1_score = 2 * (accuracy * recall) / (accuracy + recall + 1e-9)
        res = {"acc": accuracy,
               "rcl": recall,
               "f1": f1_score}

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
            predict_occ_smp= (sample_3d_occ > 0.5).float()
            target_occ_smp = (sample_3d_occ_gt > 0.5).float()
            accuracy_smp = torch.sum(predict_occ_smp*target_occ_smp)/(torch.sum(target_occ_smp)+1e-9)
            recall_smp = torch.sum(predict_occ_smp*target_occ_smp)/(torch.sum(predict_occ_smp)+1e-9)

            res = {"acc": accuracy,
                    "recall": recall,
                    "acc_smp": accuracy_smp,
                    "recall_smp": recall_smp}
        return res
