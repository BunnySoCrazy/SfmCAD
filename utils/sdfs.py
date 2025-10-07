import torch
import torch.nn as nn

def quaternion_raw_multiply(a, b):
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)

def quaternion_invert(quaternion):
    return quaternion * quaternion.new_tensor([1, -1, -1, -1])

def quaternion_apply(quaternion, point):
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, f{point.shape}.")
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., 1:]

def transform_points(quaternion, translation, points):
    quaternion = nn.functional.normalize(quaternion, dim=-1)
    transformed_points = points.unsqueeze(2) - translation.unsqueeze(1)
    transformed_points = quaternion_apply(quaternion.unsqueeze(1), transformed_points)
    return transformed_points

def sdfBox_(quaternion, translation, dims, points):
    B, N, _ = points.shape
    _, PxS, _ = quaternion.shape

    dims = torch.abs(dims)
    transformed_points = transform_points(quaternion, translation, points)
    q_points = transformed_points.abs() - dims.unsqueeze(1).repeat(1, N, 1, 1)
    lengths = (q_points.max(torch.zeros_like(q_points))).norm(dim=-1)
    zeros_points = torch.zeros_like(lengths)
    xs = q_points[..., 0]
    ys = q_points[..., 1]
    zs = q_points[..., 2]
    filling = ys.max(zs).max(xs).min(zeros_points)
    return lengths + filling

def sdfBox_roatate2d(quaternion, translation, dims, points, angle):
    B, N, _ = points.shape
    _, PxS, _ = quaternion.shape

    transformed_points = transform_points(quaternion, translation, points)

    cos_angle = torch.cos(angle).squeeze(-1)
    sin_angle = torch.sin(angle).squeeze(-1)

    x = transformed_points[..., 0]
    y = transformed_points[..., 1]
    z = transformed_points[..., 2]

    rotated_x = x * cos_angle.unsqueeze(1) - y * sin_angle.unsqueeze(1)
    rotated_y = x * sin_angle.unsqueeze(1) + y * cos_angle.unsqueeze(1)

    transformed_points = torch.stack([rotated_x, rotated_y, z], dim=-1)

    q_points = transformed_points.abs() - dims.unsqueeze(1).repeat(1, N, 1, 1)
    lengths = (q_points.max(torch.zeros_like(q_points))).norm(dim=-1)

    zeros_points = torch.zeros_like(lengths)
    xs = q_points[..., 0]
    ys = q_points[..., 1]
    zs = q_points[..., 2]
    filling = ys.max(zs).max(xs).min(zeros_points)
    return lengths + filling

def sdfExtrusion(sdf_2d, h, points):
    transformed_points = points
    h = torch.abs(h)
    d = sdf_2d

    z_diff = transformed_points[..., 2].abs() - h.unsqueeze(dim=1)
    a = d.max(z_diff).min(torch.zeros_like(z_diff))
    b = torch.cat((d.max(torch.zeros_like(z_diff)).unsqueeze(-1), (z_diff).max(torch.zeros_like(z_diff)).unsqueeze(-1)), -1)
    b = b.norm(dim=-1)

    return a + b

def sample_cubic_bezier(p0123, t):
    B, P, _, _ = p0123.shape
    S = t.shape[0]
    t = t.unsqueeze(-1).unsqueeze(0).unsqueeze(0)
    t = t.expand(B, P, S, 3)

    p0 = p0123[..., 0, :].unsqueeze(-2)
    p1 = p0123[..., 1, :].unsqueeze(-2)
    p2 = p0123[..., 2, :].unsqueeze(-2)
    p3 = p0123[..., 3, :].unsqueeze(-2)

    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * (t ** 2) * p2 + (t ** 3) * p3

def get_rotation_quaternion(p1, p2, angle_rad=None):
    direction1 = p2 - p1
    direction1 = direction1 / (torch.linalg.norm(direction1, dim=-1, keepdim=True) + 1e-8)

    direction2 = torch.tensor([0.0, 0.0, 1.0]).to(direction1.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    direction2 = direction2.expand_as(direction1)

    cross_product = torch.cross(direction1, direction2, dim=-1)
    dot_product = torch.sum(direction1 * direction2, dim=-1)
    norm_cross = torch.norm(cross_product, dim=-1)

    angle = torch.atan2(norm_cross, dot_product)
    axis = cross_product / norm_cross[..., None]

    cos_theta = torch.cos(angle / 2.0)
    sin_theta = torch.sin(angle / 2.0)

    return torch.stack([cos_theta, axis[..., 0] * sin_theta, axis[..., 1] * sin_theta, axis[..., 2] * sin_theta], dim=-1)

def get_segment_direction(p1, p2):
    return p2 - p1

def points_to_boxes_3d(points, up_dirs):
    B, P, S, _ = points.shape
    S -= 1
    p1 = points[..., :S, :]
    p2 = points[..., 1:, :]

    direction = get_segment_direction(p1, p2)
    length = torch.norm(direction, dim=-1)
    angle = get_rotation_quaternion(p1, p2, up_dirs)
    x_center = (p1[..., 0] + p2[..., 0]) / 2
    y_center = (p1[..., 1] + p2[..., 1]) / 2
    z_center = (p1[..., 2] + p2[..., 2]) / 2
    translation = torch.stack([x_center, y_center, z_center], dim=-1)

    h = length.unsqueeze(-1)/2

    result = torch.cat((angle, translation, h), dim=-1)
    return result

def sample_points_within_boxes(boxes, hw, res=(12, 12, 5), up_dir=None):
    B, P, S, _ = boxes.shape
    angle, translation, h = torch.split(boxes, [4, 3, 1], dim=-1)

    w, d = torch.split(hw[..., :2], [1, 1], dim=-1)

    x_linspace = torch.linspace(-1, 1, res[0], device=boxes.device)
    y_linspace = torch.linspace(-1, 1, res[1], device=boxes.device)
    z_linspace = torch.linspace(-1, 1, res[2], device=boxes.device)

    x_grid_base, y_grid_base, z_grid_base = torch.meshgrid(x_linspace, y_linspace, z_linspace)
    grid_shape = [B, P, S] + list(res)

    x_grid_base = x_grid_base.expand(*grid_shape)
    y_grid_base = y_grid_base.expand(*grid_shape)
    z_grid_base = z_grid_base.expand(*grid_shape)

    if up_dir is not None:
        up_dir = up_dir.reshape(B, P, S)
        cos_angle = torch.cos(up_dir).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        sin_angle = torch.sin(up_dir).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = x_grid_base
        y = y_grid_base
        x_grid_base = x * cos_angle - y * sin_angle
        y_grid_base = x * sin_angle + y * cos_angle

    x_grid = x_grid_base * w.unsqueeze(-1).unsqueeze(-1)
    y_grid = y_grid_base * d.unsqueeze(-1).unsqueeze(-1)
    z_grid = z_grid_base * h.unsqueeze(-1).unsqueeze(-1)

    samples = torch.stack([x_grid, y_grid, z_grid], dim=-1)

    samples_rotated = quaternion_apply(quaternion_invert(angle.unsqueeze(-2).unsqueeze(-2).unsqueeze(-2)), samples)

    samples_translated = samples_rotated + translation.unsqueeze(-2).unsqueeze(-2).unsqueeze(-2)

    return samples_translated.view(B, P, S, res[0]*res[1], res[2], 3)

def sdfLoft(sdf_2d_top, sdf_2d_bottom, h, points):
    transformed_points = points
    h = torch.abs(h)
    a = (transformed_points[..., 2] + h.unsqueeze(dim=1))/(h.unsqueeze(dim=1)*2)
    a = a.clamp(0, 1)
    d = sdf_2d_top*a+sdf_2d_bottom*(1-a)

    z_diff = transformed_points[..., 2].abs() - h.unsqueeze(dim=1)
    a = d.max(z_diff).min(torch.zeros_like(z_diff))
    b = torch.cat((d.max(torch.zeros_like(z_diff)).unsqueeze(-1), (z_diff).max(torch.zeros_like(z_diff)).unsqueeze(-1)), -1)
    b = b.norm(dim=-1)
    return a + b

def occ_between_2planes(transformed_points, hs):
    B, N, K, _ = transformed_points.shape
    B, K = hs.shape
    hs = hs.unsqueeze(1).repeat(1, N, 1)

    zs = transformed_points[..., 2]
    return hs.abs()-zs.abs()