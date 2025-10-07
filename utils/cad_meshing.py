import os
import time
import torch
import mcubes
from utils import utils
import numpy as np
import h5py
import trimesh
from scipy.spatial.transform import Rotation as R

def calculate_bezier_curve(P0, P1, P2, P3, num_points):
    t_values = np.linspace(0.0, 1.0, num_points)
    points = []
    for t in t_values:
        point = (1 - t)**3 * P0 + 3 * (1 - t)**2 * t * P1 + 3 * (1 - t) * t**2 * P2 + t**3 * P3
        points.append(point)
    return points

def create_pipe(points, radius=0.02, sections=20):
    meshes = []
    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]
        direction = end - start
        length = np.linalg.norm(direction)
        direction = direction / length

        cylinder = trimesh.creation.cylinder(radius, length*1.05, sections)

        rotation_direction = np.cross([0, 0, 1], direction)
        rotation_angle = np.arccos(np.dot([0, 0, 1], direction))
        rotation_matrix = R.from_rotvec(rotation_angle * rotation_direction).as_matrix()

        cylinder.apply_transform(trimesh.transformations.rotation_matrix(rotation_angle, rotation_direction))
        cylinder.apply_transform(trimesh.transformations.translation_matrix((start + end) / 2))

        meshes.append(cylinder)
    return trimesh.util.concatenate(meshes)

def create_sphere(center, radius=0.1):
    return trimesh.creation.icosphere(subdivisions=2, radius=radius, color=None).apply_translation(center)

def create_bezier_mesh(control_points, filename, scale=1):
    assert control_points.shape == (4, 3), "Input tensor should have shape (4, 3)"

    P0, P1, P2, P3 = control_points

    points = calculate_bezier_curve(P0, P1, P2, P3, num_points=10)

    pipe = create_pipe(points,radius=0.005/scale)
    sphere_start = create_sphere(P0, radius=0.01/scale)
    sphere_end = create_sphere(P3, radius=0.01/scale)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    with open(filename, 'w') as f:
        for v in pipe.vertices:
            f.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
        for face in pipe.faces:
            f.write('f {} {} {}\n'.format(face[0] + 1, face[1] + 1, face[2] + 1))

        offset = len(pipe.vertices)
        for v in sphere_start.vertices:
            f.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
        for face in sphere_start.faces:
            f.write('f {} {} {}\n'.format(face[0] + 1 + offset, face[1] + 1 + offset, face[2] + 1 + offset))

        offset += len(sphere_start.vertices)
        for v in sphere_end.vertices:
            f.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
        for face in sphere_end.faces:
            f.write('f {} {} {}\n'.format(face[0] + 1 + offset, face[1] + 1 + offset, face[2] + 1 + offset))

def create_boxes(points, box_length=0.02, box_width=0.02, rotation_angle_z_start=0, rotation_angle_z_end=0):
    meshes = []
    num_steps = len(points) - 1
    for i in range(num_steps):
        start = points[i]
        end = points[i + 1]
        direction = end - start
        box_height = np.linalg.norm(direction)
        direction = direction / box_height

        box = trimesh.creation.box((box_length, box_width, box_height))

        rotation_direction = np.cross([0, 0, 1], direction)
        rotation_angle = np.arccos(np.dot([0, 0, 1], direction))

        rotation_angle_z = np.interp(i, [0, num_steps-1], [rotation_angle_z_start, rotation_angle_z_end])

        box.apply_transform(trimesh.transformations.rotation_matrix(np.radians(rotation_angle_z), [0, 0, 1]))

        box.apply_transform(trimesh.transformations.rotation_matrix(rotation_angle, rotation_direction))
        box.apply_transform(trimesh.transformations.translation_matrix((start + end) / 2))

        meshes.append(box)
    return trimesh.util.concatenate(meshes)

def frenet_boxes(control_points, filename, hws, angles):
    assert control_points.shape == (4, 3), "Input tensor should have shape (4, 3)"

    P0, P1, P2, P3 = control_points

    box_length = hws[...,0]
    box_width = hws[...,1]

    rotation_angle_z_start = angles[...,0]
    rotation_angle_z_end = angles[...,1]

    points = calculate_bezier_curve(P0, P1, P2, P3, num_points=20)

    pipe = create_boxes(points, box_length, box_width, rotation_angle_z_start, rotation_angle_z_end)

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    with open(filename, 'w') as f:
        f.write('o boxes\n')
        for v in pipe.vertices:
            f.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
        for face in pipe.faces:
            f.write('f {} {} {}\n'.format(face[0] + 1, face[1] + 1, face[2] + 1))

def create_frenet_boxes(shape_3d, filename):
    """
    Args:
        shape_3d: 3D shape parameters.
    """
    shape_3d = shape_3d.detach().cpu()
    num_primitives = 1
    bezier_para = shape_3d[...,:(4*3)].view(-1, 1, 4, 3)
    control_points = bezier_para[0,0]

    hws = torch.abs(shape_3d[...,num_primitives*(4*3):num_primitives*(4*3+2)].view(-1, num_primitives, 2))
    top_dirs = shape_3d[..., num_primitives*(4*3+2):num_primitives*(4*3+3)].view(-1, num_primitives, 1) + 1
    bottom_dirs = shape_3d[..., num_primitives*(4*3+3):].view(-1, num_primitives, 1) + 1

    angles = torch.cat([top_dirs,bottom_dirs],dim=-1)
    hws = hws.squeeze()
    angles = angles.squeeze()

    frenet_boxes(control_points, filename+'.obj', hws, angles)

def create_curves(shape_3d, filename, Part_mode=True):
    """
    Args:
        shape_3d: 3D shape parameters.
    """
    shape_3d = shape_3d.detach().cpu()

    if Part_mode:
        bezier_para = shape_3d[...,:(4*3)].view(-1, 1, 4, 3)
        control_points = bezier_para[0,0]
        shape_name = filename.split('/')[-2]
        str_index = filename.split('/')[-1]
        h5_filepath = os.path.join('/data/dataset/Sweep/Shape_Part/Chair/Chair_off_h5_post_sample/Chair_off_h5_post', shape_name+'.h5')

        with h5py.File(h5_filepath, 'r') as f:
            scales = np.array(f['scales'][int(str_index)])
        create_bezier_mesh(control_points, filename+'.obj', scales)
    else:
        num_primitives = int(shape_3d.shape[-1]/(12+2+2))
        bezier_para = shape_3d[...,: num_primitives*(4*3)].view(-1, num_primitives, 4, 3)
        for part_idx in range(num_primitives):
            control_points = bezier_para[0, part_idx]
            create_bezier_mesh(control_points, filename + f'_{part_idx}.obj')

def create_mesh_mc(generator, shape_3d, shape_code, filename, N=128, max_batch=32**3, threshold=0.5, part_idx=-1,phase='box'):
    """
    Create a mesh using the marching cubes algorithm.

    Args:
        generator: The generator of network.
        shape_3d: 3D shape parameters.
        shape_code: Shape code.
        N: Resolution parameter.
        threshold: Marching cubes threshold value.
    """

    start = time.time()
    mesh_filename = filename

    voxel_origin = [0, 0, 0]
    voxel_size = 1

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]
    samples[:, :3] = (samples[:, :3])/(0.5*N)-1

    num_samples = N**3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()

        outputs = generator(sample_subset.unsqueeze(0), shape_3d, shape_code, phase=phase)
        occ = outputs["output_3d_occ"]
        samples[head : min(head + max_batch, num_samples), 3] = (
            occ.reshape(-1)
            .detach()
            .cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    print(f"Sampling took: {end - start:.3f} seconds")

    numpy_3d_sdf_tensor = sdf_values.numpy()

    verts, faces = mcubes.marching_cubes(numpy_3d_sdf_tensor, threshold)

    mesh_points = verts
    mesh_points = (mesh_points + 0.5) / N - 0.5

    if not os.path.exists(os.path.dirname(mesh_filename)):
        os.makedirs(os.path.dirname(mesh_filename))

    utils.save_ply_data(f"{mesh_filename}.ply", mesh_points, faces)

def create_CAD_mesh(generator, shape_code, shape_3d, CAD_mesh_filepath):
    """
    Reconstruct shapes with sketch-extrude operations.

    Notes:
        - This function currently contains no implementation and serves as a stub.
    """
    pass

def draw_2d_im_sketch(shape_code, generator, sk_filepath):
    """
    Draw a 2D sketch.

    Notes:
        - This function currently contains no implementation and serves as a stub.
    """
    pass
