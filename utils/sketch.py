import os
import torch
import numpy as np
import cv2
import trimesh
import numpy as np
from .utils import add_latent


def create_frenet_sketch(
	shape_3d, filename, fill_sk_list, part_idx
):
    """
    Args:
        shape_3d: 3D shape parameters.
    """
    shape_3d = shape_3d.detach().cpu()
    num_primitives = int(shape_3d.shape[-1]/(12+2+2))
    bezier_para = shape_3d[...,: num_primitives*(4*3)].view(-1, num_primitives, 4, 3) # B,P,4,3
    hws = torch.abs(shape_3d[...,num_primitives*(4*3):num_primitives*(4*3+2)].view(-1, num_primitives, 2)) # B,P,2
    top_dirs = shape_3d[..., num_primitives*(4*3+2):num_primitives*(4*3+3)].view(-1, num_primitives, 1) + 1 # B,P,1
    bottom_dirs = shape_3d[..., num_primitives*(4*3+3):].view(-1, num_primitives, 1) + 1 # B,P,1
    
    control_points = bezier_para[0,part_idx]
    hws = hws[0,part_idx]
    top_dirs = top_dirs[0,part_idx]
    bottom_dirs = bottom_dirs[0,part_idx]

    if shape_3d.shape[-1]==12+2+2:
        top_dirs = shape_3d[..., num_primitives*(4*3+2):num_primitives*(4*3+3)].view(-1, num_primitives, 1) + 1 # B,P,1
        bottom_dirs = shape_3d[..., num_primitives*(4*3+3):].view(-1, num_primitives, 1) + 1 # B,P,1
    elif shape_3d.shape[-1]==12+2+1:
        top_dirs = shape_3d[..., num_primitives*(4*3+2):num_primitives*(4*3+3)].view(-1, num_primitives, 1) + 1 # B,P,1
        bottom_dirs = top_dirs # B,P,1
        
    elif shape_3d.shape[-1]==12+2:
        top_dirs = torch.ones((1))
        bottom_dirs = top_dirs # B,P,1
        
    # print('hws', hws.shape)
    angles = torch.cat([top_dirs,bottom_dirs],dim=-1)
    angles = angles.squeeze()
    hws = hws.squeeze()

    frenet_sketch(control_points, fill_sk_list, filename+'.obj', hws, angles)
    

def calculate_bezier_curve(P0, P1, P2, P3, num_points):
    t_values = np.linspace(0.0, 1.0, num_points)
    points = []
    for t in t_values:
        point = (1 - t)**3 * P0 + 3 * (1 - t)**2 * t * P1 + 3 * (1 - t) * t**2 * P2 + t**3 * P3
        points.append(point)
    return points


def get_sketch_list(generator, shape_code, res, shape_3d, part_idx):
    num_primitives = int(shape_3d.shape[-1]/(12+2+2))
    hws = torch.abs(shape_3d[...,num_primitives*(4*3):num_primitives*(4*3+2)].view(-1, num_primitives, 2)) # B,P,2
    hws = hws[0,part_idx]
    # xs = -0.5*hws[0].item()
    # ys = 0.5*hws[1].item()
    # print(xs, ys)
    
    B, N = 1, res*res
    x1, x2 = np.mgrid[-0.5:0.5:complex(0,res), -0.5:0.5:complex(0,res)]
    # x1, x2 = np.mgrid[xs:ys:complex(0,res), xs:ys:complex(0,res)]
    
    
    x1 = torch.from_numpy(x1)*1
    x2 = torch.from_numpy(x2)*1
    sample_points = torch.dstack((x1,x2)).view(-1,2).unsqueeze(0).cuda()

    shape_code_cuda = shape_code.cuda()
    sdfs_2d_list = []
    head = getattr(generator, 'sketch_head')
    
    latent = add_latent(sample_points, shape_code_cuda).float()
    
    sdfs_2d = head(latent).reshape(B,N,-1).float().squeeze().detach().cpu().unsqueeze(-1).numpy()
    sdfs_2d_list.append(sdfs_2d)

    sample_points = sample_points.detach().cpu().numpy()[0][:,:2]/1+0.5

    fill_sk_list=[]
    for dis in sdfs_2d_list:
        a = np.hstack((sample_points,dis))
        canvas = np.zeros((res+80,res+80))
        for i in a:
            canvas[int((i[1])*res)][int((i[0])*res)] = i[2]
        sk = canvas
        result = sk[:res,:res]
        bin_img = (result)
        imgray = (bin_img<-0.01).astype('uint8')*255

        ret, thresh = cv2.threshold(imgray, 254, 255, 0)
        contours, b = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        my_contour_list = []
        fill_polygon_list = []
        
        if len(contours)==0:
            my_contour_list.append(None)
            fill_sk_list.append(None)
            continue

        hir=b[0][...,3]

        for c in contours:
            my_contour_list.append(c[:,0,:])
            
        polygon_list = trimesh.path.polygons.paths_to_polygons(my_contour_list)
        for polygon in polygon_list:
            if polygon == None:
                continue

            path_2d = trimesh.path.exchange.misc.polygon_to_path(polygon)
            path = trimesh.path.Path2D(path_2d['entities'],path_2d['vertices'])

            max_values = np.max(path_2d['vertices'], axis=0)
            min_values = np.min(path_2d['vertices'], axis=0)
            size = np.linalg.norm(max_values - min_values)
            smooth_value = size/2
            # smooth_value = 100
            sm_path = trimesh.path.simplify.simplify_spline(path, smooth=smooth_value, verbose=True)

            a,_ = sm_path.triangulate()
            polygon = trimesh.path.polygons.paths_to_polygons([a])
            if polygon[0] == None:
                continue
            Matrix = np.eye(3)
            Matrix[0,2] =- res/2
            Matrix[1,2] =- res/2
            polygon = trimesh.path.polygons.transform_polygon(polygon[0],Matrix)
            
            # 缩放矩阵
            scale_matrix = np.eye(3)
            # scale_matrix[0, 0] = 1 / res* hws[0].item() * 2
            # scale_matrix[1, 1] = 1 / res* hws[1].item() * 2
            scale_matrix[0, 0] = 1 / res* hws[0].item() * 1
            scale_matrix[1, 1] = 1 / res* hws[1].item() * 1
            # 将polygon缩放为原本的1/res倍
            polygon = trimesh.path.polygons.transform_polygon(polygon, scale_matrix)
            
            fill_polygon_list.append(polygon)
        if len(fill_polygon_list)==0:
            fill_sk_list.append(None)
            continue
        fill_sk = fill_polygon_list[0]
        for i in range(1,len(fill_polygon_list)):
            if hir[i]%2==1:
                fill_sk = fill_sk | fill_polygon_list[i]
            # else:
            #     fill_sk = fill_sk - fill_polygon_list[i]
        fill_sk_list.append(fill_sk)
    return fill_sk_list
    
    
def frenet_sketch(control_points, fill_sk_list, filename, hws, angles):
    assert control_points.shape == (4, 3), "Input tensor should have shape (4, 3)"

    P0, P1, P2, P3 = control_points
    # print('angles', angles.shape)
    
    box_length = hws[...,0]
    box_width = hws[...,1]
    
    rotation_angle_z_start = angles[...,0]
    rotation_angle_z_end = angles[...,1]
    
    points = calculate_bezier_curve(P0, P1, P2, P3, num_points=100)

    pipe = create_sketch_sweep(points, fill_sk_list, rotation_angle_z_start, rotation_angle_z_end)
    
    if pipe is None:
        return None
        
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
        
                
    with open(filename, 'w') as f:
        f.write('o boxes\n')
        for v in pipe.vertices:
            f.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
        for face in pipe.faces:
            f.write('f {} {} {}\n'.format(face[0] + 1, face[1] + 1, face[2] + 1))
            
            
def create_sketch_sweep(points, fill_sk_list, rotation_angle_z_start=0, rotation_angle_z_end=0):
    num_steps = len(points) - 1
    sk_meshes = []
    for fill_sk in fill_sk_list:
        if fill_sk is not None:
            vertices, faces = trimesh.creation.triangulate_polygon(fill_sk)
            vertices = np.insert(vertices, 2, values=0, axis=1)  # z as 0
            sk_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            sk_meshes.append(sk_mesh)


    meshes = []
    for i in range(num_steps):
        start = points[i]
        end = points[i + 1]
        direction = end - start
        box_height = np.linalg.norm(direction)  # height of the box is the distance between start and end
        direction = direction / box_height
        
        sk = trimesh.util.concatenate(sk_meshes)
        if isinstance(sk, list):
            print('no sk')
            return None
            
        rotation_direction = np.cross([0, 0, 1], direction)
        rotation_angle = np.arccos(np.dot([0, 0, 1], direction))

        rotation_angle_z = np.interp(i, [0, num_steps-1], [rotation_angle_z_start, rotation_angle_z_end])
            
        sk.apply_transform(trimesh.transformations.rotation_matrix(np.radians(rotation_angle_z), [0, 0, 1]))
        sk.apply_transform(trimesh.transformations.rotation_matrix(rotation_angle, rotation_direction))
        sk.apply_transform(trimesh.transformations.translation_matrix((start + end) / 2))

        meshes.append(sk)

        
    return trimesh.util.concatenate(meshes)