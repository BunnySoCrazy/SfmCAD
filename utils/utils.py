import torch

def init_seeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def add_latent(points, latent_codes):
    batch_size, num_of_points, dim = points.shape
    latent_codes = latent_codes.unsqueeze(1).repeat(1, num_of_points, 1)
    out = torch.cat([latent_codes, points], -1)
    return out

def add_latent_(points, sdf, latent_codes):
    batch_size, num_of_points, dim = points.shape
    points = points.reshape(batch_size * num_of_points, dim)
    sdf = sdf.reshape(batch_size * num_of_points, -1)
    latent_codes = latent_codes.unsqueeze(1).repeat(1, num_of_points, 1).reshape(batch_size * num_of_points, -1)
    out = torch.cat([latent_codes, points, sdf], -1)
    return out

def save_obj_data(filename, vertex, face):
    numver = vertex.shape[0]
    numfac = face.shape[0]
    with open(filename, 'w') as f:
        f.write('# %d vertices, %d faces'%(numver, numfac))
        f.write('\n')
        for v in vertex:
            f.write('v %f %f %f' %(v[0], v[1], v[2]))
            f.write('\n')
        for F in face:
            f.write('f %d %d %d' %(F[0]+1, F[1]+1, F[2]+1))
            f.write('\n')

def save_ply_data(filename, vertex, face):
    numver = vertex.shape[0]
    numfac = face.shape[0]

    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex {}\n'.format(numver))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('element face {}\n'.format(numfac))
        f.write('property list uchar int vertex_index\n')
        f.write('end_header\n')

        for v in vertex:
            f.write('%f %f %f\n' %(v[0], v[1], v[2]))

        for F in face:
            f.write('3 %d %d %d\n' %(F[0], F[1], F[2]))