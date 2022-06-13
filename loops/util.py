import os
import clip
import torch
import string
import random
import imageio

import numpy as np
import torchvision.transforms as transforms

from resize_right import resize

cosine_sim   = torch.nn.CosineSimilarity()

def cosine_avg(features, targets):
    return -cosine_sim(features, targets).mean()
    
def unit_size(mesh_):
    vmin, vmax = torch.min(mesh_.v_pos, dim=0).values, torch.max(mesh_.v_pos, dim=0).values
    scale = 2 / torch.max(vmax - vmin).item()
    v_pos = mesh_.v_pos - (vmax + vmin) / 2 # Center mesh on origin
    v_pos = v_pos * scale                  # Rescale to unit size

    return v_pos

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def random_string(length=10):
    letters = string.ascii_uppercase
    return ''.join(random.choice(letters) for i in range(length))

blurs = [
    transforms.Compose([
        transforms.GaussianBlur(11, sigma=(5, 5))
    ]),
    transforms.Compose([
        transforms.GaussianBlur(11, sigma=(2, 2))
    ]),
    transforms.Compose([
        transforms.GaussianBlur(5, sigma=(5, 5))
    ]),
    transforms.Compose([
        transforms.GaussianBlur(5, sigma=(2, 2))
    ]),
]

def get_random_bg(device, h, w):

    p = torch.rand(1)

    if p > 0.66666:
        background =  blurs[random.randint(0, 3)]( torch.rand((1, 3, h, w), device=device) ).permute(0, 2, 3, 1)
    elif p > 0.333333:
        size = random.randint(5, 10)
        background = torch.vstack([
            torch.full( (1, size, size), torch.rand(1).item() / 2, device=device),
            torch.full( (1, size, size), torch.rand(1).item() / 2, device=device ),
            torch.full( (1, size, size), torch.rand(1).item() / 2, device=device ),
        ]).unsqueeze(0)

        second = torch.rand(3)

        background[:, 0, ::2, ::2] = second[0]
        background[:, 1, ::2, ::2] = second[1]
        background[:, 2, ::2, ::2] = second[2]

        background[:, 0, 1::2, 1::2] = second[0]
        background[:, 1, 1::2, 1::2] = second[1]
        background[:, 2, 1::2, 1::2] = second[2]

        background = blurs[random.randint(0, 3)]( resize(background, out_shape=(h, w)) )

        background = background.permute(0, 2, 3, 1)

    else:
        background = torch.vstack([
            torch.full( (1, h, w), torch.rand(1).item(), device=device),
            torch.full( (1, h, w), torch.rand(1).item(), device=device ),
            torch.full( (1, h, w), torch.rand(1).item(), device=device ),
        ]).unsqueeze(0).permute(0, 2, 3, 1)

    return background

def persp_proj(fov_x=45, ar=1, near=1.0, far=50.0):
    """
    Build a perspective projection matrix.
    Parameters
    ----------
    fov_x : float
        Horizontal field of view (in degrees).
    ar : float
        Aspect ratio (w/h).
    near : float
        Depth of the near plane relative to the camera.
    far : float
        Depth of the far plane relative to the camera.
    """
    fov_rad = np.deg2rad(fov_x)

    tanhalffov = np.tan( (fov_rad / 2) )
    max_y = tanhalffov * near
    min_y = -max_y
    max_x = max_y * ar
    min_x = -max_x

    z_sign = -1.0
    proj_mat = np.array([[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])

    proj_mat[0, 0] = 2.0 * near / (max_x - min_x)
    proj_mat[1, 1] = 2.0 * near / (max_y - min_y)
    proj_mat[0, 2] = (max_x + min_x) / (max_x - min_x)
    proj_mat[1, 2] = (max_y + min_y) / (max_y - min_y)
    proj_mat[3, 2] = z_sign

    proj_mat[2, 2] = z_sign * far / (far - near)
    proj_mat[2, 3] = -(far * near) / (far - near)
    
    return proj_mat

class Video():
    def __init__(self, path, name='video_log.mp4', mode='I', fps=30, codec='libx264', bitrate='16M') -> None:
        
        if path[-1] != "/":
            path += "/"
            
        self.writer = imageio.get_writer(path+name, mode=mode, fps=fps, codec=codec, bitrate=bitrate)
    
    def ready_image(self, image, write_video=True):
        # assuming channels last - as renderer returns it
        if len(image.shape) == 4: 
            image = image.squeeze(0)[..., :3].detach().cpu().numpy()
        else:
            image = image[..., :3].detach().cpu().numpy()

        image = np.clip(np.rint(image*255.0), 0, 255).astype(np.uint8)

        if write_video:
            self.writer.append_data(image)

        return image

class CLIP():
    def __init__(self, device, model="ViT-B/32", eval=True) -> None:
        self.model, self.preprocess = clip.load(model, device=device)
        if eval:
            self.model.eval()

        self.MEAN = torch.tensor([0.48154660, 0.45782750, 0.40821073], device=device)
        self.STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)
        self.device = device
    
    def text_tokens(self, prompts_list):
        return clip.tokenize(prompts_list)

    def image_embeds(self, images, normalize=True):
        if normalize:
            return self.model.encode_image(
                (images.to(self.device) - self.MEAN[None, :, None, None]) / self.STD[None, :, None, None]
            )
        else:
            return self.model.encode_image(images.to(self.device))

    def text_embeds(self, text_tokens, with_grad=False):
        if with_grad:
            return self.model.encode_text(text_tokens.to(self.device))
        else:
            return self.model.encode_text(text_tokens.to(self.device)).detach()

def batch_rodrigues(
    rot_vecs,
    epsilon: float = 1e-8):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''
    assert len(rot_vecs.shape) == 2, (
        f'Expects an array of size Bx3, but received {rot_vecs.shape}')

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device
    dtype = rot_vecs.dtype

    angle = torch.norm(rot_vecs + epsilon, dim=1, keepdim=True, p=2)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat