import os
import sys
import yaml
import json
import torch
import argparse

sys.path.append("./nvdiffmodeling")

from loops.single import single_loop
from loops.util import random_string, set_seed

def dir_path(string):
    if os.path.exists(string):
        return string
    else:
        raise FileNotFoundError

parser = argparse.ArgumentParser(description='CLIPMatrix++ | Text to 3D character meshes, please provide a config file with --path OR pass in parameters (default values for params are in configs/base.yaml)')
parser.add_argument('--path', type=dir_path, default='configs/base.yaml')

parser.add_argument('--optim', nargs='*', help="What to optimize from [body, expression, texture, normal, specular, pose, displacement]")
parser.add_argument('--options', nargs='*', help="What views to optimize from [face, full, back]")

parser.add_argument('--epochs', type=int, help="How many epochs to run")
parser.add_argument('--gpu', type=int, help="Which GPU")
parser.add_argument('--log_int', type=int, help="How often you want to log debug (also needs --debug_log=True")
parser.add_argument('--batch_size', type=int, help="How many images per epoch")
parser.add_argument('--render_res', type=int, help="Render iamge size")
parser.add_argument('--texture_res', type=int, help="Texture resolution ex: 256x256")

parser.add_argument('--shape_lr', type=float, help="LR for expression, body and pose")
parser.add_argument('--TV_weight', type=float, help="Texture denoising weight")
parser.add_argument('--texture_lr', type=float, help="Texture weight")
parser.add_argument('--displacement_lr', type=float, help="Displacement map weight")

parser.add_argument('--face_text', type=str, help="Text prompt for face region")
parser.add_argument('--full_text', type=str, help="Text prompt for full body")
parser.add_argument('--back_text', type=str, help="Text prompt for back of body")

parser.add_argument('--CLIP', type=str, help="CLIP Model to use")
parser.add_argument('--gender', type=str, help="SMPL-X Gender")
parser.add_argument('--render', type=str, help="One from [diffuse, pbr, normal, tangent]")
parser.add_argument('--uv_path', type=str, help="Path to base obj containing UV cords and faces info")
parser.add_argument('--base_pose', type=str, help="Baseline pose during optimization")
parser.add_argument('--model_type', type=str, help="smplx or smpl, check smplx library for other options, keep smplx in most cases")
parser.add_argument('--output_path', type=str, help="Where to output result")
parser.add_argument('--model_folder', type=str, help="Path to download smplx models")
parser.add_argument('--uv_mask_path', type=str, help="Path to UV mask")


parser.add_argument('--rand_bkg', type=bool, help="Augment background during trianing")
parser.add_argument('--rand_pose', type=bool, help="Augment pose during trianing")
parser.add_argument('--debug_log', type=bool, help="Log video and training renders")


if __name__ == "__main__":
    parsed_args = parser.parse_args()
    
    with open(parsed_args.path, "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    parsed_args = vars(parsed_args)
    parsed_args = {k: v for k, v in parsed_args.items() if v is not None}

    cfg.update(parsed_args)
    print(json.dumps(cfg, sort_keys=True, indent=4))
    
    device = torch.device('cuda:' + str(cfg['gpu']))
    torch.cuda.set_device(device)

    try:
        cfg['ID'] = cfg['full_text'] + "-" + random_string(5)
    except:
        cfg['ID'] = cfg['face_text'] + "-" + random_string(5)

    set_seed(cfg['seed'])

    if cfg['loop'] == "single":
        single_loop(cfg)
