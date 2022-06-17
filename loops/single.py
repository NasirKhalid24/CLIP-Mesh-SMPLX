import os
import glm
import tqdm
import smplx
import torch
import kornia
import torchvision

import numpy as np
import nvdiffrast.torch as dr

from resize_right import resize
from resize_right.interp_methods import lanczos3

from loops.util import CLIP, Video, batch_rodrigues, cosine_avg, get_random_bg, persp_proj, unit_size

from nvdiffmodeling.src import obj
from nvdiffmodeling.src import mesh
from nvdiffmodeling.src import render
from nvdiffmodeling.src import texture

def single_loop(config):

    if config["plot"]:
        from IPython.display import clear_output
        import matplotlib.pyplot as plt

    glctx = dr.RasterizeGLContext()

    config["path"] = os.path.join(config["output_path"], config["ID"])
    os.makedirs(config["path"])

    device = torch.device('cuda')

    if config["debug_log"]:
        video = Video(os.path.join(config["path"]))
    clip_model = CLIP(device, model=config["CLIP"])

    base_pose = np.load(config["base_pose"])
    base_pose = torch.from_numpy(base_pose).reshape(-1, 3)
    base_pose = base_pose.float()

    uv_mask = resize(
        torchvision.io.read_image(config["uv_mask_path"]),
        out_shape=(config["texture_res"], config["texture_res"])
    ).permute(1, 2, 0).repeat(1, 1, 3).to(device)

    expression     = torch.zeros ([1, config["expressions"]], requires_grad=True, device=device)
    betas          = torch.zeros ([1, config["betas"]], requires_grad=True, device=device)
    normal_map_opt = texture.create_trainable(np.array([0, 0, 1]), [config["texture_res"], config["texture_res"]], auto_mipmaps=True)
    kd_map_opt     = texture.create_trainable(np.random.uniform(size=[config["texture_res"], config["texture_res"]] + [3], low=0.0 if "texture" in config["optim"] else 0.85, high=1.0), [config["texture_res"], config["texture_res"]], auto_mipmaps=True)
    ks_map_opt     = texture.create_trainable(np.array([0, 0, 0]), [config["texture_res"], config["texture_res"]], auto_mipmaps=True)
    ds_map_opt     = torch.tensor(np.zeros([config["texture_res"], config["texture_res"]] + [1], dtype=np.float32), dtype=torch.float32, device=device, requires_grad=True)
    pose_opt       = base_pose.to(device).requires_grad_(True)

    train_params     = []
    shape_params     = []
    if "expression" in config["optim"]:
        shape_params += [expression]
    if "body" in config["optim"]:
        shape_params += [betas]
    if "pose" in config["optim"]:
        shape_params += [pose_opt]
    if "texture" in config["optim"]:
        train_params += kd_map_opt.getMips()
    if "normal" in config["optim"]:
        train_params += normal_map_opt.getMips()
    if "specular" in config["optim"]:
        train_params += ks_map_opt.getMips()

    body_model = smplx.build_layer(
        config["model_folder"],
        model_type=config["model_type"],
        gender=config["gender"],
        expression=expression,
        betas=betas,
        # body_pose=stand_pose[:, 1:] if stand_pose is not None else None
    )
    body_model = body_model.to(device)

    print("Body Model\n")
    print(body_model)
    print("\n")

    base_mesh = obj.load_obj(config["uv_path"])
    
    optimizers = []
    if len(train_params) > 0:
        optimizers.append(torch.optim.Adam(train_params, lr=config["texture_lr"]))
    if len(shape_params) > 0:
        optimizers.append(torch.optim.Adam(shape_params, lr=config["shape_lr"]))

    if "displacement" in config["optim"]:
        optimizers.append(torch.optim.Adam([ds_map_opt], lr=config["displacement_lr"]))

    t_loop = tqdm.tqdm(range(config["epochs"]))


    for i in t_loop:

        if config["rand_pose"] and 'pose' not in config["optim"]:
            with torch.no_grad():
                pose_rot = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.13299670815467834, -0.08935282379388809, -1.1807936429977417, 0.0, 0.0, 1.180069923400879, -0.03147100284695625, -0.003013884648680687, -0.19093503057956696, -0.1469287872314453, 0.22144654393196106, 0.01636524312198162, -0.14798645675182343, -0.0575576014816761, -0.07414476573467255, -0.31212642788887024, 0.028369469568133354, 0.17979948222637177, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.11167845129966736, 0.04289207234978676, -0.41644084453582764, 0.10881128907203674, -0.06598565727472305, -0.756219744682312, -0.0963931530714035, -0.09091583639383316, -0.18845966458320618, -0.11809506267309189, 0.050943851470947266, -0.5295845866203308, -0.14369848370552063, 0.055241718888282776, -0.704857349395752, -0.019182899966835976, -0.0923367589712143, -0.3379131853580475, -0.45703303813934326, -0.1962839663028717, -0.6254575848579407, -0.21465237438678741, -0.06599827855825424, -0.5068942308425903, -0.36972442269325256, -0.0603446289896965, -0.07949023693799973, -0.14186954498291016, -0.08585254102945328, -0.6355276107788086, -0.3033415675163269, -0.05788097903132439, -0.6313892006874084, -0.17612087726593018, -0.13209305703639984, -0.3733545243740082, 0.850964367389679, 0.2769227623939514, -0.09154807031154633, -0.4998386800289154, 0.026556432247161865, 0.052880801260471344, 0.5355585217475891, 0.045960985124111176, -0.27735769748687744, 0.11167845129966736, -0.04289207234978676, 0.41644084453582764, 0.10881128907203674, 0.06598565727472305, 0.756219744682312, -0.0963931530714035, 0.09091583639383316, 0.18845966458320618, -0.11809506267309189, -0.050943851470947266, 0.5295845866203308, -0.14369848370552063, -0.055241718888282776, 0.704857349395752, -0.019182899966835976, 0.0923367589712143, 0.3379131853580475, -0.45703303813934326, 0.1962839663028717, 0.6254575848579407, -0.21465237438678741, 0.06599827855825424, 0.5068942308425903, -0.36972442269325256, 0.0603446289896965, 0.07949023693799973, -0.14186954498291016, 0.08585254102945328, 0.6355276107788086, -0.3033415675163269, 0.05788097903132439, 0.6313892006874084, -0.17612087726593018, 0.13209305703639984, 0.3733545243740082, 0.850964367389679, -0.2769227623939514, 0.09154807031154633, -0.4998386800289154, -0.026556432247161865, -0.052880801260471344, 0.5355585217475891, -0.045960985124111176, 0.27735769748687744])
                
                if i < config["epochs"]-1:
                    pose_rot[48:51] =  torch.randn((len(pose_rot[48:51])))
                    pose_rot[53:66] =  torch.randn((len(pose_rot[53:66])))

                pose_rot = pose_rot.reshape(-1, 3).to(device)

                stand_pose = batch_rodrigues(pose_rot).unsqueeze(0)

        else:
            stand_pose = batch_rodrigues(pose_opt).unsqueeze(0)
        
        output = body_model(
            expression=expression,
            betas=betas,
            body_pose=stand_pose[:, 1:22] if stand_pose is not None else None,
            left_hand_pose=stand_pose[:, 22:37] if stand_pose is not None else None,
            right_hand_pose=stand_pose[:, 37:52] if stand_pose is not None else None,
            return_verts=True
        )

        if i == 0:
            no_texture = texture.create_trainable(np.random.uniform(size=[config["texture_res"], config["texture_res"]] + [3], low=0.85, high=1.0), [config["texture_res"], config["texture_res"]], auto_mipmaps=True)

        if config["blur"] is True:
            # low pass filter for textures
            ready_texture = texture.Texture2D(
                kornia.filters.gaussian_blur2d(
                    kd_map_opt.data.permute(0, 3, 1, 2),
                    kernel_size=config["kernel_size"],
                    sigma=config["blur_sigma"],
                ).permute(0, 2, 3, 1).contiguous()
            )

            ready_normal = texture.Texture2D(
                kornia.filters.gaussian_blur2d(
                    normal_map_opt.data.permute(0, 3, 1, 2),
                    kernel_size=config["kernel_size"],
                    sigma=config["blur_sigma"],
                ).permute(0, 2, 3, 1).contiguous()
            )

            ready_displ = kornia.filters.gaussian_blur2d(
              ds_map_opt.unsqueeze(0).permute(0, 3, 1, 2),
              kernel_size=config["kernel_size"],
              sigma=config["blur_sigma"],
            ).permute(0, 2, 3, 1).contiguous().squeeze(0)

            ready_mesh = mesh.Mesh(
                output.vertices[0],
                body_model.faces_tensor,
                material={
                    'bsdf': config['render'],
                    'kd': ready_texture,
                    'ks': ks_map_opt,
                    'normal': ready_normal,
                },
                base=base_mesh
            )

            notex_mesh = mesh.Mesh(
                output.vertices[0],
                body_model.faces_tensor,
                material={
                    'bsdf': config['render'],
                    'kd': no_texture,
                    'ks': ks_map_opt,
                    'normal': normal_map_opt,
                },
                base=base_mesh
            )

        else:

            ready_mesh = mesh.Mesh(
                output.vertices[0],
                body_model.faces_tensor,
                material={
                    'bsdf': config['render'],
                    'kd': kd_map_opt,
                    'ks': ks_map_opt,
                    'normal': normal_map_opt,
                },
                base=base_mesh
            )
            
            notex_mesh = mesh.Mesh(
                output.vertices[0],
                body_model.faces_tensor,
                material={
                    'bsdf': config['render'],
                    'kd': no_texture,
                    'ks': ks_map_opt,
                    'normal': normal_map_opt,
                },
                base=base_mesh
            )
        
        ready_mesh = mesh.Mesh(unit_size(ready_mesh), base=ready_mesh)
        ready_mesh = mesh.auto_normals(ready_mesh)
        ready_mesh = mesh.compute_tangents(ready_mesh)

        if "displacement" in config["optim"]:
            ready_mesh = mesh.displace(ready_mesh, ready_displ)

        notex_mesh = mesh.Mesh(unit_size(notex_mesh), base=notex_mesh)
        notex_mesh = mesh.auto_normals(notex_mesh)
        notex_mesh = mesh.compute_tangents(notex_mesh)

        if "displacement" in config["optim"]:
            notex_mesh = mesh.displace(notex_mesh, ready_displ)

        mvp = np.zeros((config["batch_size"], 4,4),  dtype=np.float32)
        campos   = np.zeros((config["batch_size"], 3), dtype=np.float32)
        lightpos = np.zeros((config["batch_size"], 3), dtype=np.float32)
        bkgs = torch.zeros((config["batch_size"], config["render_res"], config["render_res"], 3)).to(device)
        prompts = []

        face_idx = []

        for b in range(config["batch_size"]):

            op_ = config["options"][np.random.randint( len(config["options"]) )]

            if op_ == "face":
                face_idx.append(b)

            if b == 0 and 'full' in config["options"]:
                elev = np.radians( (config["cameras"]['full']["elev"][0] + config["cameras"]['full']["elev"][1])/4.0 ) 
                azim = np.radians( (config["cameras"]['full']["azim"][0] + config["cameras"]['full']["azim"][1])/2.0 )
                dist = config["cameras"]['full']["dist"][1]
                fov  = config["cameras"]['full']["fov"][1]
                offsets = config["cameras"]['full']["offset"]
                
                bkgs[b] = torch.ones( (config["render_res"], config["render_res"], 3), device=device)

                limit  =  0.

                prompts.append(config['full_text'])

            elif b == 1  and 'face' in config["options"]:

                elev = np.radians( (config["cameras"]['face']["elev"][0] + config["cameras"]['face']["elev"][1])/2.0 ) 
                azim = np.radians( (config["cameras"]['face']["azim"][0] + config["cameras"]['face']["azim"][1])/2.0 )
                dist = config["cameras"]['face']["dist"][0]
                fov  = config["cameras"]['face']["fov"][0]
                offsets = config["cameras"]['face']["offset"]
                
                bkgs[b] = torch.ones( (config["render_res"], config["render_res"], 3), device=device)

                limit  =  0.

                prompts.append(config['face_text'])

            else:
                elev = np.radians( np.random.uniform( config["cameras"][op_]["elev"][0], config["cameras"][op_]["elev"][1] ) )
                azim = np.radians( np.random.uniform( config["cameras"][op_]["azim"][0], config["cameras"][op_]["azim"][1] ) )
                dist = np.random.uniform( config["cameras"][op_]["dist"][0], config["cameras"][op_]["dist"][1] ) 
                fov = np.random.uniform( config["cameras"][op_]["fov"][0], config["cameras"][op_]["fov"][1] ) 
                offsets = config["cameras"][op_]["offset"]

                bkgs[b] = get_random_bg(device, config["render_res"], config["render_res"]).squeeze(0) if config["rand_bkg"] else torch.ones( (config["render_res"], config["render_res"], 3), device=device)

                limit  =  config["cameras"][op_]["dist"][0] / 4.0
                
                if op_ == "face":
                    prompts.append(config['face_text'])
                elif op_ == "full":
                    prompts.append(config['full_text'])
                elif op_ == "back":
                    prompts.append(config['back_text'])

            proj_mtx = persp_proj(fov)

            # Generate random view
            cam_z = dist * np.cos(elev) * np.sin(azim)
            cam_y = dist * np.sin(elev)
            cam_x = dist * np.cos(elev) * np.cos(azim)
            
            # Random offset
            rand_x = np.random.uniform( -limit, limit )
            rand_y = np.random.uniform( -limit, limit )

            modl = glm.translate(glm.mat4(), glm.vec3(rand_x, rand_y, 0))

            # modl = glm.mat4()
                
            view  = glm.lookAt(
                glm.vec3(cam_x, cam_y, cam_z),
                glm.vec3(0 + offsets[0], 0 + offsets[1], 0 + offsets[2]),
                glm.vec3(0, -1, 0),
            )

            r_mv = view * modl
            r_mv = np.array(r_mv.to_list()).T

            mvp[b]     = np.matmul(proj_mtx, r_mv).astype(np.float32)
            campos[b]  = np.linalg.inv(r_mv)[:3, 3]
            lightpos[b] = campos[b]

        params = {
            'mvp': mvp,
            'lightpos': lightpos,
            'campos': campos,
            'resolution': [config["render_res"], config["render_res"]]
        }

        if i > config["epochs"] // 2:
                
            log_image = render.render_mesh(
                glctx,
                ready_mesh.eval(params),
                mvp,
                campos,
                lightpos,
                2.0,
                config["render_res"],
                background=bkgs
            )

        else:

            with_tex = config["batch_size"] // 2

            with_tex_params = {
                'mvp': mvp[:with_tex],
                'lightpos': lightpos[:with_tex],
                'campos': campos[:with_tex],
                'resolution': [config["render_res"], config["render_res"]]
            }

            no_tex_params = {
                'mvp': mvp[with_tex:],
                'lightpos': lightpos[with_tex:],
                'campos': campos[with_tex:],
                'resolution': [config["render_res"], config["render_res"]]
            }

            with_tex_train_render = render.render_mesh(
                glctx,
                ready_mesh.eval(with_tex_params),
                mvp[:with_tex],
                campos[:with_tex],
                lightpos[:with_tex],
                2.0,
                config["render_res"],
                num_layers=2,
                background=bkgs[:with_tex],
            )

            no_tex_train_render = render.render_mesh(
                glctx,
                notex_mesh.eval(no_tex_params),
                mvp[with_tex:],
                campos[with_tex:],
                lightpos[with_tex:],
                2.0,
                config["render_res"],
                num_layers=2,
                background=bkgs[with_tex:],
            )

            log_image = torch.cat([
                with_tex_train_render,
                no_tex_train_render
            ])
        
        if i % config["log_int"] == 0 and config["debug_log"]:

            if 'full' in config["options"] and 'face' in config["options"]:
                log = torchvision.utils.make_grid(log_image[:2].permute(0, 3, 1, 2))
                if config["plot"]:
                    clear_output()
                    plt.imshow( log.permute(1, 2, 0).detach().cpu().numpy() )
                    plt.show()
                else:
                    torchvision.utils.save_image(log_image.permute(0, 3, 1, 2), os.path.join(config["path"], 'img_%d.png' % i))
                video.ready_image( log.permute(1, 2, 0) )
            elif 'face' in config["options"]:
                log = torchvision.utils.make_grid(log_image[1].unsqueeze(0).permute(0, 3, 1, 2))
                if config["plot"]:
                    clear_output()
                    plt.imshow( log.permute(1, 2, 0).detach().cpu().numpy() )
                    plt.show()
                else:
                    torchvision.utils.save_image(log_image.permute(0, 3, 1, 2), os.path.join(config["path"], 'img_%d.png' % i))
                video.ready_image( log.permute(1, 2, 0) )
            elif 'full' in config["options"]:
                log = torchvision.utils.make_grid(log_image[0].unsqueeze(0).permute(0, 3, 1, 2))
                if config["plot"]:
                    clear_output()
                    plt.imshow( log.permute(1, 2, 0).detach().cpu().numpy() )
                    plt.show()
                else:
                    torchvision.utils.save_image(log_image.permute(0, 3, 1, 2), os.path.join(config["path"], 'img_%d.png' % i))
                video.ready_image( log.permute(1, 2, 0) )

        log_image = resize(
            log_image.permute(0, 3, 1, 2),
            out_shape=(224, 224) if config["CLIP"] != "ViT-L/14@336px" else (336, 336), # resize to clip
            interp_method=lanczos3
        )

        image_embeds = clip_model.image_embeds( log_image )
        texts_embeds = clip_model.text_embeds ( clip_model.text_tokens(prompts_list=prompts) )

        clip_loss = cosine_avg(image_embeds, texts_embeds)
        loss_t = clip_loss
        
        log_str = "CLIP Loss = %.3f" % clip_loss.item()
        if "texture" in config["optim"]:
            t_l = kornia.losses.total_variation( (ready_mesh.eval().material['kd'].data[0] * uv_mask).permute(2, 0, 1))
            
            log_str = "CLIP Loss = %.3f | TVL Tex =  %.3f" % (clip_loss.item(), t_l.item())
            loss_t += config["TV_weight"] * t_l

        if "normal" in config["optim"]:
            t_n = kornia.losses.total_variation( (ready_mesh.eval().material['normal'].data[0] * uv_mask).permute(2, 0, 1))

            log_str = "CLIP Loss = %.3f | TVL Tex =  %.3f | TVL Nrm =  %.3f" % (clip_loss.item(), t_l.item(), t_n.item())
            loss_t += config["TV_weight"] * t_n

        if "specular" in config["optim"]:
            t_s = kornia.losses.total_variation( (ready_mesh.eval().material['ks'].data[0] * uv_mask).permute(2, 0, 1))
            
            log_str = "CLIP Loss = %.3f | TVL Tex =  %.3f | TVL Nrm =  %.3f | TVL Spc =  %.3f" % (clip_loss.item(), t_l.item(), t_n.item(), t_s.item())
            loss_t += config["TV_weight"] * t_s

        for optimizer in optimizers:
            optimizer.zero_grad()
        loss_t.backward()
        for optimizer in optimizers:
            optimizer.step()

        kd_map_opt.clamp_(min=0, max=1)
        normal_map_opt.clamp_(min=-1, max=1)
        ks_map_opt.clamp_rgb_(minR=0, maxR=1, minG=0.5, maxG=1.0, minB=0.0, maxB=1.0)

        torch.clamp(ds_map_opt, min=0, max=1)

        t_loop.set_description(log_str)
        
    obj.write_obj(
        os.path.join(config["path"]),
        ready_mesh.eval()
    )
