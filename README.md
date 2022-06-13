# CLIPMesh-SMPLX

The following is a research project initially inspired by [CLIPMatrix](https://arxiv.org/abs/2109.12922). Following the release of [CLIP-Actor](https://arxiv.org/abs/2206.04382) and [AvatarCLIP](https://hongfz16.github.io/projects/AvatarCLIP.html) I've decided to just release my code the works are all quite similar. The main difference is that this uses meshes which are must faster and less memory-intensive

## License

### SMPL-X

This codebase uses SMPL-X Models and the smplx pip library. By using this code you agree to the [SMPL-X Model License](https://smpl-x.is.tue.mpg.de/modellicense.html) and [smplx License](https://github.com/vchoutas/smplify-x#license)

To run the code you will need to download 'SMPL-X v1.1' (830 MB) models from [here](https://smpl-x.is.tue.mpg.de/download.php)

### Renderer

This code base relies on [nvdiffmodeling](https://github.com/NVlabs/nvdiffmodeling) and in turn [nvdiffrast](https://nvlabs.github.io/nvdiffrast/#licenses).

## Demos

Get started by testing out the features through the collabs

|<img src="./assets/README/single.gif" width="310"/>|<img src="./assets/README/expressions.gif" width="310"/>|
:--------------------------------------------------:|:--------------------------------------------------:|
| [Create a character from just a text prompt]()  | [Change an expression with text prompt]() |

|<img src="./assets/README/pose.gif" width="310"/>|<img src="./assets/README/clipmatrix.gif" width="310"/>|
:--------------------------------------------------:|:--------------------------------------------------:|
| [Pose with a description (⚠️ WIP)]()             | [Create fantastic creatures like CLIPMatrix]() |

## Setup

```
git clone
cd 
pip install virtualenv
virtualenv ENV
source ENV/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Then download the 'SMPL-X v1.1' (830 MB) models from [here](https://smpl-x.is.tue.mpg.de/download.php) and place the extracted folder in the root of this project. So that you get the following directory
```
CLIPMesh-Avatars
│
└───models
│   └───smplx
│       │   SMPLX_NEUTRAL.npz
│       │   SMPLX_NEUTRAL.pkl
│       │   ....
│       │   ....
```

To replicate the demos you can use the configs provided.

```
# For a single character generation
python main.py --path=configs/single.yaml

# For a single expression generation
python main.py --path=configs/expression.yaml

# For a single pose generation
python main.py --path=configs/pose.yaml

# For a CLIPMatrix type generation
python main.py --path=configs/pose.yaml

# For something else use params
python main.py \
--optim   body expression texture normal \
--options face full back \
--epochs  1000 \
--gpu     0 \
--face_text "a thin 3D render of the face of a James Bond" \
--full_text "a thin 3D render of the James Bond" \
--back_text "a thin 3D render of the back of James Bond" \
--debug_log true
--log_int 250
```