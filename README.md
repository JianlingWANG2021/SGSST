<div align="center">

# *SGSST: Scaling Gaussian Splatting Style Transfer*
<font size="4">
Simon Niedermayr &emsp; Josef Stumpfegger  &emsp; Rüdiger Westermann
Bruno Galerne  &emsp; Jianling WANG  &emsp;  Lara Raad  &emsp; Jean-Michel Morel
</font>
<br>

<font size="4">
University of Orleans, CNRS
</font>

<a href="https://www.idpoisson.fr/">Webpage</a> | <a href="https://arxiv.org/abs/">arXiv</a>
</div>

Our implementation is based in the original 3D Gaussian splatting implementation available [here](https://github.com/graphdeco-inria/gaussian-splatting).


Note that we implemented our method based on a former version of this software,
so for consistence we recommend to use the same version.

## Installation

Download 3D Gaussian Splatting
```shell
wget https://github.com/graphdeco-inria/gaussian-splatting/archive/d9fad7b3450bf4bd29316315032d57157e23a515.zip
unzip d9fad7b3450bf4bd29316315032d57157e23a515.zip
mv gaussian-splatting-d9fad7b3450bf4bd29316315032d57157e23a515  SGSST
```

We have used the style transfer loss from "SPST: Scaling Paiting Style Transfer" which should be cloned within the SGSST folder

```shell
cd SGSST
git clone https://github.com/bgalerne/scaling_painting_style_transfer.git
```
Download the original VGG19 weights ```vgg_conv.pth``` from [here](https://drive.google.com/uc?id=1lLSi8BXd_9EtudRbIwxvmTQ3Ms-Qh6C8&export=download)
and save them in the ```scaling_painting_style_transfer/model/```folder.

E.g. using gdown:
>```
>import gdown
>gdown.download("https://drive.google.com/uc?id=1lLSi8BXd_9EtudRbIwxvmTQ3Ms-Qh6C8", "model/vgg_conv.pth")
>```

Copy our stylization script and environment setting  into SGSST/
```shell
cp  stylize.py  environment.yml  SGSST/

```

### Local Setup

Our default installation method is based on Conda package and environment management:
```shell
conda env create --file environment.yml
conda activate SGSST
```

### Download data

These datasets can be download following the instruction of  ARF, 3D gaussian splatting, and Plenoxels :


The SfM data sets from 3D gaussian splatting for Tanks&Temples and Deep Blending can be downloaded [here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip).

The MipNeRF360 scenes are hosted by the paper authors [here](https://jonbarron.info/mipnerf360/). You can put all of these data in the director of datasets.


## Run SGSST:

Stylizing a scene is a two step procedure: First train a realistic 3DGS, then stylize it with our code.

- Run 3DGS reconstruction for the scene.

```shell
python train.py --source_path <path to COLMAP or NeRF Synthetic dataset> --model_path <path of the 3DGS output model>
```

#### Example:
```shell
python train.py  --source_path ./datasets/truck --model_path  ./output/model_truck --iterations 30000 --checkpoint_iterations 30000
```

- Stylize the scene with a given style image

```shell
python stylize.py --source_path <path to COLMAP or NeRF Synthetic dataset> --model_path <path of the stylized 3DGS output model>  --start_checkpoint <path of the 3DGS input model> --style_img  <path of stylized image> --iterations 50000 --resolution 1  
```
#### Example:
```shell
python stylize.py --source_path ./datasets/truck  --model_path output/model_truck_stylized  --start_checkpoint ./output/model_truck/chkpnt30000.pth --style_img  datasets/styles/112.jpg --iterations 50000 --resolution 1  
```
- Render the stylized scene
```shell
python render.py -m <path to the stylized 3DGS model> --source_path  <path to COLMAP or NeRF Synthetic dataset>
```
#### Example:
```shell
python render.py -m output/model_truck_stylized  --source_path  ./datasets/truck
```