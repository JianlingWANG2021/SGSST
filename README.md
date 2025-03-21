<div align="center">

# *SGSST: Scaling Gaussian Splatting Style Transfer*
<font size="4">
 <a href="https://idpoisson.fr/galerne/">Bruno Galerne</a><sup>1,2</sup>  &emsp; <a href='https://github.com/JianlingWANG2021/SGSST'>Jianling WANG</a><sup>1</sup>  &emsp;  <a href="http://dev.ipol.im/~lraad/">Lara Raad</a><sup>3</sup>  &emsp; <a href="https://sites.google.com/site/jeanmichelmorelcmlaenscachan/">Jean-Michel Morel</a><sup>4</sup>
</font>
<br><br>

<font size="4">
<sup>1</sup>Université d'Orléans, Université de Tours, CNRS, IDP, UMR 7013, Orléans, France
<br>
<sup>2</sup>Institut Universitaire de France (IUF)
<br>
<sup>3</sup>Instituto de Ingeniería Eléctrica, Facultad de Ingeniería, Universidad de la República
<br>
<sup>4</sup>City University of Hong Kong
<br>
<strong>Accepted at CVPR 2025</strong>
</font>

<a href="https://www.idpoisson.fr/galerne/sgsst">Webpage</a> | <a href="https://arxiv.org/abs/2412.03371">arXiv</a>

</div>

Our implementation is based on the original 3D Gaussian splatting implementation available [here](https://github.com/graphdeco-inria/gaussian-splatting).

Note that we implemented our method based on a former version of this software,
so for consistence we recommend to use the same version.

## Installation

Download 3D Gaussian Splatting
```shell
wget https://github.com/graphdeco-inria/gaussian-splatting/archive/d9fad7b3450bf4bd29316315032d57157e23a515.zip
unzip d9fad7b3450bf4bd29316315032d57157e23a515.zip
mv gaussian-splatting-d9fad7b3450bf4bd29316315032d57157e23a515  SGSST
```

We have used the style transfer loss from [SPST: Scaling Paiting Style Transfer](https://github.com/bgalerne/scaling_painting_style_transfer) which should be cloned within the SGSST folder

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

These datasets can be downloaded following the instruction of  [ARF](https://www.cs.cornell.edu/projects/arf), [3D gaussian splatting](https://github.com/graphdeco-inria/gaussian-splatting), and [Plenoxels](https://github.com/sxyu/svox2) :


The SfM data sets from 3D gaussian splatting for Tanks&Temples and Deep Blending can be downloaded [here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip).

The MipNeRF360 scenes are hosted by the paper authors [here](https://jonbarron.info/mipnerf360/). You can put all of these data into the ```datasets``` folder.


## Run SGSST:

Stylizing a scene is a two step procedure: First train a realistic 3DGS, then stylize it with our code.

- Run 3DGS reconstruction for the scene.

```shell
python train.py --source_path <path to COLMAP or NeRF Synthetic dataset>  \
                --model_path <path of the 3DGS output model> \
                --resolution 1  
```

#### Example:
```shell
python train.py  --source_path ./datasets/truck \
                 --model_path  ./output/model_truck \
                 --iterations 30000 \
                 --checkpoint_iterations 30000 \
                 --resolution 1  
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
