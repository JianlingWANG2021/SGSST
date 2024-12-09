# Source code for 
# SGSST: Scaling Gaussian Splatting Style Transfer
# Bruno Galerne   Jianling WANG   Lara Raad   Jean-Michel Morel
# Preprint: https://arxiv.org/abs/2412.03371
# Project webpage: https://www.idpoisson.fr/galerne/sgsst/
# Github: https://github.com/JianlingWANG2021/SGSST/

import os
from os import makedirs
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

import numpy
import torch.nn.functional as F
import torchvision
from torchvision import transforms


# import from spst:
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from scaling_painting_style_transfer.vgg_by_blocks import VGG_BY_BLOCKS  
from scaling_painting_style_transfer.vgg_gatys import VGG, prep, postp   # code from Gatys' repo https://github.com/leongatys/PytorchNeuralStyleTransfer
from scaling_painting_style_transfer.utils import resize_height_pil, zoomout_pil



try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, fn_style_img, args):

    #get VGG network (from https://github.com/leongatys/PytorchNeuralStyleTransfer)
    vgg = VGG()
    model_dir = os.path.join(os.getcwd(), 'scaling_painting_style_transfer/model')
    #Available here:https://drive.google.com/uc?id=1lLSi8BXd_9EtudRbIwxvmTQ3Ms-Qh6C8&export=download
    vgg.load_state_dict(torch.load(os.path.join(model_dir,'vgg_conv.pth')))


    for param in vgg.parameters():
        param.requires_grad = False
    if torch.cuda.is_available():
        vgg.to("cuda")
        
    style_layers = ['r11','r21','r31','r41', 'r51']
    content_layers = ['r42']
    loss_layers = style_layers + content_layers

    wmeanstd = 1000
    weights_Gram_matrices = [1e3/n**2 for n in [64,128,256,512,512]]
    content_weights = [args.content_weights]
    weights_layers_means = [w*n**2*wmeanstd/100 for n, w in zip([64,128,256,512,512], weights_Gram_matrices)]
    weights_layers_stds = weights_layers_means
    style_weights = (weights_Gram_matrices, weights_layers_means, weights_layers_stds)


    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree )  
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda" )
 

    img_W = scene.getTrainCameras()[0].image_width
    img_H = scene.getTrainCameras()[0].image_height
    
    #determine how many scales used
    if args.resize_images[0]  <= 0 :
         wco, hco = img_W, img_H
         image_sizes = [min(wco,hco)]
         while image_sizes[0]> args.min_size_threshold:
              image_sizes.insert(0,image_sizes[0]//2)
         nbscales = len(image_sizes)
         args.resize_images = [x*1.0/image_sizes[-1]  for x in image_sizes]
         args.weight_images = [1.]* nbscales
    
    if args.optimize_size <=0 :
         args.optimize_size = args.resize_images[0]

    style_img_pil_hr = Image.open(fn_style_img).convert("RGB")

    hco, wco = img_H, img_W
    if min(style_img_pil_hr.width, style_img_pil_hr.height) > 1.1*min(hco,wco):
          ratio = float(min(hco,wco))/float(min(style_img_pil_hr.width, style_img_pil_hr.height))
    else:
          ratio = 1

    new_height = round(float(style_img_pil_hr.height)*ratio)
    style_img_pil_hr = resize_height_pil(style_img_pil_hr, new_height)
    style_img_pil_hr.save(  os.path.join(args.model_path, "style_image.png") )

    style_targets_by_blocks_arr=[]

    for i in range(len( args.resize_images ) ):
          style_image = resize_height_pil(style_img_pil_hr, round(new_height * args.resize_images[i]) )
          style_image.save(  os.path.join(args.model_path, "style_image_{}.png".format( int(1/args.resize_images[i]) )) ) 
          style_image = prep(style_image).unsqueeze(0).to("cuda")
          vgg_blocks_style_img = VGG_BY_BLOCKS(vgg, style_image, style_layers, content_layers = [])
          style_targets_by_blocks_arr.append( vgg_blocks_style_img.global_Gram_matrices_means_and_stds() )

    if args.optimize_iteration > 0:
          style_image = resize_height_pil(style_img_pil_hr, round(new_height * args.optimize_size) )
          style_image = prep(style_image).unsqueeze(0).to("cuda")
          vgg_blocks_style_img = VGG_BY_BLOCKS(vgg, style_image, style_layers, content_layers = [])
          style_targets_by_blocks_gradopt =  vgg_blocks_style_img.global_Gram_matrices_means_and_stds() 
    

    prep_tensor_img = transforms.Compose([
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
                                                std=[1,1,1]),
                           transforms.Lambda(lambda x: x.mul_(255)),
                          ])

    
    l = [
            {'params': [gaussians._features_dc], 'lr': args.feature_lr, "name": "f_dc"},
        ]

    gaussians.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        image.retain_grad()

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        
        # optimize the style transfer loss at the lowest resolution for first part of iterations
        if args.optimize_iteration >0 and  iteration - first_iter < args.optimize_iteration :
           
                    rgb_pred = prep_tensor_img(image).unsqueeze(0)
                    rgb_pred.retain_grad()
                    rgb_gt = prep_tensor_img(gt_image).unsqueeze(0)

                    rgb_pred= F.interpolate( rgb_pred, size=None, scale_factor=args.optimize_size , mode="bilinear")
                    rgb_gt  = F.interpolate( rgb_gt,   size=None, scale_factor=args.optimize_size,  mode="bilinear")

                    vgg_blocks_content_img = VGG_BY_BLOCKS(vgg, rgb_gt, style_layers,
                                                           content_layers = content_layers,
                                                           verbose_mode=False)

                    content_targets_by_blocks = vgg_blocks_content_img.compute_content_layer_by_blocks()

                    vgg_blocks_opt_img = VGG_BY_BLOCKS(vgg, rgb_pred, style_layers,
                                           content_layers = content_layers,
                                           verbose_mode=False)
                    loss = vgg_blocks_opt_img.global_content_plus_Gram_means_stds_loss_with_gradient(
                                                        style_targets_by_blocks_gradopt, style_weights,
                                                        content_targets_by_blocks, content_weights)

                    torch.autograd.backward(rgb_pred, grad_tensors=rgb_pred.grad)
        else:

             #SOS: Simultaneously Optimized Scales loss
             for i_resize in range(len(args.resize_images) ):
                    if args.resize_images[i_resize] != 1:
                       image_s     = F.interpolate( image.unsqueeze(0),      size=None, scale_factor=args.resize_images[i_resize], mode="bilinear").squeeze()
                       gt_image_s  = F.interpolate( gt_image.unsqueeze(0),   size=None, scale_factor=args.resize_images[i_resize], mode="bilinear").squeeze()
                    else:
                       image_s = image
                       gt_image_s = gt_image
 
                    rgb_pred = prep_tensor_img(image_s).unsqueeze(0)
                    rgb_pred.retain_grad()
                    rgb_gt = prep_tensor_img(gt_image_s).unsqueeze(0)
 
                    vgg_blocks_content_img = VGG_BY_BLOCKS(vgg, rgb_gt, style_layers,
                                                           content_layers = content_layers,
                                                           verbose_mode=False)
                    content_targets_by_blocks = vgg_blocks_content_img.compute_content_layer_by_blocks()
 
                    vgg_blocks_opt_img = VGG_BY_BLOCKS(vgg, rgb_pred, style_layers,
                                           content_layers = content_layers,
                                           verbose_mode=False)
 
                    loss_hand = vgg_blocks_opt_img.global_content_plus_Gram_means_stds_loss_with_gradient(
                                                        style_targets_by_blocks_arr[i_resize], style_weights,
                                                        content_targets_by_blocks, content_weights)
                    torch.autograd.backward(rgb_pred, grad_tensors=rgb_pred.grad * args.weight_images[i_resize]/numpy.sum(args.weight_images), retain_graph=True, inputs=[image])

                    if i_resize == 0:
                       loss = loss_hand * args.weight_images[i_resize]
                    else:
                       loss = loss + loss_hand * args.weight_images[i_resize] 

             loss = loss/numpy.sum(args.weight_images)
             torch.autograd.backward(image, grad_tensors=image.grad )

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")



def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)


    parser.add_argument("--style_img",      type=str,   default="datasets/styles/29.jpg")
    parser.add_argument("--content_weights", type=float, default=1.0)

    parser.add_argument("--resize_images",   nargs="+", type=float,   default=[-1], help="factors of resizing images for optimisation ")  
    parser.add_argument("--weight_images",   nargs="+", type=float,   default=[1.0], help="weights for the loss of different resizing images")  
    parser.add_argument("--min_size_threshold", type=int,   default=511, help="minimun pix size of image for automatically scaling")  

    parser.add_argument("--optimize_size",      type=float, default=-1,  help="the minumum factor of resizing images for optimising training. <0 means using minimum one of resize_image")  
    parser.add_argument("--optimize_iteration", type=int,   default=10000, help="iteration step for optimising training, setting <=0 for ignore pre-optimisation")  


    args = parser.parse_args(sys.argv[1:])


    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)

    args.checkpoint_iterations.append(args.iterations)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.style_img, args)

    # All done
    print("\nTraining complete.")


