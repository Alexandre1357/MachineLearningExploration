import torch
import numpy as np
import flip_loss
import matplotlib.pyplot as plt
import torchmetrics
from IPython.display import clear_output
import time

def psnr(y_pred, y, data_range):
    dim = tuple(range(1, y.ndim))
    mse_error = torch.pow(y_pred - y.view_as(y_pred), 2).mean(dim=dim)
    val = torch.sum(10.0 * torch.log10(data_range**2 / (mse_error + 1e-10))).cpu().item()

    val = val / y.shape[0]
    
    return val

def test_model(model, dataset, loss_fn, albedo_data, arm_data, normal_data):
    model.eval()

    with torch.no_grad():
        test_pred = model(dataset.input_features_tensor)
        test_loss = loss_fn(test_pred, dataset.output_features_tensor)

    test_pred_features = test_pred.cpu().view((dataset.sidelength, dataset.sidelength, -1)).detach()
    
    albedo_pred_raw = test_pred_features[..., :3]
    arm_pred_raw = test_pred_features[..., 3:6]
    normal_pred_raw = test_pred_features[..., 6:]        
    
    albedo_pred = np.clip(albedo_pred_raw.numpy(), 0.0, 1.0)
    arm_pred = np.clip(arm_pred_raw.numpy(), 0.0, 1.0)
    normal_pred = np.clip(normal_pred_raw.numpy(), 0.0, 1.0)

    NCHW_shape = (1, 3, albedo_pred.shape[0], albedo_pred.shape[0])
    HWC_shape = (albedo_pred.shape[0], albedo_pred.shape[0], 1)

    pixels_per_degree = (0.7 * 3840 / 0.7) * np.pi / 180
    qc = 0.7
    qf = 0.5
    pc = 0.4
    pt = 0.95
    eps = 1e-15
    
    albedo_flip = flip_loss.compute_ldrflip(albedo_pred_raw.cuda().detach().permute(2, 0, 1).reshape(NCHW_shape), 
                                            albedo_data.cuda().detach().permute(2, 0, 1).reshape(NCHW_shape),
                                            pixels_per_degree, qc, qf, pc, pt, eps)
    arm_flip = flip_loss.compute_ldrflip(arm_pred_raw.cuda().detach().permute(2, 0, 1).reshape(NCHW_shape), 
                                         arm_data.cuda().detach().permute(2, 0, 1).reshape(NCHW_shape),
                                            pixels_per_degree, qc, qf, pc, pt, eps)
    normal_flip = flip_loss.compute_ldrflip(normal_pred_raw.cuda().detach().permute(2, 0, 1).reshape(NCHW_shape), 
                                            normal_data.cuda().detach().permute(2, 0, 1).reshape(NCHW_shape),
                                            pixels_per_degree, qc, qf, pc, pt, eps)

    albedo_flip = albedo_flip.cpu().permute(0, 2, 3, 1).reshape(HWC_shape)
    arm_flip = arm_flip.cpu().permute(0, 2, 3, 1).reshape(HWC_shape)
    normal_flip = normal_flip.cpu().permute(0, 2, 3, 1).reshape(HWC_shape)
    
    fig, axes = plt.subplots(3, 3, figsize=(18,12))
    axes[0, 0].imshow(albedo_data)
    axes[1, 0].imshow(arm_data)
    axes[2, 0].imshow(normal_data)
    axes[0, 1].imshow(albedo_pred)
    axes[1, 1].imshow(arm_pred)
    axes[2, 1].imshow(normal_pred)
    axes[0, 2].imshow(albedo_flip)
    axes[1, 2].imshow(arm_flip)
    axes[2, 2].imshow(normal_flip)

    plt.show()

    return albedo_pred_raw, arm_pred_raw, normal_pred_raw, albedo_flip, arm_flip, normal_flip

def output_test_data(writer, albedo_pred, arm_pred, normal_pred, albedo_flip, arm_flip, normal_flip, 
                     albedo_data, arm_data, normal_data, current_epoch_arg):
    writer.add_scalar('psnr/albedo', psnr(albedo_pred, albedo_data, 1.0), current_epoch_arg)
    writer.add_scalar('psnr/arm', psnr(arm_pred, arm_data, 1.0), current_epoch_arg)
    writer.add_scalar('psnr/normal', psnr(normal_pred, normal_data, 1.0), current_epoch_arg)

    NCHW_shape = (1, 3, albedo_pred.shape[0], albedo_pred.shape[0])
    
    msssim_albedo = torchmetrics.functional.image.multiscale_structural_similarity_index_measure(
                    albedo_pred.detach().reshape(NCHW_shape), albedo_data.detach().reshape(NCHW_shape),
                    data_range=1.0)

    msssim_arm =    torchmetrics.functional.image.multiscale_structural_similarity_index_measure(
                    arm_pred.reshape(NCHW_shape), arm_data.reshape(NCHW_shape),
                    data_range=1.0)

    msssim_normal = torchmetrics.functional.image.multiscale_structural_similarity_index_measure(
                    normal_pred.reshape(NCHW_shape), normal_data.reshape(NCHW_shape),
                    data_range=1.0)

    writer.add_scalar('ms-ssim/albedo', msssim_albedo, current_epoch_arg)
    writer.add_scalar('ms-ssim/arm', msssim_arm, current_epoch_arg)
    writer.add_scalar('ms-ssim/normal', msssim_normal, current_epoch_arg)

    albedo_image = np.clip(albedo_pred.numpy(), 0.0, 1.0)
    arm_image = np.clip(arm_pred.numpy(), 0.0, 1.0)
    normal_image = np.clip(normal_pred.numpy(), 0.0, 1.0)
    
    writer.add_image('image/albedo', albedo_image, current_epoch_arg, dataformats='HWC')
    writer.add_image('image/arm', arm_image, current_epoch_arg, dataformats='HWC')
    writer.add_image('image/normal', normal_image, current_epoch_arg, dataformats='HWC')

    flip_albedo_image = np.clip(albedo_flip.numpy(), 0.0, 1.0)
    flip_arm_image = np.clip(arm_flip.numpy(), 0.0, 1.0)
    flip_normal_image = np.clip(normal_flip.numpy(), 0.0, 1.0)

    fig_albedo_flip, ax_albedo_flip = plt.subplots(figsize=(10, 10))
    ax_albedo_flip.imshow(flip_albedo_image)
    fig_arm_flip, ax_arm_flip = plt.subplots(figsize=(10, 10))
    ax_arm_flip.imshow(flip_arm_image)
    fig_normal_flip, ax_normal_flip = plt.subplots(figsize=(10, 10))
    ax_normal_flip.imshow(flip_normal_image)

    writer.add_figure('flip/albedo', fig_albedo_flip, current_epoch_arg)
    writer.add_figure('flip/arm', fig_arm_flip, current_epoch_arg)
    writer.add_figure('flip/normal', fig_normal_flip, current_epoch_arg)

def training_loop(model, optimizer, dataloader, writer, loss_fn, num_epochs, current_epoch, summary_interval):
    dataset = dataloader.dataset
    
    data_size = len(dataset)

    output_features = dataset.output_features_tensor.cpu().view((dataset.sidelength, dataset.sidelength, -1)).detach()

    albedo_data = output_features[..., :3]
    arm_data = output_features[..., 3:6]
    normal_data = output_features[..., 6:]

    for epoch_idx in range(num_epochs):        
        current_epoch_loc = current_epoch + epoch_idx
        
        clear_output(wait=True)
        
        albedo_pred, arm_pred, normal_pred, albedo_flip, arm_flip, normal_flip = test_model(model, dataset, loss_fn, albedo_data, arm_data, normal_data)

        if epoch_idx != 0 or current_epoch_loc == 0:
            output_test_data(writer, albedo_pred, arm_pred, normal_pred, albedo_flip, arm_flip, 
                             normal_flip, albedo_data, arm_data, normal_data, current_epoch_loc)
            
        start = time.process_time()

        print(f"epoch: {current_epoch_loc:>3d}")
        
        model.train()
        for batch_idx, (input_tensor, ground_truth) in enumerate(dataloader):
            pred = model(input_tensor)
            loss = loss_fn(pred, ground_truth)

            writer.add_scalar('loss', loss.item(), batch_idx + len(dataloader) * (current_epoch_loc))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % summary_interval == 0:
                print(f"loss: {loss.item():>7f} [{(batch_idx * dataloader.batch_size + len(input_tensor)):>5d}/{data_size:>5d}]")

        writer.flush()

        end = time.process_time()

        writer.add_scalar('epoch time', end - start, current_epoch_loc)

        

    clear_output(wait=True)

    albedo_pred, arm_pred, normal_pred, albedo_flip, arm_flip, normal_flip = test_model(model, dataset, loss_fn, albedo_data, arm_data, normal_data)
    output_test_data(writer, albedo_pred, arm_pred, normal_pred, albedo_flip, arm_flip, 
                     normal_flip, albedo_data, arm_data, normal_data, current_epoch + num_epochs)
    writer.flush()

    print(f"epoch: {(current_epoch + num_epochs):>3d}")