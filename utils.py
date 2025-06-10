import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import wandb
import torch.optim as optim
from torch.distributions import MultivariateNormal

# Set MPS fallback to enable operations not supported natively on Apple Silicon
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import bopt
import h5py as h5

# Let's write a function that shifts a stimulus by a certain number of pixels
def shift_stimulus(stimulus, dy, dx):
    """
    Shift the stimulus by a certain number of pixels. We fill in the empty pixels with zeros.
    This function shifts the stimulus in the y and x directions based on the provided dy and dx tensors.
    
    Parameters:
    - stimulus: A torch.Tensor of shape (batch, height, width) representing the stimulus.
    - dy: The number of pixels to shift in the y direction (positive for up, negative for down). torch.Tensor of shape (N)
    - dx: The number of pixels to shift in the x direction (positive for right, negative for left). torch.Tensor of shape (N)
    Returns:
    - shifted_stimulus: A torch.Tensor of shape (batch, N, height, width) representing the shifted stimulus.
    """
    batch, height, width = stimulus.shape
    N = dy.shape[0]
    
    # Create an empty tensor to hold the shifted stimuli
    shifted_stimulus = torch.zeros((batch, N, height, width), dtype=stimulus.dtype, device=stimulus.device)
    
    for i in range(N):
        # Flip the sign of the shifts and convert to integers
        y_shift = int(-dy[i].item())
        x_shift = int(-dx[i].item())
        
        # Calculate the new coordinates
        if y_shift >= 0:
            y_start = y_shift
            y_end = height
            y_new_start = 0
            y_new_end = height - y_shift
        else:
            y_start = 0
            y_end = height + y_shift
            y_new_start = -y_shift
            y_new_end = height
        
        if x_shift >= 0:
            x_start = x_shift
            x_end = width
            x_new_start = 0
            x_new_end = width - x_shift
        else:
            x_start = 0
            x_end = width + x_shift
            x_new_start = -x_shift
            x_new_end = width
        
        # Fill the shifted stimulus with the original stimulus values
        shifted_stimulus[:, i, y_new_start:y_new_end, x_new_start:x_new_end] = stimulus[:, y_start:y_end, x_start:x_end]
    
    return shifted_stimulus

def model_log_lkhd(x, y, z_grid_masked, model, var=131.6, batch_size=256, device='cpu', requires_grad=False):
    """
    Compute the log likelihood of the model given the input data.
    Assume the eye position doens't change during the window.

    Parameters:
    x (torch.Tensor): Originally shifted stimulus. Shape (T, H, W).
    y (torch.Tensor): Ground truth neural activity. Shape (N).
    z_grid_masked (torch.Tensor): Masked grid of latent eye position. Shape (M, 2).
    model (torch.nn.Module): The trained model.
    var (float): Variance for Gaussian noise assumption.
    batch_size (int): Batch size for processing z_grid_masked positions.
    device (torch.device): The device to run the model on.
    requires_grad (bool): Whether to compute gradients. If False, uses torch.no_grad().
    """
    def _compute_log_likelihood():
        x_device = x.to(device).float()
        y_device = y.to(device).float()
        z_grid_masked_device = z_grid_masked.to(device).float()
        
        M = z_grid_masked_device.shape[0]
        log_likelihoods = []
        
        # Process z_grid_masked in batches
        for i in range(0, M, batch_size):
            end_idx = min(i + batch_size, M)
            z_batch = z_grid_masked_device[i:end_idx]  # (batch_size, 2)
            
            # Shift the stimulus for this batch
            x_shifted = shift_stimulus(x_device, z_batch[:, 0], z_batch[:, 1])  # (T, batch_size, H, W)
            x_shifted = x_shifted.transpose(0, 1)  # (batch_size, T, H, W)
            
            # Pass through the model
            y_pred = model(x_shifted)  # (batch_size, N)
            
            # Compute the log likelihood assuming Gaussian noise
            batch_log_likelihood = -0.5 * torch.sum((y_pred - y_device.unsqueeze(0)) ** 2 / var, dim=1)
            log_likelihoods.append(batch_log_likelihood)
        
        # Concatenate all batch results
        log_likelihood = torch.cat(log_likelihoods, dim=0)
        return log_likelihood
    
    if requires_grad:
        return _compute_log_likelihood()
    else:
        with torch.no_grad():
            return _compute_log_likelihood()

def E_step(x, y, z_hat, theta, z_sample, sigma=2.0, y_var=131.6, device='cpu'):
    """
    E-step: Infer z based on current guess of parameters

    Parameters:
    x, y     : stimulus and observed neural response, inputs to model_lkhd
    z_hat    : torch tensor, observed eye trace location, shape [2]
    theta    : model parameters
    z_sample : np.ndarray of shape [H, W], count of times each pixel was sampled
    sigma    : float, std deviation for Gaussian likelihood p(z_hat | z)

    Returns:
    q_z : np.ndarray of shape [H, W], posterior probability of sampling at each pixel
    """
# H, W = z_sample.shape
    # cov = torch.eye(2) * sigma**2
    # mvn = MultivariateNormal(z_hat, cov)

    # q_z = np.zeros((H, W))
    # total_samples = z_sample.sum()
    # for h in range(H):
    #     for w in range(W):
    #         count = z_sample[h, w]
    #         if count == 0:
    #             continue
    #         else:
    #             z = torch.tensor([h, w])
    #             p_z_hat_given_z = torch.exp(mvn.log_prob(z)).item()
    #             lkhd = model_lkhd(x, y, z, theta)
    #             q_z[h, w] = lkhd * p_z_hat_given_z * count
    # q_z /= q_z.sum()
    # return q_z

    z_hat = z_hat.to(device)
    x = x.to(device)
    y = y.to(device)
    z_sample = z_sample.to(device)
    theta = theta.to(device)  # Ensure the model is on the correct device
    
    # Ensure input tensors are float32
    x = x.float()
    y = y.float()
    z_hat = z_hat.float()
    
    H, W = z_sample.shape
    h_coords, w_coords = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32), 
                                       torch.arange(W, device=device, dtype=torch.float32), 
                                       indexing='ij')
    z_grid = torch.stack([h_coords, w_coords], dim=-1).reshape(-1, 2)  # [H*W,2]
    z_sample_flat = z_sample.flatten()  # [N] - z_sample is already a tensor on the device

    # Mask only sampled pixels
    mask = z_sample_flat > 0  # [H*W]
    z_grid_masked = z_grid[mask]  # [M,2]
    sample_weights = z_sample_flat[mask]  # [M]
    total_samples = sample_weights.sum()

    mvn = MultivariateNormal(z_hat, torch.eye(2, device=device) * sigma**2)
    log_p_z_hat_given_z_masked = mvn.log_prob(z_grid_masked)
    log_lkhd_masked = model_log_lkhd(x, y, z_grid_masked, theta, var=y_var, device=device)  # Pass device to model_log_lkhd
    log_q_z = log_lkhd_masked + log_p_z_hat_given_z_masked + sample_weights.log()
    q_z_unnorm = torch.exp(log_q_z - log_q_z.max())

    q_flat = torch.zeros_like(z_sample_flat)
    q_flat[mask] = q_z_unnorm
    q_z = q_flat.reshape(H, W)
    q_z /= q_z.sum()
    
    # Expand masked results back to full grid for consistent output shapes
    log_p_z_hat_given_z_full = torch.full((H * W,), float('-inf'), device=device)
    log_p_z_hat_given_z_full[mask] = log_p_z_hat_given_z_masked
    
    log_lkhd_full = torch.full((H * W,), float('-inf'), device=device)
    log_lkhd_full[mask] = log_lkhd_masked
    
    return q_z, log_lkhd_full, log_p_z_hat_given_z_full

def sample_z(q_z, num_samples):
    """
    Sample z from q(z)

    Parameters:
    q_z         : np.ndarray of shape [H, W], posterior probability of sampling at each pixel
    num_samples : int, number of samples

    Returns:
    z_sample    : np.ndarray of shape [H, W], count of times each pixel was sampled
    """
    H, W = q_z.shape
    flat_q = q_z.flatten()
    # flat_q /= flat_q.sum()
    sampled_indices = np.random.choice(len(flat_q), size=num_samples, p=flat_q)
    counts_flat = np.bincount(sampled_indices, minlength=H * W)
    z_sample = counts_flat.reshape(H, W)
    return z_sample

def M_step(x, y, q_z, model, y_var=131.6, batch_size=256, lr=1e-3, steps=10, device='cpu'):
    """
    M-step: update theta to maximize E_q(z)[log p(y | z, x, theta)]

    Parameters:
        x, y       : stimulus and observed neural response, inputs to model_lkhd
        q_z        : np.ndarray of shape [H, W], posterior probability over latent z
        model      : torch.nn.Module or torch.nn.Parameter (parameters to optimize)
        batch_size : int, batch size for processing z locations to avoid OOM
        lr         : learning rate
        steps      : number of gradient descent steps
        device     : device to run computations on

    Returns:
        Updated model, list of ELBOs, current learning rate
    """
# H, W = q_z.shape
    # optimizer = optim.Adam(theta.parameters(), lr=lr)

    # for step in range(steps):
    #     optimizer.zero_grad()
    #     loss = 0
    #     for h in range(H):
    #         for w in range(W):
    #             weight = q_z[h, w]
    #             if weight == 0:
    #                 continue
    #             else:
    #                 z = torch.tensor([h, w])
    #                 log_lkhd = model_log_lkhd(x, y, z, theta)
    #                 loss -= weight * log_lkhd
    #     loss.backward()
    #     optimizer.step()
    # return theta

    x = x.to(device)
    y = y.to(device)
    x = x.float()
    y = y.float()
    H, W = q_z.shape
    
    # Fix tensor construction warning
    if isinstance(q_z, torch.Tensor):
        q_z_flat = q_z.clone().detach().flatten().to(device)
    else:
        q_z_flat = torch.tensor(q_z, device=device).flatten()
    
    h_coords, w_coords = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32), 
                                       torch.arange(W, device=device, dtype=torch.float32), 
                                       indexing='ij')
    z_grid = torch.stack([h_coords, w_coords], dim=-1).reshape(-1, 2)  # [H*W,2]

    mask = q_z_flat > 0
    z_grid_masked = z_grid[mask]
    sample_weights = q_z_flat[mask]

    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       mode='min',
                                                       factor=0.2,
                                                       patience=1)
    model.train()  # Set model to training mode
    elbos = []
    
    # Calculate number of batches
    num_samples = z_grid_masked.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
    
    for step in range(steps):
        optimizer.zero_grad()
        total_loss = 0.0
        
        # Process in batches
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            
            # Get batch data
            z_batch = z_grid_masked[start_idx:end_idx]
            weights_batch = sample_weights[start_idx:end_idx]
            
            # Compute log likelihood for this batch
            log_lkhd_batch = model_log_lkhd(x, y, z_batch, model, var=y_var, device=device, requires_grad=True)
            batch_loss = -(weights_batch * log_lkhd_batch).sum()
            
            # Backward pass (accumulates gradients)
            batch_loss.backward()
            total_loss += batch_loss.item()
        
        print(f"Step {step+1}/{steps}, Loss: {total_loss:.4f}")
        elbos.append(-total_loss)
        scheduler.step(total_loss)
        optimizer.step()
    model.eval()  # Set model back to evaluation mode
    
    # Get the current learning rate from the optimizer
    current_lr = optimizer.param_groups[0]['lr']
    return model, elbos, current_lr

def run_EM(x, y, z_hat, model, q_z_prior, sigma=2.0, y_var=131.6, batch_size=256, EM_steps=3, train_steps=10, lr=1e-3, device='cpu'):
    """
    Run the EM algorithm to estimate the posterior distribution of z and update model parameters. We will run one more E-step than M-step.

    Parameters:
    x (torch.Tensor): Stimulus tensor of shape (T, H, W).
    y (torch.Tensor): Neural activity tensor of shape (N).
    z_hat (torch.Tensor): Initial guess for eye position, shape (2,).
    model (torch.nn.Module): The model to optimize.
    q_z_prior (np.ndarray): Prior distribution over z, shape (H, W).
    sigma (float): Standard deviation for Gaussian likelihood.
    y_var (float): Variance for the neural activity likelihood.
    batch_size (int): Batch size for processing z locations.
    EM_steps (int): Number of EM iterations.
    train_steps (int): Number of training steps in M-step.
    lr (float): Learning rate for optimization.
    device (str or torch.device): Device to run computations on.

    Returns:
    posteriors (list): List of posterior distributions over z after each E step.
    log_lkhds (list): List of log likelihoods after each E step.
    log_p_z_hat_given_z_list (list): List of log p(z_hat | z) values after each E step.
    model (torch.nn.Module): Updated model after M-step.
    elbos (list): List of ELBOs after each M-step.
    """
    
    # Convert inputs to tensors and move to device
    x = x.to(device).float()
    y = y.to(device).float()
    z_hat = z_hat.to(device).float()
    
    # Convert prior to tensor if it's numpy array
    if isinstance(q_z_prior, np.ndarray):
        q_z_tensor = torch.tensor(q_z_prior, dtype=torch.float32, device=device)
    else:
        q_z_tensor = q_z_prior.to(device).float()
    
    # Initialize lists to store results
    posteriors = []
    log_lkhds = []
    log_p_z_hat_given_z_list = []
    all_elbos = []
    
    # Current learning rate (starts with the provided lr)
    current_lr = lr
    
    # Run EM iterations
    for em_step in range(EM_steps):
        print(f"\nEM Step {em_step + 1}/{EM_steps}")
        
        # E-step: Infer latent variable z based on current guess of parameters
        q_z, log_lkhd_full, log_p_z_hat_given_z_full = E_step(
            x, y, z_hat, model, q_z_tensor, 
            sigma=sigma, y_var=y_var, device=device
        )
        
        # Store results
        posteriors.append(q_z.clone())
        log_lkhds.append(log_lkhd_full.clone())
        log_p_z_hat_given_z_list.append(log_p_z_hat_given_z_full.clone())
        
        # M-step: Update parameters θ to maximize the expected log-likelihood
        model, elbos, current_lr = M_step(
            x, y, q_z, model, 
            y_var=y_var, batch_size=batch_size, 
            lr=current_lr, steps=train_steps, device=device
        )
        
        # Store ELBOs from this M-step
        all_elbos.extend(elbos)
        
        # Update q_z_tensor for next iteration (use current posterior as prior)
        q_z_tensor = q_z.clone()
        
        print(f"Updated learning rate: {current_lr:.6f}")
    
    # Run one final E-step (as mentioned in docstring: "one more E-step than M-step")
    print(f"\nFinal E-step")
    q_z_final, log_lkhd_final, log_p_z_hat_given_z_final = E_step(
        x, y, z_hat, model, q_z_tensor,
        sigma=sigma, y_var=y_var, device=device
    )
    
    # Store final results
    posteriors.append(q_z_final.clone())
    log_lkhds.append(log_lkhd_final.clone())
    log_p_z_hat_given_z_list.append(log_p_z_hat_given_z_final.clone())
    
    return posteriors, log_lkhds, log_p_z_hat_given_z_list, model, all_elbos

def batch_E_step(indices, dataset, model, z_hat, q_z_prior, sigma=2.0, y_var=131.6, device='cpu'):
    """
    Run E-step for multiple dataset indices and store results for plotting.
    
    Parameters:
    indices (list or array): List of dataset indices to process
    dataset (torch.utils.data.Dataset): The dataset to sample from
    model (torch.nn.Module): The trained model
    z_hat (torch.Tensor): Observed eye trace location, shape [2]
    q_z_prior (torch.Tensor): Prior distribution over z, shape (H, W)
    sigma (float): Standard deviation for Gaussian likelihood p(z_hat | z)
    y_var (float): Variance for the neural activity likelihood
    device (str or torch.device): Device to run computations on
    
    Returns:
    posteriors (torch.Tensor): Tensor of posterior distributions, shape (len(indices), H, W)
    log_lkhds (torch.Tensor): Tensor of log likelihoods, shape (len(indices), H*W)
    log_p_z_hat_given_z_list (torch.Tensor): Tensor of log p(z_hat | z), shape (len(indices), H*W)
    """
    
    # Convert inputs to appropriate types
    if not isinstance(indices, (list, tuple)):
        indices = [indices]
    
    z_hat = z_hat.to(device).float()
    
    # Convert prior to tensor if it's numpy array
    if isinstance(q_z_prior, np.ndarray):
        q_z_tensor = torch.tensor(q_z_prior, dtype=torch.float32, device=device)
    else:
        q_z_tensor = q_z_prior.to(device).float()
    
    H, W = q_z_tensor.shape
    
    # Initialize storage tensors
    posteriors = torch.zeros((len(indices), H, W), device=device)
    log_lkhds = torch.zeros((len(indices), H * W), device=device)
    log_p_z_hat_given_z_list = torch.zeros((len(indices), H * W), device=device)
    
    print(f"Running batch E-step for {len(indices)} samples...")
    
    # Process each index
    for i, idx in enumerate(indices):
        print(f"Processing sample {i+1}/{len(indices)} (dataset index {idx})")
        
        # Get data from dataset
        data = dataset[idx]
        x = data[0]  # stimulus: (T, H, W)
        y = data[1]  # response: (N,)
        
        # Move to device and ensure float32
        x = x.to(device).float()
        y = y.to(device).float()
        
        # Run E-step for this sample
        q_z, log_lkhd_full, log_p_z_hat_given_z_full = E_step(
            x, y, z_hat, model, q_z_tensor, 
            sigma=sigma, y_var=y_var, device=device
        )
        
        # Store results
        posteriors[i] = q_z
        log_lkhds[i] = log_lkhd_full
        log_p_z_hat_given_z_list[i] = log_p_z_hat_given_z_full
        
        # Print some statistics
        print(f"  Posterior sum: {q_z.sum().item():.6f}, max: {q_z.max().item():.6f}")
        
        # Use current posterior as prior for next iteration (optional)
        # q_z_tensor = q_z.clone()
    
    print("Batch E-step completed!")
    return posteriors, log_lkhds, log_p_z_hat_given_z_list

# Plotting code

def plot_frame(frame, flip_y=False, title=None, cmap='gray', vmin=None, vmax=None):
    """
    Plot a single frame of the stimulus.
    
    Parameters:
    - frame: A 2D numpy array or torch tensor representing the frame to plot.
    - flip_y: If True, flip the y-axis.
    - title: Title for the plot.
    - cmap: Colormap to use for the plot.
    - vmin: Minimum value for color scaling.
    - vmax: Maximum value for color scaling.
    """
    if isinstance(frame, torch.Tensor):
        frame = frame.cpu().numpy()
    
    plt.imshow(frame, cmap=cmap, vmin=vmin, vmax=vmax)
    if flip_y:
        plt.gca().invert_yaxis()
    if title:
        plt.title(title)
    
    # Show tick numbers but no axis labels
    ax = plt.gca()
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    plt.show()

def plot_stimulus(stim, flip_y=False, title=None, cmap='gray', vmin=None, vmax=None):
    """
    Plot a stimulus with multiple frames.
    
    Parameters:
    - stim: A 3D numpy array or torch tensor representing the stimulus (T, H, W).
    - flip_y: If True, flip the y-axis.
    - title: Title for the plot.
    - cmap: Colormap to use for the plot.
    - vmin: Minimum value for color scaling.
    - vmax: Maximum value for color scaling.
    """
    if isinstance(stim, torch.Tensor):
        stim = stim.cpu().numpy()
    
    T, H, W = stim.shape
    
    # Calculate grid dimensions for roughly square layout
    if T == 1:
        nrows, ncols = 1, 1
    else:
        ncols = int(np.ceil(np.sqrt(T)))
        nrows = int(np.ceil(T / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    
    # Handle single subplot case
    if T == 1:
        axes = [axes]
    elif nrows == 1 or ncols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i in range(T):
        axes[i].imshow(stim[i], cmap=cmap, vmin=vmin, vmax=vmax)
        if flip_y:
            axes[i].invert_yaxis()
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        if title:
            axes[i].set_title(f"{title} Frame {i+1}")
    
    # Hide unused subplots
    for i in range(T, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout(pad=0.5, w_pad=0.2, h_pad=0.2)
    plt.show()

def show_stimulus_movie(stim, fps=50, figsize=(8, 6), flip_y=False, title=None, cmap='gray', vmin=None, vmax=None):
    """
    Display a stimulus as an interactive frame viewer that works reliably in notebooks.
    Falls back to showing a grid of frames if widgets are not available.
    
    Parameters:
    - stim: A 3D numpy array or torch tensor representing the stimulus (T, H, W).
    - fps: Frames per second (for reference, not used in interactive mode).
    - figsize: Figure size tuple.
    - flip_y: If True, flip the y-axis.
    - title: Title for the plot.
    - cmap: Colormap to use for the plot.
    - vmin: Minimum value for color scaling.
    - vmax: Maximum value for color scaling.
    """
    if isinstance(stim, torch.Tensor):
        stim = stim.cpu().numpy()
    
    T, H, W = stim.shape
    
    # Try widget-based animation first
    try:
        from ipywidgets import interact, IntSlider
        from IPython.display import display, clear_output
        
        # Create interactive widget
        def show_frame(frame_idx):
            plt.figure(figsize=figsize)
            plt.imshow(stim[frame_idx], cmap=cmap, vmin=vmin, vmax=vmax)
            if flip_y:
                plt.gca().invert_yaxis()
            plt.xlabel('')
            plt.ylabel('')
            if title:
                plt.title(f"{title} - Frame {frame_idx+1}/{T}")
            else:
                plt.title(f"Frame {frame_idx+1}/{T}")
            plt.tight_layout()
            plt.show()
        
        # Create slider widget
        frame_slider = IntSlider(
            value=0,
            min=0,
            max=T-1,
            step=1,
            description='Frame:',
            continuous_update=True
        )
        
        print(f"Interactive slider for {T} frames (use slider below to navigate)")
        interact(show_frame, frame_idx=frame_slider)
        
    except ImportError:
        # Fallback: show grid of frames
        print(f"Widget not available, showing all {T} frames in a grid")
        plot_stimulus(stim, flip_y=flip_y, title=title, cmap=cmap, vmin=vmin, vmax=vmax)

def plot_distribution(p, flip_y=False, title=None, cmap='viridis', vmin=None, vmax=None):
    """
    Plot a 2D distribution as a heatmap.
    
    Parameters:
    - p: A 2D numpy array or torch tensor representing the distribution.
    - flip_y: If True, flip the y-axis.
    - title: Title for the plot.
    - cmap: Colormap to use for the plot.
    - vmin: Minimum value for color scaling.
    - vmax: Maximum value for color scaling.
    """
    if isinstance(p, torch.Tensor):
        p = p.cpu().numpy()
    
    plt.figure(figsize=(8, 6))
    plt.imshow(p, cmap=cmap, vmin=vmin, vmax=vmax)
    if flip_y:
        plt.gca().invert_yaxis()
    plt.colorbar(label='Probability')
    if title:
        plt.title(title)
    plt.xlabel('Width (pixels)')
    plt.ylabel('Height (pixels)')
    plt.show()