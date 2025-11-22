"""
Video Style Transfer Program - VGG19-based Color Enhanced Version (RAFT Optical Flow Enhanced)
Command-line version without GUI (configurable resolution)
Usage: python cv_off.py --video input.mp4 --style style.jpg --output output.mp4 [--resolution 256|original] [--raft small|large]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
import os
import sys
import argparse

# Try to import scipy for flow smoothing (optional)
try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# -------------------------- RAFT Optical Flow Configuration --------------------------
# RAFT uses torchvision built-in models, no additional installation needed
try:
    from torchvision.models.optical_flow import raft_large, raft_small, Raft_Large_Weights, Raft_Small_Weights
    RAFT_AVAILABLE = True
except ImportError:
    RAFT_AVAILABLE = False
    print("Warning: torchvision version too low, cannot use RAFT, will use simplified optical flow estimation")

# RAFT model size (global variable, set by command line)
RAFT_MODEL_SIZE = 'small'
TEMPORAL_WEIGHT = 1e5  # Much stronger RAFT optical flow temporal loss weight for better frame coherence


# ==================== Core Algorithm ====================

# -------------------------- RAFT Optical Flow Estimator --------------------------
class RAFTEstimator:
    def __init__(self, device, model_size='small'):
        """
        Initialize RAFT optical flow estimator
        model_size: 'small' or 'large', small is faster, large is more accurate
        """
        self.device = device
        self.model_size = model_size
        
        if not RAFT_AVAILABLE:
            raise ImportError("RAFT unavailable, please upgrade torchvision to 0.13+")
        
        # Load RAFT model
        if model_size == 'large':
            weights = Raft_Large_Weights.DEFAULT
            self.model = raft_large(weights=weights, progress=False).to(device)
        else:
            weights = Raft_Small_Weights.DEFAULT
            self.model = raft_small(weights=weights, progress=False).to(device)
        
        self.model.eval()
        
        # RAFT preprocessing transforms
        self.transforms = weights.transforms()

    def compute_flow(self, prev_frame, curr_frame):
        """
        Compute optical flow from previous frame to current frame (using RAFT)
        prev_frame: previous frame (PIL.Image, RGB format)
        curr_frame: current frame (PIL.Image, RGB format)
        return: optical flow tensor (1, 2, H, W), (u, v) motion vectors
        """
        # Convert to RGB numpy arrays
        prev_np = np.array(prev_frame).astype(np.uint8)
        curr_np = np.array(curr_frame).astype(np.uint8)
        
        # Use RAFT preprocessing
        prev_tensor, curr_tensor = self.transforms(prev_np, curr_np)
        prev_tensor = prev_tensor.unsqueeze(0).to(self.device)  # (1, 3, H, W)
        curr_tensor = curr_tensor.unsqueeze(0).to(self.device)  # (1, 3, H, W)
        
        # RAFT inference
        with torch.no_grad():
            # RAFT returns a list, the last element is the final optical flow
            flow_list = self.model(prev_tensor, curr_tensor)
            flow = flow_list[-1]  # Get final optical flow estimate (1, 2, H, W)
        
        return flow


def smooth_flow(flow, kernel_size=3):
    """
    Smooth optical flow using median filtering to reduce noise and jitter
    """
    if not SCIPY_AVAILABLE:
        # Fallback: use simple Gaussian blur if scipy not available
        flow_smooth = F.avg_pool2d(flow, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        return flow_smooth
    
    # Convert to numpy for median filtering
    flow_np = flow.squeeze(0).cpu().numpy()  # (2, H, W)
    
    # Apply median filter to each channel separately
    flow_smooth = np.zeros_like(flow_np)
    for i in range(2):
        flow_smooth[i] = ndimage.median_filter(flow_np[i], size=kernel_size)
    
    # Convert back to tensor
    flow_smooth = torch.from_numpy(flow_smooth).unsqueeze(0).to(flow.device)
    return flow_smooth


def warp_frame(prev_stylized, flow, device, smooth=True):
    """
    Enhanced warp function with flow smoothing for better temporal consistency
    """
    B, C, H, W = prev_stylized.shape
    
    # Smooth optical flow to reduce jitter
    if smooth:
        try:
            flow = smooth_flow(flow, kernel_size=3)  # Light smoothing
        except:
            pass  # If scipy not available, skip smoothing
    
    # If optical flow size doesn't match stylized result, adjust it
    if flow.shape[2:] != (H, W):
        flow = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)
    
    # Generate grid coordinates (-1 to 1 range)
    x = torch.linspace(-1, 1, W, device=device)
    y = torch.linspace(-1, 1, H, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    grid = torch.stack([xx, yy], dim=2).unsqueeze(0)  # (1, H, W, 2)
    
    # Normalize optical flow to grid offset range
    flow_norm = torch.cat([
        flow[:, 0:1, :, :] / ((W - 1) / 2),  # u channel normalization
        flow[:, 1:2, :, :] / ((H - 1) / 2)   # v channel normalization
    ], dim=1).permute(0, 2, 3, 1)  # (1, H, W, 2)
    
    # Warped grid = original grid + optical flow offset
    warped_grid = grid + flow_norm
    # Use bilinear interpolation with reflection padding for better edge handling
    warped = F.grid_sample(prev_stylized, warped_grid, mode="bilinear", 
                          padding_mode="reflection", align_corners=False)
    return warped


# 1. AdaIN (Adaptive Instance Normalization) - Advanced style transfer method
def adaptive_instance_normalization(content_feat, style_feat):
    """
    AdaIN: Match mean and std of content features to style features
    This is more effective than Gram matrix for style transfer
    """
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def calc_mean_std(feat, eps=1e-5):
    """
    Calculate mean and standard deviation for AdaIN
    """
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adain_style_loss(gen_feat, style_feat):
    """
    AdaIN style loss: match feature statistics (mean and std)
    More effective than Gram matrix for color and texture transfer
    """
    gen_mean, gen_std = calc_mean_std(gen_feat)
    style_mean, style_std = calc_mean_std(style_feat)
    
    mean_loss = F.mse_loss(gen_mean, style_mean)
    std_loss = F.mse_loss(gen_std, style_std)
    
    return mean_loss + std_loss


# Enhanced Gram matrix (kept as backup/alternative)
def gram_matrix(x):
    b, c, h, w = x.size()
    x = x.view(b * c, h * w)
    return torch.mm(x, x.t()) / (b * h * w)


# 2. Enhanced VGG feature extractor: focus on shallow layers for geometric/abstract styles
class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        for param in self.vgg.parameters():
            param.requires_grad = False
        # Focus on shallow layers for geometric/abstract styles (edges, shapes, colors)
        # Shallow layers capture edges and geometric patterns better
        self.style_layers = ["1", "6", "11", "20"]  # relu1_1, relu2_1, relu3_1, relu4_1 (emphasize edges)
        self.content_layer = "25"  # relu4_2 (deep content layer)

    def forward(self, x):
        style_feats = []
        content_feat = None
        layer_idx = 0

        for name, layer in self.vgg.named_children():
            x = layer(x)
            if name in self.style_layers:
                style_feats.append(x)
            if name == self.content_layer:
                content_feat = x
            if content_feat is not None and len(style_feats) == len(self.style_layers):
                break
        return style_feats, content_feat


# 3. Enhanced loss functions for strong geometric/abstract styles
def calculate_content_loss(gen_feat, orig_feat):
    # Minimize content weight to almost completely remove original video colors
    return F.mse_loss(gen_feat, orig_feat) * 5e2  # Very low to almost eliminate content colors


def calculate_edge_loss(gen_tensor, style_tensor, edge_scale_factor=1.0):
    """
    Enhanced edge/contour loss to strengthen boundary learning
    Uses Sobel-like operators to detect edges
    
    Args:
        gen_tensor: Generated image tensor (B, C, H, W)
        style_tensor: Style image tensor (B, C, H, W) - will be resized to match gen_tensor if needed
        edge_scale_factor: Scale factor for edge detection to adapt line thickness based on size ratio
                          If video > style image: > 1.0 (thicker lines)
                          If video < style image: < 1.0 (thinner lines)
                          If video == style image: 1.0 (original lines)
    """
    # Ensure style_tensor matches gen_tensor size
    if style_tensor.shape[2:] != gen_tensor.shape[2:]:
        style_tensor = F.interpolate(style_tensor, size=gen_tensor.shape[2:], 
                                     mode='bilinear', align_corners=False)
    
    # Adjust edge detection scale based on size ratio (both directions)
    # Limit scale factor to reasonable range (0.25x to 4x)
    kernel_scale = max(0.25, min(edge_scale_factor, 4.0))
    
    # Base Sobel-like edge detection kernels
    sobel_x_base = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=gen_tensor.dtype, device=gen_tensor.device).view(1, 1, 3, 3)
    sobel_y_base = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=gen_tensor.dtype, device=gen_tensor.device).view(1, 1, 3, 3)
    
    edge_loss = 0.0
    for c in range(gen_tensor.size(1)):  # For each RGB channel
        gen_ch = gen_tensor[:, c:c+1, :, :]
        style_ch = style_tensor[:, c:c+1, :, :]
        
        # Compute edge magnitude with base kernels
        gen_edge_x = F.conv2d(gen_ch, sobel_x_base, padding=1)
        gen_edge_y = F.conv2d(gen_ch, sobel_y_base, padding=1)
        gen_edge_mag = torch.sqrt(gen_edge_x**2 + gen_edge_y**2 + 1e-6)
        
        style_edge_x = F.conv2d(style_ch, sobel_x_base, padding=1)
        style_edge_y = F.conv2d(style_ch, sobel_y_base, padding=1)
        style_edge_mag = torch.sqrt(style_edge_x**2 + style_edge_y**2 + 1e-6)
        
        # Apply scale-adaptive edge processing based on size ratio
        if kernel_scale > 1.0:
            # Video is larger than style image: make lines thicker (blur/dilate)
            # Apply average pooling to simulate thicker lines (proportional to scale)
            blur_kernel_size = int(2 * kernel_scale + 1)  # Odd number, proportional to scale
            blur_kernel_size = min(blur_kernel_size, 9)  # Limit to reasonable size
            if blur_kernel_size >= 3:
                # Use average pooling as approximation of Gaussian blur (simulates thicker lines)
                gen_edge_mag = F.avg_pool2d(gen_edge_mag, kernel_size=blur_kernel_size, stride=1, padding=blur_kernel_size//2)
                style_edge_mag = F.avg_pool2d(style_edge_mag, kernel_size=blur_kernel_size, stride=1, padding=blur_kernel_size//2)
        elif kernel_scale < 1.0:
            # Video is smaller than style image: make lines thinner (sharpen)
            # Apply sharpening to make edges more precise (inverse of blur)
            # Use a smaller kernel or skip pooling to preserve fine details
            # For very small scale factors, we can apply edge enhancement
            sharpen_factor = 1.0 / kernel_scale  # Inverse relationship
            # Apply slight sharpening by emphasizing high-frequency components
            # This is done by subtracting a blurred version (unsharp mask effect)
            if sharpen_factor > 1.0:
                # Create a small blur kernel for unsharp masking
                blur_size = max(3, int(3 / sharpen_factor))  # Smaller blur for sharper edges
                if blur_size >= 3:
                    blurred_gen = F.avg_pool2d(gen_edge_mag, kernel_size=blur_size, stride=1, padding=blur_size//2)
                    blurred_style = F.avg_pool2d(style_edge_mag, kernel_size=blur_size, stride=1, padding=blur_size//2)
                    # Unsharp mask: original + (original - blurred) * factor
                    sharpening_strength = min(0.3, (sharpen_factor - 1.0) * 0.1)  # Limit sharpening strength
                    gen_edge_mag = gen_edge_mag + (gen_edge_mag - blurred_gen) * sharpening_strength
                    style_edge_mag = style_edge_mag + (style_edge_mag - blurred_style) * sharpening_strength
                    gen_edge_mag = torch.clamp(gen_edge_mag, min=0.0)  # Ensure non-negative
                    style_edge_mag = torch.clamp(style_edge_mag, min=0.0)
        # If kernel_scale == 1.0, no adjustment needed (use original edges)
        
        # Match edge strength and distribution
        edge_loss += F.mse_loss(gen_edge_mag, style_edge_mag)
        edge_loss += F.mse_loss(gen_edge_mag.mean(), style_edge_mag.mean())
        edge_loss += F.mse_loss(gen_edge_mag.std(), style_edge_mag.std())
        
        # Additional edge enhancement: encourage stronger edges in generated image
        # Penalize weak edges and reward strong edges
        edge_strength_loss = F.mse_loss(gen_edge_mag.max(), style_edge_mag.max())
        edge_loss += edge_strength_loss * 0.5
    
    return edge_loss * 2e4  # Increased weight for stronger edge learning (from 1e4 to 2e4)


def calculate_style_loss_gram(gen_style_feats, orig_style_feats):
    """
    Gram matrix loss - better for geometric/abstract styles with strong patterns
    Gram matrix captures texture patterns and geometric relationships
    Enhanced with stronger emphasis on shallow layers for edge learning
    """
    # Weight shallow layers much more heavily for edges and geometric shapes
    # [relu1_1, relu2_1, relu3_1, relu4_1] - emphasize first layer for edges
    layer_weights = [5.0, 3.0, 2.0, 1.5]  # Much heavier weight for first layer (edges/contours)
    
    loss = 0.0
    for gen, orig, weight in zip(gen_style_feats, orig_style_feats, layer_weights):
        gram_gen = gram_matrix(gen)
        gram_orig = gram_matrix(orig)
        # Use MSE loss for more stable optimization
        gram_loss = F.mse_loss(gram_gen, gram_orig)
        loss += weight * gram_loss
    
    return loss * 1e10  # Balanced weight (reduced from 5e11 to prevent black images)


def calculate_style_loss_adain(gen_style_feats, orig_style_feats, use_adain=False):
    """
    Style loss - use Gram matrix for geometric styles, AdaIN for others
    """
    if use_adain:
        # AdaIN for smooth styles
        layer_weights = [1.0, 1.0, 1.0, 1.0]
        loss = 0.0
        for gen, orig, weight in zip(gen_style_feats, orig_style_feats, layer_weights):
            adain_loss = adain_style_loss(gen, orig)
            loss += weight * adain_loss * 1e5
        return loss
    else:
        # Gram matrix for geometric/abstract styles (better for edges and patterns)
        return calculate_style_loss_gram(gen_style_feats, orig_style_feats)


def calculate_color_difference_loss(gen_tensor, style_tensor):
    """
    Learn the color difference/contrast between color blocks in style image
    This helps mimic the color block structure of the style image
    
    Args:
        gen_tensor: Generated image tensor (B, C, H, W)
        style_tensor: Style image tensor (B, C, H, W) - will be resized to match gen_tensor if needed
    """
    # Ensure style_tensor matches gen_tensor size
    if style_tensor.shape[2:] != gen_tensor.shape[2:]:
        style_tensor = F.interpolate(style_tensor, size=gen_tensor.shape[2:], 
                                     mode='bilinear', align_corners=False)
    
    # Compute local color differences using gradients (Sobel-like)
    # Horizontal differences
    gen_h_diff = torch.abs(gen_tensor[:, :, :, 1:] - gen_tensor[:, :, :, :-1])
    style_h_diff = torch.abs(style_tensor[:, :, :, 1:] - style_tensor[:, :, :, :-1])
    # Vertical differences
    gen_v_diff = torch.abs(gen_tensor[:, :, 1:, :] - gen_tensor[:, :, :-1, :])
    style_v_diff = torch.abs(style_tensor[:, :, 1:, :] - style_tensor[:, :, :-1, :])
    
    # Match the magnitude of color differences (how distinct color blocks are)
    h_diff_loss = F.mse_loss(gen_h_diff.mean(), style_h_diff.mean())
    v_diff_loss = F.mse_loss(gen_v_diff.mean(), style_v_diff.mean())
    
    # Also match the variance of differences (color block size distribution)
    h_var_loss = F.mse_loss(gen_h_diff.var(), style_h_diff.var())
    v_var_loss = F.mse_loss(gen_v_diff.var(), style_v_diff.var())
    
    return (h_diff_loss + v_diff_loss + h_var_loss + v_var_loss) * 2e5


def calculate_direct_color_match_loss(gen_tensor, style_tensor):
    """
    Extremely strong color matching - force style colors to match almost exactly
    Also learn color difference patterns between color blocks
    
    Args:
        gen_tensor: Generated image tensor (B, C, H, W)
        style_tensor: Style image tensor (B, C, H, W) - will be resized to match gen_tensor if needed
    """
    # Ensure style_tensor matches gen_tensor size
    if style_tensor.shape[2:] != gen_tensor.shape[2:]:
        style_tensor = F.interpolate(style_tensor, size=gen_tensor.shape[2:], 
                                     mode='bilinear', align_corners=False)
    
    gen_mean, gen_std = calc_mean_std(gen_tensor)
    style_mean, style_std = calc_mean_std(style_tensor)
    
    mean_loss = F.mse_loss(gen_mean, style_mean)  # MSE for stability
    std_loss = F.mse_loss(gen_std, style_std)
    
    # Add pixel-level color matching for even stronger color transfer
    # Match RGB channels directly (normalized)
    gen_rgb = gen_tensor.view(gen_tensor.size(0), gen_tensor.size(1), -1)  # (B, C, H*W)
    style_rgb = style_tensor.view(style_tensor.size(0), style_tensor.size(1), -1)
    
    # Match per-channel histograms (simplified: match mean and std per channel)
    pixel_loss = 0.0
    for c in range(gen_tensor.size(1)):  # For each RGB channel
        gen_channel = gen_rgb[:, c, :]
        style_channel = style_rgb[:, c, :]
        # Match channel statistics
        gen_ch_mean = gen_channel.mean()
        gen_ch_std = gen_channel.std() + 1e-5
        style_ch_mean = style_channel.mean()
        style_ch_std = style_channel.std() + 1e-5
        pixel_loss += F.mse_loss(gen_ch_mean, style_ch_mean) + F.mse_loss(gen_ch_std, style_ch_std)
    
    # Add color difference learning to mimic color block structure
    color_diff_loss = calculate_color_difference_loss(gen_tensor, style_tensor)
    
    # Extremely strong weight to force style colors to match almost exactly
    return (mean_loss + std_loss) * 5e5 + pixel_loss * 1e5 + color_diff_loss  # Much stronger color matching


def calculate_color_consistency_loss(gen_tensor):
    # Enhanced color consistency - encourage vibrant colors
    mean = gen_tensor.mean(dim=[2, 3], keepdim=True)  # Color tone
    std = gen_tensor.std(dim=[2, 3], keepdim=True)    # Color saturation
    
    # Encourage more vibrant colors (higher std, moderate mean)
    mean_target = torch.clamp(mean, 0.3, 0.7)  # Moderate brightness
    std_target = torch.clamp(std, 0.15, 0.6)   # Higher saturation range
    
    mean_loss = F.mse_loss(mean, mean_target)
    std_loss = F.mse_loss(std, std_target)
    return (mean_loss + std_loss) * 1.5e4  # Increased weight


def calculate_total_variation_loss(gen_tensor):
    """
    Reduced TV loss for geometric styles - we want sharp edges, not smooth
    """
    batch_size = gen_tensor.size()[0]
    tv_h = torch.pow(gen_tensor[:, :, 1:, :] - gen_tensor[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(gen_tensor[:, :, :, 1:] - gen_tensor[:, :, :, :-1], 2).sum()
    return (tv_h + tv_w) / batch_size * 1e1  # Much lower penalty for sharp edges


def calculate_temporal_loss(prev_gen_tensor, curr_gen_tensor):
    # Only constrain smooth color changes between frames, don't affect details
    if prev_gen_tensor is None or curr_gen_tensor is None:
        return torch.tensor(0.0, device=curr_gen_tensor.device)
    if prev_gen_tensor.shape != curr_gen_tensor.shape:
        curr_gen_tensor = F.interpolate(curr_gen_tensor, size=prev_gen_tensor.shape[2:], mode='bilinear')
    curr_color = curr_gen_tensor.mean(dim=1, keepdim=True)  # Color mean
    prev_color = prev_gen_tensor.mean(dim=1, keepdim=True)
    return F.mse_loss(curr_color, prev_color) * 5e3


# 4. Enhanced single frame transfer for strong geometric/abstract styles
def stylize_single_frame(content_img, style_feats_orig, content_feat_orig, extractor, device, 
                         style_tensor=None, warped_prev=None, use_adain=False, target_size=None, edge_scale_factor=1.0):
    """
    Stylize a single frame with adaptive edge scaling based on video/style size ratio
    
    Args:
        content_img: PIL Image of content frame
        style_feats_orig: Style features extracted from style image
        content_feat_orig: Content features extracted from content frame
        extractor: VGGFeatureExtractor for feature extraction
        device: torch.device (cuda or cpu)
        style_tensor: Style tensor for direct color matching (optional)
        warped_prev: Warped previous frame for temporal consistency (optional)
        use_adain: Whether to use AdaIN style loss (default: False, uses Gram matrix)
        target_size: Target size (width, height) for processing. If None, uses content_img.size
        edge_scale_factor: Scale factor for adaptive edge/line thickness adjustment
                          - If > 1.0: video is larger than style image, lines will be thicker
                          - If < 1.0: video is smaller than style image, lines will be thinner
                          - If = 1.0: same size, no adjustment
                          This ensures lines maintain proportional thickness regardless of size difference
    """
    # Determine processing size
    if target_size is not None:
        process_size = target_size  # (width, height)
    else:
        process_size = content_img.size  # Use content image size
    
    # Ensure content image matches process size
    if content_img.size != process_size:
        print(f"  Resizing content image from {content_img.size} to {process_size}", flush=True)
        content_img = content_img.resize(process_size, Image.Resampling.LANCZOS)
    
    # Preprocessing - no resize needed since image is already correct size
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    content_tensor = transform(content_img).unsqueeze(0).to(device)
    
    # Verify tensor size
    tensor_h, tensor_w = content_tensor.shape[2], content_tensor.shape[3]
    expected_h, expected_w = process_size[1], process_size[0]  # process_size is (width, height)
    if tensor_h != expected_h or tensor_w != expected_w:
        print(f"  ERROR: content_tensor size ({tensor_w}, {tensor_h}) != expected {process_size}", flush=True)

    # Initialize from style colors, almost completely remove original video colors
    gen_tensor = content_tensor.clone().requires_grad_(True)
    with torch.no_grad():
        if style_tensor is not None:
            # Start with style image colors (almost 100% style, minimal content)
            # Blend: 2% content structure + 98% style colors (almost completely remove original colors)
            style_mean, style_std = calc_mean_std(style_tensor)
            content_mean, content_std = calc_mean_std(content_tensor)
            
            # Normalize content to match style statistics
            gen_normalized = (content_tensor - content_mean) / (content_std + 1e-5)
            gen_init = gen_normalized * style_std + style_mean
            
            # Blend with original content (2% content, 98% style-colored) - almost completely remove original colors
            gen_init = 0.02 * content_tensor + 0.98 * gen_init
            
            # Update gen_tensor in-place to preserve requires_grad
            gen_tensor.data.copy_(gen_init.data)
        else:
            # Fallback: start from content
            gen_tensor.add_(torch.randn_like(gen_tensor) * 0.1)
    
    # Ensure requires_grad is True
    gen_tensor.requires_grad_(True)

    # Use Adam with moderate learning rate for stable convergence
    optimizer = optim.Adam([gen_tensor], lr=2e-3)  # Moderate LR for stable learning
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.8)
    epochs = 400  # Sufficient iterations

    # Optimize loop
    for epoch in range(epochs):
        try:
            optimizer.zero_grad()
            gen_style_feats, gen_content_feat = extractor(gen_tensor)

            if gen_content_feat is None or content_feat_orig is None:
                raise ValueError("Content feature extraction failed")
            if len(gen_style_feats) != len(style_feats_orig):
                raise ValueError(f"Style feature count mismatch: got {len(gen_style_feats)}, expected {len(style_feats_orig)}")
        except Exception as e:
            print(f"ERROR in optimization iteration {epoch + 1}: {str(e)}", flush=True)
            raise

        try:
            # Enhanced loss calculation for strong geometric styles
            c_loss = calculate_content_loss(gen_content_feat, content_feat_orig)
            s_loss = calculate_style_loss_adain(gen_style_feats, style_feats_orig, use_adain=use_adain)
            tv_loss = calculate_total_variation_loss(gen_tensor)  # Low penalty for sharp edges
            
            # Strong color matching with color difference learning
            color_match_loss = torch.tensor(0.0, device=device)
            edge_loss = torch.tensor(0.0, device=device)
            if style_tensor is not None:
                color_match_loss = calculate_direct_color_match_loss(gen_tensor, style_tensor)
                edge_loss = calculate_edge_loss(gen_tensor, style_tensor, edge_scale_factor=edge_scale_factor)  # Enhanced edge/contour learning with scale adaptation
            
            # Enhanced RAFT temporal loss - multiple scales for better consistency
            temporal_loss = torch.tensor(0.0, device=device)
            if warped_prev is not None:
                if gen_tensor.shape != warped_prev.shape:
                    warped_prev = F.interpolate(warped_prev, size=gen_tensor.shape[2:], mode='bilinear', align_corners=False)
                
                # Ensure warped_prev requires grad (or detach it if it doesn't)
                if not warped_prev.requires_grad:
                    # If warped_prev doesn't require grad, detach it and compute loss
                    warped_prev = warped_prev.detach()
                
                # Enhanced temporal loss for much better frame coherence
                # Primary pixel-level loss (L1 for robustness)
                pixel_loss = F.l1_loss(gen_tensor, warped_prev)
                
                # Additional smoothness: penalize large differences more
                diff = torch.abs(gen_tensor - warped_prev)
                smoothness_loss = torch.mean(diff * diff)  # L2 on differences
                
                # Edge consistency: maintain edge structure between frames
                # Compute edge differences to preserve contour boundaries
                edge_diff = torch.abs(gen_tensor[:, :, 1:, :] - gen_tensor[:, :, :-1, :]) - \
                           torch.abs(warped_prev[:, :, 1:, :] - warped_prev[:, :, :-1, :])
                edge_consistency_loss = torch.mean(edge_diff * edge_diff)
                
                # Color consistency: maintain color distribution between frames
                gen_color_mean = gen_tensor.mean(dim=[2, 3], keepdim=True)
                warped_color_mean = warped_prev.mean(dim=[2, 3], keepdim=True)
                color_consistency_loss = F.mse_loss(gen_color_mean, warped_color_mean)
                
                # Combined temporal loss with multiple consistency terms
                temporal_loss = pixel_loss + 0.2 * smoothness_loss + 0.3 * edge_consistency_loss + 0.2 * color_consistency_loss
            
            # Combined loss - style and color dominate, content almost eliminated, enhanced edges and temporal coherence
            total_loss = (c_loss * 0.05 +  # Almost eliminate content weight
                         s_loss + 
                         color_match_loss * 8.0 +  # Very strong color matching weight
                         edge_loss * 4.0 +  # Increased edge/contour learning weight (from 2.0 to 4.0) for more visible lines
                         tv_loss * 5e1 +  # Lower TV weight (from 1e2 to 5e1) to allow sharper edges
                         TEMPORAL_WEIGHT * temporal_loss)  # Much stronger temporal coherence

            # Check for NaN or Inf
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"WARNING: Invalid loss value at iteration {epoch + 1}: {total_loss.item()}", flush=True)
                print(f"  c_loss: {c_loss.item()}, s_loss: {s_loss.item()}, color: {color_match_loss.item()}", flush=True)
                break  # Stop optimization if loss is invalid

            total_loss.backward()
            
            # Gradient clipping to prevent explosion and black images
            torch.nn.utils.clip_grad_norm_([gen_tensor], max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Clamp values to prevent going to extreme (black/white)
            with torch.no_grad():
                gen_tensor.clamp_(-2.0, 2.0)  # Keep in reasonable range
        except Exception as e:
            print(f"ERROR in loss calculation/optimization at iteration {epoch + 1}: {str(e)}", flush=True)
            import traceback
            print(traceback.format_exc(), flush=True)
            raise

        if (epoch + 1) % 20 == 0:  # More frequent output (every 20 iterations)
            print(f"Iteration {epoch+1}/{epochs}, Style:Content:Color:Edge:TV:Temporal = "
                  f"{s_loss.item():.1f} : {c_loss.item():.1f} : {color_match_loss.item():.1f} : "
                  f"{edge_loss.item():.1f} : {tv_loss.item():.1f} : {temporal_loss.item():.1f}", flush=True)

    # Post-processing: minimal processing to avoid white noise
    try:
        # Convert tensor to numpy and denormalize (essential base conversion)
        gen_np = gen_tensor.squeeze(0).detach().cpu().numpy()
        gen_np = gen_np.transpose(1, 2, 0)
        # Denormalize from ImageNet normalization
        gen_np = gen_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        gen_np = np.clip(gen_np, 0, 1)
        
        # Basic color transfer (if style_tensor provided) - minimal processing
        if style_tensor is not None:
            style_np = style_tensor.squeeze(0).detach().cpu().numpy()
            style_np = style_np.transpose(1, 2, 0)
            style_np = style_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            style_np = np.clip(style_np, 0, 1)
            
            # Basic color transfer: match mean and std
            gen_mean = gen_np.mean(axis=(0, 1), keepdims=True)
            gen_std = gen_np.std(axis=(0, 1), keepdims=True) + 1e-5
            style_mean = style_np.mean(axis=(0, 1), keepdims=True)
            style_std = style_np.std(axis=(0, 1), keepdims=True) + 1e-5
            
            # Transfer style colors: normalize gen to style statistics
            gen_np = (gen_np - gen_mean) / gen_std * style_std + style_mean
            gen_np = np.clip(gen_np, 0, 1)
        
        # Convert to uint8 first for sharpening
        gen_np_uint8 = (gen_np * 255).astype(np.uint8)
        
        # Stronger sharpening for more visible lines and edges
        # Using unsharp mask kernel with enhanced edge enhancement
        sharpen_kernel = np.array([
            [0, -0.12, 0],
            [-0.12, 1.5, -0.12],
            [0, -0.12, 0]
        ], dtype=np.float32)
        
        # Apply sharpening to each channel separately
        gen_np_sharpened = np.zeros_like(gen_np_uint8)
        for c in range(3):
            gen_np_sharpened[:, :, c] = cv2.filter2D(gen_np_uint8[:, :, c], -1, sharpen_kernel)
        
        # Clip to valid range
        gen_np_sharpened = np.clip(gen_np_sharpened, 0, 255).astype(np.uint8)
        
        # Enhance color contrast and saturation for more distinct colors
        # Convert to HSV for saturation and brightness adjustment
        gen_hsv = cv2.cvtColor(gen_np_sharpened, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Increase saturation by 20% to make colors more distinct
        gen_hsv[:, :, 1] = np.clip(gen_hsv[:, :, 1] * 1.2, 0, 255)
        
        # Slight contrast enhancement in brightness (V channel)
        # Apply a gentle S-curve to enhance contrast
        v_channel = gen_hsv[:, :, 2] / 255.0
        v_enhanced = np.power(v_channel, 0.9) * 255.0  # Slight brightening for better contrast
        gen_hsv[:, :, 2] = np.clip(v_enhanced, 0, 255)
        
        # Convert back to RGB
        gen_np_enhanced = cv2.cvtColor(gen_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        return Image.fromarray(gen_np_enhanced), gen_tensor
    except Exception as e:
        import traceback
        print(f"ERROR in post-processing: {str(e)}", flush=True)
        print(traceback.format_exc(), flush=True)
        raise


# 5. Video transfer: enhance inter-frame color consistency (integrate RAFT optical flow)
def stylize_video(content_video_path, style_image_path, output_video_path, device, resolution='256', progress_callback=None):
    """
    Stylize entire video with temporal consistency.

    Args:
        content_video_path: Path to input video file
        style_image_path: Path to style image file
        output_video_path: Path to save output stylized video
        device: torch.device (cuda or cpu)
        resolution: '256' for fixed 256x256 output, 'original' to keep source resolution
        progress_callback: Optional callback function(frame_count, total_frames) for progress updates

    Features:
        - Optional fixed 256x256 resolution for fast processing
        - Optional original-resolution output for higher fidelity
        - Bidirectional RAFT optical flow for temporal consistency
        - HSV color fusion between frames for smooth transitions
    """
    # Open video and collect metadata
    cap = cv2.VideoCapture(content_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if resolution == 'original' and (orig_width <= 0 or orig_height <= 0):
        # Fallback: read first frame to infer size when metadata is missing
        ret, probe_frame = cap.read()
        if ret:
            orig_height, orig_width = probe_frame.shape[:2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if resolution == 'original' and orig_width > 0 and orig_height > 0:
        video_size = (orig_width, orig_height)
    else:
        video_size = (256, 256)
    output_video_size = video_size
    
    # Load style image and resize to processing resolution
    style_img = Image.open(style_image_path).convert('RGB')
    print(f"Style image original size: {style_img.size[0]}x{style_img.size[1]}", flush=True)
    style_img_resized = style_img.resize(video_size, Image.Resampling.LANCZOS)
    print(f"Style image resized to: {video_size[0]}x{video_size[1]} for processing", flush=True)
    
    # Initialize VGG feature extractor
    extractor = VGGFeatureExtractor().to(device).eval()
    
    # Style image preprocessing
    style_size = video_size
    style_size_tensor = (video_size[1], video_size[0])  # (height, width) for tensor
    
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    style_tensor = style_transform(style_img_resized).unsqueeze(0).to(device)
    
    # Verify style_tensor size matches expected
    if style_tensor.shape[2:] != style_size_tensor:
        print(f"  Warning: style_tensor size {style_tensor.shape[2:]} != expected {style_size_tensor}, resizing...", flush=True)
        style_tensor = F.interpolate(style_tensor, size=style_size_tensor, mode='bilinear', align_corners=False)
    style_feats_orig, _ = extractor(style_tensor)
    if len(style_feats_orig) != 4:  # Updated for 4 style layers (geometric focus)
        raise ValueError(f"Style feature layer count error: got {len(style_feats_orig)}, expected 4")
    
    # Store style_tensor for direct color matching
    style_tensor_for_color = style_tensor
    
    # Edge scale factor adapts line thickness based on output size
    edge_scale_factor = max(video_size) / 256.0

    # Initialize RAFT optical flow estimator
    global RAFT_MODEL_SIZE
    flow_estimator = None
    if RAFT_AVAILABLE:
        try:
            flow_estimator = RAFTEstimator(device, model_size=RAFT_MODEL_SIZE)
            print(f"RAFT-{RAFT_MODEL_SIZE} model loaded successfully, optical flow computation enabled")
        except Exception as e:
            print(f"Warning: RAFT loading failed: {str(e)}, will use simplified temporal consistency")
            flow_estimator = None
    else:
        print("Warning: RAFT unavailable, will use simplified temporal consistency")

    # Output resolution is determined by the selected mode
    frame_width, frame_height = video_size
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    prev_gen_tensor = None
    prev_gen_frame = None
    prev_frame_pil = None  # Previous frame (PIL format, for RAFT input)
    print(f"\nStarting video processing (total {total_frames} frames):")
    print("Note: Each frame takes ~30-60 seconds to process (400 iterations)...", flush=True)

    try:
        while cap.isOpened() and frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Failed to read frame {frame_count + 1}, stopping.", flush=True)
                break

            print(f"\nProcessing frame {frame_count + 1}/{total_frames}...", flush=True)
            
            try:
                content_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Resize content image to video_size if needed
                if content_img.size != video_size:
                    content_img = content_img.resize(video_size, Image.Resampling.LANCZOS)
                
                curr_frame_pil = content_img  # For RAFT input (use same size)
                
                # Use same transform as style (ImageNet normalization, no resize needed)
                content_tensor = style_transform(content_img).unsqueeze(0).to(device)
                _, content_feat_orig = extractor(content_tensor)
                if content_feat_orig is None:
                    raise ValueError(f"Frame {frame_count} content feature extraction failed")
                
                print(f"  Starting style transfer optimization...", flush=True)
            except Exception as e:
                print(f"ERROR preparing frame {frame_count + 1}: {str(e)}", flush=True)
                import traceback
                print(traceback.format_exc(), flush=True)
                raise

            # Compute optical flow and warp previous frame (starting from second frame)
            warped_prev = None
            if frame_count > 0 and prev_frame_pil is not None and prev_gen_tensor is not None:
                if flow_estimator is not None:
                    # Use RAFT to compute optical flow
                    try:
                        flow = flow_estimator.compute_flow(prev_frame_pil, curr_frame_pil)
                        # Warp previous stylized result to current frame viewpoint with smoothing
                        warped_prev = warp_frame(prev_gen_tensor, flow, device, smooth=True)
                        if frame_count % 10 == 0:
                            print(f"  Computed optical flow and warped previous frame", flush=True)
                    except Exception as e:
                        print(f"  Warning: Optical flow computation failed: {str(e)}, skipping temporal consistency constraint", flush=True)
                        warped_prev = None
                else:
                    # If no RAFT, use simple inter-frame smoothing
                    warped_prev = prev_gen_tensor

            try:
                gen_img, gen_tensor = stylize_single_frame(
                    content_img, style_feats_orig, content_feat_orig, extractor, device, 
                    style_tensor=style_tensor_for_color, warped_prev=warped_prev,
                    target_size=video_size,
                    edge_scale_factor=edge_scale_factor
                )
                
                # Ensure output image matches target resolution
                if gen_img.size != video_size:
                    gen_img = gen_img.resize(video_size, Image.Resampling.LANCZOS)
                print(f"  Frame {frame_count + 1} completed!", flush=True)
            except Exception as e:
                import traceback
                error_msg = f"Error processing frame {frame_count + 1}:\n{str(e)}\n{traceback.format_exc()}"
                print(f"ERROR: {error_msg}", flush=True)
                raise RuntimeError(f"Failed to process frame {frame_count + 1}: {str(e)}")

            # Enhanced inter-frame fusion: much stronger temporal smoothing for better coherence
            if prev_gen_frame is not None:
                curr_np = np.array(gen_img).astype(np.float32)
                prev_np = np.array(prev_gen_frame).astype(np.float32)
                
                # Much stronger blending for smoother transitions (increased from 0.85/0.15 to 0.75/0.25)
                # Blend in RGB space first (preserves color relationships)
                blended_rgb = cv2.addWeighted(curr_np, 0.75, prev_np, 0.25, 0)
                
                # Additional HSV smoothing for color consistency
                curr_hsv = cv2.cvtColor(blended_rgb.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
                prev_hsv = cv2.cvtColor(prev_np.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
                
                # Much stronger smoothing for saturation and brightness (increased from 0.85/0.15 to 0.75/0.25)
                curr_hsv[:, :, 1] = cv2.addWeighted(curr_hsv[:, :, 1], 0.75, prev_hsv[:, :, 1], 0.25, 0)
                curr_hsv[:, :, 2] = cv2.addWeighted(curr_hsv[:, :, 2], 0.75, prev_hsv[:, :, 2], 0.25, 0)
                
                # Preserve hue but allow stronger smoothing for better coherence
                hue_diff = np.abs(curr_hsv[:, :, 0] - prev_hsv[:, :, 0])
                hue_diff = np.minimum(hue_diff, 360 - hue_diff)  # Circular difference
                mask = hue_diff < 40  # Increased threshold (from 30 to 40) for more smoothing
                curr_hsv[:, :, 0] = np.where(mask, 
                                            cv2.addWeighted(curr_hsv[:, :, 0], 0.8, prev_hsv[:, :, 0], 0.2, 0),  # Stronger hue smoothing
                                            curr_hsv[:, :, 0])
                
                gen_img = Image.fromarray(cv2.cvtColor(curr_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB))
                
                # Ensure fused image matches target resolution
                if gen_img.size != video_size:
                    gen_img = gen_img.resize(video_size, Image.Resampling.LANCZOS)

            prev_gen_tensor = gen_tensor.detach()
            prev_gen_frame = gen_img
            prev_frame_pil = curr_frame_pil.copy()
            
            # Final check: Ensure output image matches target resolution
            if gen_img.size != video_size:
                gen_img = gen_img.resize(video_size, Image.Resampling.LANCZOS)
            
            gen_frame_cv = cv2.cvtColor(np.array(gen_img), cv2.COLOR_RGB2BGR)
            
            # Final verification: frame size must match target resolution
            actual_h, actual_w = gen_frame_cv.shape[:2]
            if actual_h != frame_height or actual_w != frame_width:
                gen_frame_cv = cv2.resize(gen_frame_cv, (frame_width, frame_height), interpolation=cv2.INTER_LANCZOS4)
            
            out.write(gen_frame_cv)

            frame_count += 1
            
            # Call progress callback for UI update
            if progress_callback is not None:
                progress_callback(frame_count, total_frames)
            
            if frame_count % 5 == 0 or frame_count == 1:  # Print every 5 frames or first frame
                print(f"Progress: {frame_count}/{total_frames} frames ({100*frame_count/total_frames:.1f}%)", flush=True)
    except Exception as e:
        import traceback
        error_msg = f"Error during video processing:\n{str(e)}\n{traceback.format_exc()}"
        print(f"FATAL ERROR: {error_msg}", flush=True)
        raise
    finally:
        cap.release()
        out.release()
        print(f"\nVideo processing completed! Output path: {output_video_path}")


# ==================== Command Line Interface ====================
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Video Style Transfer - Command Line Version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (256x256 resolution)
  python cv_off.py --video input.mp4 --style style.jpg --output output.mp4

  # Keep original video resolution
  python cv_off.py --video input.mp4 --style style.jpg --output output.mp4 --resolution original
  
  # Use large RAFT model
  python cv_off.py --video input.mp4 --style style.jpg --output output.mp4 --raft large
  
  # Force CPU usage
  python cv_off.py --video input.mp4 --style style.jpg --output output.mp4 --device cpu
        """
    )
    
    parser.add_argument(
        '--video', '-v',
        type=str,
        required=True,
        help='Input video file path'
    )
    
    parser.add_argument(
        '--style', '-s',
        type=str,
        required=True,
        help='Style image file path'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output video file path'
    )
    
    parser.add_argument(
        '--raft',
        type=str,
        choices=['small', 'large'],
        default='small',
        help='RAFT model size: small (fast) or large (more accurate) (default: small)'
    )

    parser.add_argument(
        '--resolution',
        type=str,
        choices=['256', 'original'],
        default='256',
        help='Output resolution: 256 (fixed 256x256) or original (use source video resolution)'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cuda', 'cpu'],
        default='auto',
        help='Compute device: auto (auto-detect), cuda (GPU), or cpu (default: auto)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet mode: suppress progress output'
    )
    
    return parser.parse_args()


def validate_files(video_path, style_path):
    """Validate input files exist"""
    errors = []
    
    if not os.path.exists(video_path):
        errors.append(f"Error: Video file does not exist: {video_path}")
    
    if not os.path.exists(style_path):
        errors.append(f"Error: Style image does not exist: {style_path}")
    
    return errors


def get_device(device_arg):
    """Get compute device"""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    elif device_arg == 'cuda':
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, will use CPU")
            return torch.device("cpu")
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def progress_callback(frame_count, total_frames, quiet=False):
    """Progress callback function"""
    if quiet:
        return
    
    progress_pct = (frame_count / total_frames) * 100
    bar_length = 40
    filled_length = int(bar_length * frame_count / total_frames)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    print(f'\rProgress: [{bar}] {progress_pct:.1f}% ({frame_count}/{total_frames})', end='', flush=True)
    
    if frame_count == total_frames:
        print()  # New line when complete


# ==================== Main Program ====================
def main():
    """Main function"""
    args = parse_arguments()
    
    # Validate files
    errors = validate_files(args.video, args.style)
    if errors:
        for error in errors:
            print(error, flush=True)
        sys.exit(1)
    
    # Check output directory
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            print(f"Error: Cannot create output directory: {e}", flush=True)
            sys.exit(1)
    
    # Get device
    device = get_device(args.device)
    
    # Set global RAFT model size
    global RAFT_MODEL_SIZE
    RAFT_MODEL_SIZE = args.raft
    
    # Print configuration
    if not args.quiet:
        print("=" * 60)
        print("Video Style Transfer - Command Line Version")
        print("=" * 60)
        print(f"Input video: {args.video}")
        print(f"Style image: {args.style}")
        print(f"Output video: {args.output}")
        res_desc = '256x256 fast mode' if args.resolution == '256' else 'original resolution high quality mode'
        print(f"Resolution: {args.resolution} ({res_desc})")
        print(f"RAFT model: {args.raft}")
        print(f"Device: {device} ({'GPU' if device.type == 'cuda' else 'CPU'})")
        if device.type == 'cpu':
            print("Warning: Using CPU, processing will be very slow!")
        if not RAFT_AVAILABLE:
            print("Warning: RAFT unavailable, will use simplified temporal consistency")
        print("=" * 60)
        print()
    
    # Create progress callback
    def progress_cb(frame_count, total_frames):
        progress_callback(frame_count, total_frames, args.quiet)
    
    # Process video
    try:
        stylize_video(
            content_video_path=args.video,
            style_image_path=args.style,
            output_video_path=args.output,
            device=device,
            resolution=args.resolution,
            progress_callback=progress_cb
        )
        
        if not args.quiet:
            print("\n" + "=" * 60)
            print("✅ Processing completed successfully!")
            print(f"Output file: {args.output}")
            print("=" * 60)
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user", flush=True)
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\n❌ Error during processing: {str(e)}", flush=True)
        if not args.quiet:
            print(traceback.format_exc(), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

