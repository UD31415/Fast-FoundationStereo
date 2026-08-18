#!/usr/bin/env python

'''
Reverse engineering Fast Foundation Depth.
Merging boxfilter functions with shazam keypoint detection and matching to create a depth estimator that can run in real time on CPU.

Usage:

Environment : 
    .\\Envs\\depthrs

Install : 



'''
from math import dist
from multiprocessing.util import debug
import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib import cm, image
from scipy.ndimage import zoom
from scipy.ndimage import correlate
from scipy.interpolate import RegularGridInterpolator
from scipy import ndimage
import sys 
sys.path.append(r'C:\Work\Projects\Utils\src')
from opencv_realsense_camera import RealSense, draw_str
from common import RectSelector
from logger import log
# from extract_images_from_ros1bag import read_bin_file
# from image_source import DataSource as DataSourceBin
# from measure_camera_noise import NoiseEstimator
sys.path.append(r'C:\Work\Projects\DepthRS\src')
from depth_data_source import DataSource

# ----------------------
# Helpers
from numpy.lib.stride_tricks import sliding_window_view

def extract_diag_band(corr: np.ndarray, B: int) -> np.ndarray:
    """
    Extract a band of B rows from `corr` (MxM) starting at the main diagonal
    and going down B-1 rows.

    disp[b, i] = corr[i - b, i]   for b in [0, B), i in [0, M)

    Out-of-range entries (i - b < 0) are filled with NaN.
    """
    M = corr.shape[0]
    if corr.ndim != 2 or corr.shape[1] != M:
        raise ValueError("corr must be a square 2D matrix")

    disp = np.full((B, M), np.nan, dtype=corr.dtype) if np.issubdtype(corr.dtype, np.floating) \
        else np.zeros((B, M), dtype=corr.dtype)

    cols = np.arange(M)
    for b in range(B):
        rows = cols - b
        valid = rows >= 0 
        disp[b, valid] = corr[rows[valid], cols[valid]]
    return disp

def extract_diag_band_fast(corr: np.ndarray, B: int) -> np.ndarray:
    """
    Extract a band of B rows from `corr` (MxM) starting at the main diagonal
    and going UP B-1 rows (i.e. entries above the diagonal).

    disp[b, i] = corr[i, i + b]   for b in [0, B), i in [0, M)

    Out-of-range entries (i + b >= M) are filled with 0.
    """
    M = corr.shape[0]
    # Pad right with B-1 columns so every row has B valid entries to the right
    padded = np.pad(corr, ((0, 0), (0, B - 1)), mode='constant', constant_values=0)
    # Window of shape (1, B) sliding along each row, then take diagonals
    win = sliding_window_view(padded, (1, B))[:, :, 0, :]  # shape (M, M, B)
    return win[np.arange(M), np.arange(M)].T  # (B, M)

def softmax_columns(x: np.ndarray, dim: int = 0, T: float = 1.0) -> np.ndarray:
    """
    Apply softmax independently to dimension `dim` of a 3D array.
    x_thr - provides a minimal threshold for the input values, below which the output will be zero. This can help suppress noise and focus on more confident matches. 
    The threshold can be tuned based on the expected range of input values and the desired level of sparsity in the output.

    For matrices: 
    For column j: out[i, j] = exp(x[i, j] - max_j) / sum_i exp(x[i, j] - max_j)
    """
    x_shifted = x - np.max(x, axis=dim, keepdims=True)   # numerical stability
    exp_x     = np.exp(x_shifted / T)
    return exp_x / np.sum(exp_x, axis=dim, keepdims=True)

def anisotropic_diffusion(img, num_iter=20, delta=0.25, kappa=0.1):
    """
    Perona-Malik Anisotropic Diffusion.
    img: 2D grayscale float array
    kappa: Edge threshold (higher = smoother, lower = preserves sharper edges)
    """
    out = img.astype(np.float32)
    kernel_size = (3, 3)  # Size of the Gaussian kernel for smoothing gradients
    
    for i in range(num_iter):
        # Calculate gradients in North, South, East, West directions
        grad_N = np.roll(out, -1, axis=0) - out
        grad_S = np.roll(out,  1, axis=0) - out
        grad_E = np.roll(out, -1, axis=1) - out
        grad_W = np.roll(out,  1, axis=1) - out

        # # add spatial filtering to the gradients to make them more robust to noise
        # grad_N = cv.GaussianBlur(grad_N, kernel_size, 0)
        # grad_S = cv.GaussianBlur(grad_S, kernel_size, 0)
        # grad_E = cv.GaussianBlur(grad_E, kernel_size, 0)
        # grad_W = cv.GaussianBlur(grad_W, kernel_size, 0)

        # grad_N = cv.Sobel(out, cv.CV_32F, 1, 0, ksize=3)
        # grad_S = cv.Sobel(out, cv.CV_32F, 1, 0, ksize=3)
        # grad_E = cv.Sobel(out, cv.CV_32F, 0, 1, ksize=3)
        # grad_W = cv.Sobel(out, cv.CV_32F, 0, 1, ksize=3)
        
        # Exponential conduction function
        c_N = np.exp(-(grad_N / kappa)**2)
        c_S = np.exp(-(grad_S / kappa)**2)
        c_E = np.exp(-(grad_E / kappa)**2)
        c_W = np.exp(-(grad_W / kappa)**2)

        # # Exponential conduction function
        # c_N = np.exp(-(cv.GaussianBlur(grad_N, kernel_size, 0) / kappa)**2)
        # c_S = np.exp(-(cv.GaussianBlur(grad_S, kernel_size, 0) / kappa)**2)
        # c_E = np.exp(-(cv.GaussianBlur(grad_E, kernel_size, 0) / kappa)**2)
        # c_W = np.exp(-(cv.GaussianBlur(grad_W, kernel_size, 0) / kappa)**2)        
        
        # Update image
        out += delta * (c_N * grad_N + c_S * grad_S + c_E * grad_E + c_W * grad_W)
        
    return np.clip(out, 0, 255).astype(np.uint8)

def grid_interpolation(data, factor = 2):
    
    log.info('Starting grid interpolation with factor: %d', factor)
    # Original grid dimensions
    nx, ny, nz = data.shape

    # Define the original coordinates
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.linspace(0, 1, nz)

    # Create the interpolator function
    interpolator = RegularGridInterpolator((x, y, z), data, method='linear')

    # Define the new dense grid (Factor of 2 -> 20 points per axis)
    new_nx, new_ny, new_nz = nx * factor, ny * factor, nz * factor
    new_x = np.linspace(0, 1, new_nx)
    new_y = np.linspace(0, 1, new_ny)
    new_z = np.linspace(0, 1, new_nz)

    # Generate a 3D meshgrid of the new coordinates
    X, Y, Z = np.meshgrid(new_x, new_y, new_z, indexing='ij')

    # Flatten the grid points to pass to the interpolator
    pts = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    # Interpolate and reshape back to 3D
    array_interp_2x = interpolator(pts).reshape(new_nx, new_ny, new_nz)
    log.info("Interpolated shape via grid: %s", array_interp_2x.shape)
    return array_interp_2x

def disparity_to_volume(disparity: np.ndarray, D: int, invalid_value=None) -> np.ndarray:
    """
    Convert a disparity image of shape (N, M) into a one-hot 3D volume of
    shape (N, M, D).

    For every pixel (i, j) with disparity d = disparity[i, j], the output
    volume has a 1 at channel d and 0 elsewhere:

        volume[i, j, d] = 1
        volume[i, j, k] = 0   for k != d

    Parameters
    ----------
    disparity : np.ndarray
        2D disparity image of shape (N, M). Values are expected in [0, D-1].
    D : int
        Number of disparity channels (depth of the output volume).
    invalid_value : optional
        If given, pixels equal to this value (e.g. NaN or a sentinel like -1)
        are treated as invalid and produce an all-zero column. Pixels whose
        rounded disparity falls outside [0, D-1] are also treated as invalid.

    Returns
    -------
    np.ndarray
        One-hot volume of shape (N, M, D) and dtype uint8.
    """
    if disparity.ndim != 2:
        raise ValueError(f"disparity must be 2D, got shape {disparity.shape}")
    if D <= 0:
        raise ValueError(f"D must be positive, got {D}")

    N, M = disparity.shape

    # Build a validity mask and integer disparity indices.
    if np.issubdtype(disparity.dtype, np.floating):
        valid = np.isfinite(disparity)
        d_int = np.where(valid, np.rint(disparity), 0).astype(np.int64)
    else:
        valid = np.ones_like(disparity, dtype=bool)
        d_int = disparity.astype(np.int64)

    if invalid_value is not None and not (
        isinstance(invalid_value, float) and np.isnan(invalid_value)
    ):
        valid &= disparity != invalid_value

    valid &= (d_int >= 0) & (d_int < D)

    volume = np.zeros((N, M, D), dtype=np.uint8)

    rows, cols = np.nonzero(valid)
    volume[rows, cols, d_int[rows, cols]] = 1

    return volume

# ----------------------
#%% Main
class ShazamDepthEstimator:

    def __init__(self, noise_type = 'T'):

        self.frame_size     = (640,480)
        self.imgD           = None  # depth from real sense
        self.imgC           = None  # estimation results
        self.imgL           = None
        self.imgR           = None
        self.img_type       = 0     # which image to load
        self.use_measure    = True # noise measurement
        self.debug_on       = False  # debug mode
        self.real_time_on   = False
        self.camera_bf      = 45000

        # params
        #self.MIN_STD_ERROR   = 0.01
        self.algo_type      = 1     # which algo to run - see update function
        self.noise_estimator_type = noise_type  # plane noise estimator

        # keypoint data
        self.detect_type    = 'ORB'  # or 'ORB' 'SIFT', 'AKAZE', 'BRISK'
        self.matcher_type   = 'BF'  # or 'FLANN'
        self.keypoint_detector = None  # keypoint detector
        self.keypoint_matcher = None  # keypoint detector

        #log.info('stop processing')
        self.kp_left        = None
        self.kp_right       = None
        self.matches        = None      

        # 3d conversion
        self.cam_matrix     = np.array([[1000,0,self.frame_size[0]/2],[0,1000,self.frame_size[1]/2],[0,0,1]], dtype = np.float32)
        self.cam_distort    = np.array([0,0,0,0,0],dtype = np.float32)    

        self.matrix_dir     = None     # direct u,v,1
        self.matrix_xyz     = None     # direct u,v,1 multiplied by z   
        self.rect_3d        = None    # roi but projected on 3D             

        # ROI for processing
        self.rect        = None
        self.grid_x      = None   # serach grid for ROI
        self.grid_y      = None

        # fit help
        self.mtrx_inv    = None

        # show 3d in real time
        self.show_info   = None
        self.debug_show  = True

        # opencv stereo matcher
        self.camera_bf   = 95*100   # camera basline multiplied by focal
        self.cv_stereo   = None 

        # noise estimators for the original and improved depth
        self.noise_rs  = None
        self.noise_fft  = None        

        log.info('ShazamDepthEstimator initialized')

    def init_roi(self, w = 0, h = 0, roi_type = 9):
        "load the roi case"
        if self.rect is not None:
            return self.rect
        
        if w < 5 or h < 5:
            w,h = self.frame_size
        
        #w,h     = img.shape
        w1, h1  = w>>1, h>>1
        h2, w2  = h>>2,w>>2           
        roi     = [0,0,w,h]
        if roi_type == 1:
            roi = [w1-32,h1-32,w1+32,h1+32] # xlu, ylu, xrb, yrb
        elif roi_type == 2:
            roi  = [w1-h2, h1-h2, w1+h2, h1+h2] 
        elif roi_type == 3:
            roi = [280,200,360,280] # xlu, ylu, xrb, yrb     
        elif roi_type == 4:
            roi = [w1-64,h1-48,w1+64,h1+48] # xlu, ylu, xrb, yrb                      
        elif roi_type == 5:
            roi = [w1-128,h1-128,w1+128,h1+128] # xlu, ylu, xrb, yrb    
        elif roi_type == 6:
            roi = [64,64,w-64,h-64] # xlu, ylu, xrb, yrb         
        elif roi_type == 7:
            roi = [128,128,w-128,h-128] # xlu, ylu, xrb, yrb        
        elif roi_type == 8:
            roi = [256,256,w-256,h-256] # xlu, ylu, xrb, yrb      
        elif roi_type == 9:
            roi = [32,32,w-32,h-32] # xlu, ylu, xrb, yrb 
        elif roi_type == 21:  # cube data
            roi = [500,240,660,360] # xlu, ylu, xrb, yrb             

        self.rect       = roi    
        #log.debug(f'Using ROI : {roi}')   
        return roi  

    def get_baseline_focal_product(self, srcId = 121):
        "compute baseline BF parameter"
        if srcId == 121 or srcId == 122 or srcId == 301 or srcId == 302:
            BF                  = 653.8470458984375 * 95.2104263305664   # set - 121 61692.038315 #62252.77287 
        elif srcId == 123 or srcId == 124:
            BF                  = 640.255615234375 * 94.83711242675781     # cube         
        elif srcId == 131:            
            BF                  = 650.4146118164062 * 94.85032653808594   # set - 131  
        elif srcId == 132:
            BF                  = 650.4146118164062 * 94.85032653808594   
        elif srcId == 500:
            BF                  = self.camera_bf  # using predefined BF
        elif srcId == 1:       
            BF                  = 1000  # do nothing
        else:
            raise ValueError('Bad srcId')   
        return BF
    
    def convert_disparity_to_depth(self, disparity, srcId = 121):
        "from GIL"
        BF                  = self.get_baseline_focal_product(srcId)
        disparity           = disparity.astype(np.float32) 
        depth               = np.zeros_like(disparity) 
        disparity_valid     = disparity > 0.1
        depth[disparity_valid]   = BF / disparity[disparity_valid]
        #depth[disparity_valid]   += 0.5  # LUT in the simulator
        return depth.astype(np.uint16)
    
    def convert_depth_to_disparity(self, depth, srcId = 121):
        "check image alignment from depth data"
        BF                  = self.get_baseline_focal_product(srcId)
        depth               = depth.astype(np.float32)        
        disparity           = np.zeros_like(depth) #- 0.5 #+ 0.5  # this is in the LUT
        depth_valid         = depth > 0.1
        disparity[depth_valid]   = BF / depth[depth_valid]
        return disparity #.astype(np.uint16)        
     
    def softmax_with_threshold(self, x: np.ndarray, dim: int = 0, T: float = 1.0, x_thr: float = -2.5) -> np.ndarray:
        """
        Apply softmax independently to dimension `dim` of a 3D array.
        x_thr - provides a minimal threshold for the input values, below which the output will be zero. This can help suppress noise and focus on more confident matches. 
        The threshold can be tuned based on the expected range of input values and the desired level of sparsity in the output.

        For matrices: 
        For column j: out[i, j] = exp(x[i, j] - max_j) / sum_i exp(x[i, j] - max_j)
        """
        #x_clipped = np.maximum(x, x_thr)  # clip values below threshold to x_thr
        # if x is below x_thr then exp(x_shifted / T) will be very small.
        x_eps     = np.exp(x_thr / T)  # shift by threshold to control sparsity
        # it means low probability.
        x_shifted = x #- np.maximum(x, axis=dim, keepdims=True)   # numerical stability
        exp_x     = np.exp(x_shifted / T)
        return exp_x / (np.sum(exp_x, axis=dim, keepdims=True) + x_eps)

    def kalman_pixel_fusion(self, img_A, p_A, img_B, p_B, epsilon=1e-8):
        """
        Fuses two images based on their pixel-wise probability matrices 
        using Kalman-like update equations.
        
        Parameters:
        -----------
        img_A : ndarray
            First image, shape (H, W) or (H, W, C).
        p_A   : ndarray
            Probability/confidence matrix for img_A, shape (H, W).
        img_B : ndarray
            Second image, same shape as img_A.
        p_B   : ndarray
            Probability/confidence matrix for img_B, shape (H, W).
        epsilon : float
            Small value to prevent division by zero.
            
        Returns:
        --------
        img_C : ndarray
            Fused image.
        p_C   : ndarray
            Fused probability matrix.
        """
        # Ensure probabilities match image spatial dimensions
        # If images are RGB (H, W, 3), we broadcast the weights across channels
        if len(img_A.shape) == 3 and len(p_A.shape) == 2:
            p_A_expanded = np.expand_dims(p_A, axis=-1)
            p_B_expanded = np.expand_dims(p_B, axis=-1)
        else:
            p_A_expanded = p_A
            p_B_expanded = p_B

        # 1. Compute Combined Probability Matrix (Posterior Precision)
        p_C = p_A + p_B
        p_C_expanded = p_A_expanded + p_B_expanded

        # 2. Compute Kalman Gain (Weight of measurement B relative to total precision)
        # Adding epsilon to denominator handles completely unconfident pixels safely
        K = p_B_expanded / (p_C_expanded + epsilon)

        # 3. Measurement Update Equation
        # C = A + K * (B - A)
        img_C = img_A + K * (img_B - img_A)

        # Handle edge case: if both p_A and p_B are 0, fall back to simple average
        zero_mask = (p_C_expanded == 0)
        if np.any(zero_mask):
            img_C = np.where(zero_mask, (img_A + img_B) / 2.0, img_C)

        return img_C, p_C
    
    def generate_gaussian_kernel(self,size=5, sigma=1.0):
        """Generates a 2D Gaussian kernel to serve as spatial weights."""
        ax = np.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, size)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
        kernel = np.outer(gauss, gauss)
        return kernel / np.sum(kernel)

    def kalman_neighborhood_fusion_two_images(self, img_A, p_A, img_B, p_B, kernel_size=5, sigma=1.5, epsilon=1e-8):
        """
        Fuses two images by aggregating pixel values and probabilities over a 
        local neighborhood using a Kalman-like formulation.
        
        Supports both grayscale (H, W) and multichannel (H, W, C) images.
        """

        # 1. Initialize spatial distance weights
        W_s = self.generate_gaussian_kernel(size=kernel_size, sigma=sigma)
        
        # 2. Compute effective neighborhood precisions (P'_A and P'_B)
        # This sums up (W_s * P) for every 5x5 neighborhood
        p_A_prime = correlate(p_A, W_s, mode='nearest')
        p_B_prime = correlate(p_B, W_s, mode='nearest')
        
        # Combined Precision Matrix
        p_C = p_A_prime + p_B_prime
        
        # 3. Compute confidence-weighted neighborhood images
        # Numerator: sum(W_s * P * Img)
        # We handle multichannel images by correlating channel-by-channel if needed
        # If images are RGB (H, W, 3), we broadcast the weights across channels
        if len(img_A.shape) == 2:
            img_A = np.expand_dims(img_A, axis=-1)
            img_B = np.expand_dims(img_B, axis=-1)
        
        # Broadcast p_prime shapes for element-wise division
        p_A_exp   = np.expand_dims(p_A_prime, axis=-1)
        p_B_exp   = np.expand_dims(p_B_prime, axis=-1)
        p_C_exp   = np.expand_dims(p_C, axis=-1)            

        num_channels = img_A.shape[2]
        sum_W_P_A = np.zeros_like(img_A)
        sum_W_P_B = np.zeros_like(img_B)
        for c in range(num_channels):
            sum_W_P_A[..., c] = correlate(img_A[..., c] * p_A, W_s, mode='nearest')
            sum_W_P_B[..., c] = correlate(img_B[..., c] * p_B, W_s, mode='nearest')

        # sum_W_P_A = img_A * p_A_exp
        # sum_W_P_B = img_B * p_B_exp
            
        # Compute localized mean estimations (A_bar and B_bar)
        A_bar = sum_W_P_A / (p_A_exp + epsilon)
        B_bar = sum_W_P_B / (p_B_exp + epsilon)
        
        # 4. Calculate Neighborhood Kalman Gain
        K = p_B_exp / (p_C_exp + epsilon)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(p_A, cmap='viridis')
        plt.colorbar()
        plt.title("Original p_A")
        plt.subplot(1, 3, 2)
        plt.imshow(p_B, cmap='viridis')
        plt.colorbar()
        plt.title("Original p_B")
        plt.subplot(1, 3, 3)
        plt.imshow(K.squeeze(), cmap='viridis')
        plt.colorbar()
        plt.title("Neighborhood Kalman Gain")
        plt.show(block=False)
        
        # 5. Measurement Update to produce fused image C
        img_C = A_bar + K * (B_bar - A_bar)
        
        # Edge case: If an entire neighborhood has zero confidence in both images,
        # fallback to a local spatial blur of the unweighted image midpoint
        # zero_mask = (p_C_exp == 0)
        # if np.any(zero_mask):
        #     fallback = np.zeros_like(img_C)
        #     if img_A.ndim == 3:
        #         for c in range(num_channels):
        #             fallback[..., c] = correlate((img_A[..., c] + img_B[..., c]) / 2.0, W_s, mode='nearest')
        #     else:
        #         fallback = correlate((img_A + img_B) / 2.0, W_s, mode='nearest')
        #     img_C = np.where(zero_mask, fallback, img_C)
            
        return img_C, p_C    

    def kalman_neighborhood_fusion(self, img_A, p_A, kernel_size=5):
        """
        Fuses two images by aggregating pixel values and probabilities over a 
        local neighborhood using a Kalman-like formulation.
        
        Supports both grayscale (H, W) and multichannel (H, W, C) images.
        """
        epsilon     = 1e-8

        if len(img_A.shape) != len(p_A.shape):
            img_A = np.expand_dims(img_A, axis=-1)
            p_A   = np.expand_dims(p_A, axis=-1)    

        # Apply using convolution
        k_size = kernel_size
        #kernel = np.ones((k_size, k_size, k_size)) / (k_size ** 3) 
        if len(img_A.shape) == 3:
            kernel = np.ones((k_size, k_size, k_size)) 
        else:     
            kernel = np.ones((k_size, k_size)) 
         

        # 1. Initialize spatial distance weights
        #sum_P_A   = ndimage.uniform_filter(p_A * img_A, size=kernel_size)
        sum_P_A = ndimage.convolve(p_A * img_A, kernel, mode='reflect')           
        
        # 2. Compute effective neighborhood precisions (P'_A )
        #sum_P   = ndimage.uniform_filter(p_A, size=kernel_size)
        sum_P   = ndimage.convolve(p_A, kernel, mode='reflect')        

        # 3. Intensity
        img_C    = sum_P_A / (sum_P + epsilon)

        # 4. Combined Probability Matrix
        #sum_P_P   = ndimage.uniform_filter(p_A * p_A, size=kernel_size)
        sum_P_P   = ndimage.convolve(p_A * p_A, kernel, mode='reflect')
        p_C      = sum_P_P / (sum_P + epsilon)
        
        
        return img_C, p_C    

    def joint_bilateral_upsampling(self, lr_img, hr_guide, spatial_sigma=3.0, range_sigma=0.1, radius=4):
        # =====================================================================
        # 1. High-Performance NumPy JBU Implementation
        # =====================================================================
        log.info('Starting joint bilateral upsampling')
        if lr_img.ndim == 2:
            lr_img = lr_img[..., np.newaxis]
        if hr_guide.ndim == 2:
            hr_guide = hr_guide[..., np.newaxis]
            
        H_lr, W_lr, C_lr = lr_img.shape
        H_hr, W_hr, C_g = hr_guide.shape
        
        scale_y = H_hr / H_lr
        scale_x = W_hr / W_lr
        
        # Precompute spatial Gaussian weights
        y_coords, x_coords = np.mgrid[-radius:radius+1, -radius:radius+1]
        spatial_dist_sq = y_coords**2 + x_coords**2
        spatial_weights = np.exp(-spatial_dist_sq / (2 * spatial_sigma**2))
        
        upsampled = np.zeros_like(hr_guide, dtype=np.float32) if C_lr == C_g else np.zeros((H_hr, W_hr, C_lr), dtype=np.float32)
        norm_factor = np.zeros((H_hr, W_hr, 1), dtype=np.float32)
        hr_y, hr_x = np.meshgrid(np.arange(H_hr), np.arange(W_hr), indexing='ij')
        
        # Neighborhood search loop
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                s_w = spatial_weights[dy + radius, dx + radius]
                if s_w < 1e-4:
                    continue
                
                lr_y_idx        = np.clip(np.round(hr_y / scale_y).astype(np.int32) + dy, 0, H_lr - 1)
                lr_x_idx        = np.clip(np.round(hr_x / scale_x).astype(np.int32) + dx, 0, W_lr - 1)
                
                lr_val          = lr_img[lr_y_idx, lr_x_idx, :]
                
                hr_y_from_lr    = np.clip(np.round(lr_y_idx * scale_y).astype(np.int32), 0, H_hr - 1)
                hr_x_from_lr    = np.clip(np.round(lr_x_idx * scale_x).astype(np.int32), 0, W_hr - 1)
                
                guide_diff      = hr_guide - hr_guide[hr_y_from_lr, hr_x_from_lr]
                range_dist_sq   = np.sum(guide_diff**2, axis=-1, keepdims=True)
                range_weights   = np.exp(-range_dist_sq / (2 * range_sigma**2))
                
                weight          = s_w * range_weights
                upsampled      += lr_val * weight
                norm_factor    += weight
                
        upsampled /= (norm_factor + 1e-8)
        log.info('Done')
        return np.squeeze(upsampled)    

    def joint_bilateral_filtering(self, img_left, img_volume, spatial_sigma=3.0, range_sigma=0.1, radius=4, iter_num=1):
        # =====================================================================
        # 1. High-Performance NumPy JBU Implementation
        # =====================================================================
        log.info('Starting joint bilateral filtering with spatial_sigma=%.2f, range_sigma=%.2f, radius=%d, iter_num=%d', spatial_sigma, range_sigma, radius, iter_num)
        if img_left.ndim == 2:
            img_left = img_left[..., np.newaxis]
        if img_volume.ndim == 2:
            img_volume = img_volume[..., np.newaxis]
            
        H_lr, W_lr, C_lr    = img_left.shape
        H_hr, W_hr, C_hr    = img_volume.shape

        
        # Precompute spatial Gaussian weights
        y_coords, x_coords  = np.mgrid[-radius:radius+1, -radius:radius+1]
        spatial_dist_sq     = y_coords**2 + x_coords**2
        spatial_weights     = np.exp(-spatial_dist_sq / (2 * spatial_sigma**2))
        
        #img_volume_filtered = img_volume.copy() #np.zeros_like(img_volume, dtype=np.float32) #if C_lr == C_hr else np.zeros((H_hr, W_hr, C_lr), dtype=np.float32)
        
        #hr_y, hr_x     = np.meshgrid(np.arange(H_hr), np.arange(W_hr), indexing='ij')
        y_index             = np.arange(radius, H_hr - radius - 1).reshape(-1, 1)
        x_index             = np.arange(radius, W_hr - radius - 1).reshape(1, -1)
        
        # Neighborhood match loop
        for k in range(iter_num):
            img_volume_filtered     = np.zeros_like(img_volume, dtype=np.float32)
            norm_factor             = np.zeros_like(img_left, dtype=np.float32)
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    s_w             = spatial_weights[dy + radius, dx + radius]

                    #lr_val         = img_volume[y_index, x_index,:] - img_volume[y_index + dy, x_index + dx,:]
                    lr_val          = img_volume[y_index + dy, x_index + dx,:]
                    guide_diff      = img_left[y_index, x_index,:] - img_left[y_index + dy, x_index + dx,:]
                    range_weights   = np.exp(-guide_diff**2 / (2 * range_sigma**2))
                    
                    weight                                      = s_w * range_weights #* 1.1 - 0.1 # -0.1 high pass
                    img_volume_filtered[y_index, x_index,:]    += lr_val * weight
                    norm_factor[y_index, x_index,:]            += weight                
                    
            img_volume_filtered /= (norm_factor + 1e-8)
            img_volume = img_volume_filtered.copy()
        log.info('Done')
        return np.squeeze(img_volume_filtered)  

    def probability_bilateral_filtering(self, img_left, prob_volume, spatial_sigma=3.0, range_sigma=0.1, radius=4, iter_num=1):
        # =====================================================================
        # 1. fills low probabilities using neighbrhood information defined by the left image
        # =====================================================================
        if img_left.ndim == 2:
            img_left = img_left[..., np.newaxis]
        if prob_volume.ndim == 2:
            prob_volume = prob_volume[..., np.newaxis]
            
        H_lr, W_lr, C_lr    = img_left.shape
        H_hr, W_hr, C_hr    = prob_volume.shape
        prob_volume         = np.clip(prob_volume, 0.0, 1.0)

        
        # Precompute spatial Gaussian weights
        y_coords, x_coords  = np.mgrid[-radius:radius+1, -radius:radius+1]
        spatial_dist_sq     = y_coords**2 + x_coords**2
        spatial_weights     = np.exp(-spatial_dist_sq / (2 * spatial_sigma**2))
        
        #img_volume_filtered = img_volume.copy() #np.zeros_like(img_volume, dtype=np.float32) #if C_lr == C_hr else np.zeros((H_hr, W_hr, C_lr), dtype=np.float32)
        
        #hr_y, hr_x     = np.meshgrid(np.arange(H_hr), np.arange(W_hr), indexing='ij')
        y_index             = np.arange(radius, H_hr - radius - 1).reshape(-1, 1)
        x_index             = np.arange(radius, W_hr - radius - 1).reshape(1, -1)
        
        # Neighborhood match loop
        for k in range(iter_num):
            prob_volume_filtered    = np.zeros_like(prob_volume, dtype=np.float32)
            norm_factor             = np.zeros_like(img_left, dtype=np.float32)
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    s_w             = spatial_weights[dy + radius, dx + radius]

                    #lr_val         = prob_volume[y_index, x_index,:] - prob_volume[y_index + dy, x_index + dx,:]
                    prob_val        = (1-prob_volume[y_index, x_index,:]) * prob_volume[y_index + dy, x_index + dx,:]
                    guide_diff      = img_left[y_index, x_index,:] - img_left[y_index + dy, x_index + dx,:]
                    range_weights   = np.exp(-guide_diff**2 / (2 * range_sigma**2))
                    
                    weight                                      = s_w * range_weights - 0.1 # -0.1 high pass
                    prob_volume_filtered[y_index, x_index,:]    += prob_val * weight
                    norm_factor[y_index, x_index,:]             += weight                
                    
            prob_volume_filtered /= (norm_factor + 1e-8)
            prob_volume = prob_volume_filtered.copy()
        return np.squeeze(prob_volume_filtered)  

    #%% -----------------------------------------
    def compute_edges(self, img = None):
        "edges of the image"
        if img is None:
            raise ValueError('image is not defined')

        # Convert to grayscale, as edge detection works on intensity
        img = img if len(img.shape)<3 else cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # 2. Use a Gaussian Blur for noise reduction (optional, but recommended)
        img_blur = cv.GaussianBlur(img, (3, 3), 0) # 5x5 kernel size
        #img_blur  = cv.pyrDown(img)
        #img_blur  = cv.pyrDown(img_blur)
        #img_blur  = cv.pyrDown(img_blur)

        # 3. Calculate the Sobel Derivatives
        # ddepth=cv.CV_64F is used for accuracy to prevent overflow, as derivatives can be negative.
        # ksize=3 is the 3x3 kernel size.

        # Sobel X (Detects vertical edges)
        sobelx = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=1, dy=0, ksize=3)

        # Sobel Y (Detects horizontal edges)
        sobely = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=0, dy=1, ksize=3)

        # 4. Convert back to 8-bit image and find the absolute value
        # The original Sobel output contains negative values, so we take the absolute value
        # and scale it back to the 0-255 range for display.
        #abs_sobelx = np.clip(sobelx + 128, 0, 255) #cv.convertScaleAbs(sobelx)
        #abs_sobely = np.clip(sobely + 128, 0, 255) #cv.convertScaleAbs(sobely)
        abs_sobelx = cv.convertScaleAbs(sobelx)
        abs_sobely = cv.convertScaleAbs(sobely)        

        # 5. Combine the X and Y gradient components
        # The final edge map (magnitude) is a weighted combination of the X and Y results.
        # Gradient Magnitude G = sqrt(Gx^2 + Gy^2) (or approximated as |Gx| + |Gy|)
        #sobel_combined = cv.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)
        sobel_combined = np.hypot(sobelx, sobely)  # more accurate magnitude

        #sobel_combined  = cv.pyrUp(sobel_combined)
        #sobel_combined  = cv.pyrUp(sobel_combined)      
        #sobel_combined  = cv.pyrUp(sobel_combined)  
        # img_edges       = np.zeros((img.shape[0], img.shape[1],3), dtype=np.float32)
        # img_edges[:,:,0] = sobelx
        # img_edges[:,:,1] = sobely
        # img_edges[:,:,2] = sobel_combined

        # # 6. Display the results
        # titles = ['Original Image', 'Sobel X (Vertical Edges)', 'Sobel Y (Horizontal Edges)', 'Combined Sobel Edges']
        # images = [img, abs_sobelx, abs_sobely, sobel_combined]
        # self.show_subset(images, titles)

        return sobel_combined

    def pixel_features(self, img, feat_type = 'center'):
        """
        Converts the image to a hashable feature. Each image offset is compared to each other
        """
        # Convert image to grayscale if it is not already
        img_h, img_w    = img.shape[:2]
        if img_h < 32 or img_w < 32:
            raise ValueError("Image must be at least 32x32 pixels in size.")
        
        
        # Apply box filter
        #filtered_img    = self.apply_box_filter(img)
        filtered_img    = img.copy().astype(np.float32)

        # Create compares the central pixel with the rest according to the offset
        if feat_type == 'center':
            pixel_offsets   = np.array([(1, 0), (0, 1), (-1, 0), (0, -1)])
        elif feat_type == 'right':
            pixel_offsets   = np.array([(2, 0), (1, 0), (2, 1), (2, -1)]) # bias right
        elif feat_type == 'left':
            pixel_offsets   = np.array([(-2, 0), (-1, 0), (-2, 1), (-2, -1)]) # bias left
        elif feat_type == 'up':
            pixel_offsets   = np.array([(-1, 2), (0, 2), (1, 2), (0, 1)]) # bias up     
        elif feat_type == 'down':
            pixel_offsets   = np.array([(-1, -2), (0, -2), (1, -2), (0, -1)]) # bias down  
        elif feat_type == 'center_big':
            pixel_offsets   = np.array([(3, 0), (0, 3), (-3, 0), (0, -3)])                               
        else:
            pixel_offsets   = np.array([(1, 0), (0, 1), (-1, 0), (0, -1)])
        #pixel_offsets   = np.array([(2, 0), (1, 0), (2, 1), (2, -1)]) # bias right
        #pixel_offsets   = np.array([(1, 0), (0, 1), (-1, 0), (0, -1), (3, 0), (0, 2), (-3, 0), (0, -2)])*3
        #pixel_offsets   = np.array([(1, 0), (-1, 0), (3, 0), (-3, 0), (7, 0), (-7, 0)])*2
        max_offset      = np.max(np.abs(pixel_offsets))+1
        num_offsets    = pixel_offsets.shape[0]

        # Initialize the hash values
        img_feat        = np.zeros((img_h, img_w, num_offsets), dtype=np.float32)
        bit_index       = 0
        center_img      = filtered_img[max_offset:-max_offset,max_offset:-max_offset]
        for k in range(num_offsets):
            offset      = pixel_offsets[k]
            # image difference
            offset_img  = filtered_img[max_offset + offset[1]:-max_offset + offset[1], max_offset + offset[0]:-max_offset + offset[0]]
            diff_img    = offset_img #- center_img
            #sign_img    = np.sign(diff_img)  # Get the sign of the difference
            img_feat[max_offset:-max_offset,max_offset:-max_offset,k] = diff_img

        #print(f"Created complex hash with {bit_index} bits.")
        return img_feat 

    def inverse_filter_with_edges(self, prob_distance, img_left, num_iter = 8):
        """
        Makes the inversion on the edges before interpolation
        """        
        num_iter    = np.maximum(4,num_iter)
        delta       = 0.24
        kappa       = 50
        s           = 1
        log.debug(f'Starting inversion filtering with num_iter={num_iter}, delta={delta}, kappa={kappa}')

        img_blur   = cv.GaussianBlur(img_left, (3, 3), 0) # 5x5 kernel size
        grad_img_S  = cv.Sobel(img_blur, cv.CV_32F, 0, 1, ksize=3)
        grad_img_E  = cv.Sobel(img_blur, cv.CV_32F, 1, 0, ksize=3)  
        img_edges   = np.hypot(grad_img_S, grad_img_E)  # more accurate magnitude

        #img_edges   = self.compute_edges(img_left)
        grad_img    = np.exp(-img_edges/ kappa)
        grad_img    = grad_img - 0.2

        out         = prob_distance * grad_img[:, :, np.newaxis] 


        log.debug('Finished anisotropic diffusion filtering after %d iterations', num_iter)
        return out

    def softmax_local_maxima(self, img = None, kernel_size = 7, T = 1.0, x_thr = -2.5):
        "compute local spatial maxima of an image"
        if img is None:
            raise ValueError('image is not defined')
        
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Convert image to float to avoid issues with arithmetic operations
        img_float = img.astype(np.float32)

        # 2. Perform Dilation
        # Dilation replaces each pixel with the MAXIMUM pixel value found in its 
        # surrounding neighborhood defined by the kernel.
        
        # Create the structuring element (kernel)
        kernel      = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Apply dilation
        # The result 'dilated_img' contains, at every pixel (x, y), the maximum value 
        # of the original image within the kernel_size window centered at (x, y).
        img_max    = cv.dilate(img_float, kernel, iterations=1)
        
        # 3. Find Maxima using Comparison
        # A pixel (x, y) in the original image is a TRUE local maximum only if:
        # img_float[x, y] == dilated_img[x, y]
        # AND it is NOT equal to a plateau of adjacent equal-valued pixels.
        
        # To enforce strict local maxima (strictly greater than neighbors):
        # We compare the original image with its dilated version MINUS a small epsilon.
        # epsilon = 0.0001
        
        # Boolean mask: True where original pixel value > maximum of its neighbors (excluding itself potentially)
        # The use of 'dilated_img' ensures comparison against ALL neighbors.
        #maxima_mask = (img_float > (dilated_img - epsilon))        
        x_eps     = np.exp(x_thr / T)  # shift by threshold to control sparsity
        # it means low probability.
        x_shifted = img_max - img_float #- np.maximum(x, axis=dim, keepdims=True)   # numerical stability
        exp_x     = np.exp(-x_shifted / T)

        img_prob =  exp_x #/ (np.sum(exp_x, axis=(0,1), keepdims=True) + x_eps)

        # local maxima mask
        #local_max_mask          = (maxima_mask*250).astype(np.uint8)

        # titles = ['Edge Magnitudes', 'Dilated Edges', 'Local Maxima Mask']
        # images = [img, dilated_img, local_max_mask]  
        # self.show_images_debug(images, titles, False)  

        return img_prob

    #%% -----------------------------------------

    def gabor_bank_channels(
        self,
        img_gray,
        ksize=21,
        sigma=4.0,
        lambdas=None,
        gamma=0.5,
        psis=None,
        thetas=None,
    ):
        """
        Process an image through a Gabor filter bank and return multi-channel output.
        
        Applies a bank of Gabor filters to an input image and returns all responses
        stacked as a single (H, W, C) array where C is the number of filters.
        
        Parameters
        ----------
        img_gray : np.ndarray
            Input grayscale image with shape (H, W). Any numeric dtype is accepted.
        ksize : int, default=21
            Gabor kernel size (will be made odd).
        sigma : float, default=4.0
            Gaussian standard deviation.
        lambdas : list of float, default=[8.0, 16.0, 32.0]
            Wavelengths for the Gabor filters.
        gamma : float, default=0.5
            Aspect ratio of the Gabor envelope.
        psis : list of float, default=[0.0, π/2]
            Phase offsets for the Gabor filters.
        thetas : list of float, default=[0, π/4, π/2, 3π/4]
            Orientations for the Gabor filters.
            
        Returns
        -------
        np.ndarray
            Multi-channel response array of shape (H, W, C) where C is the total
            number of Gabor filters applied. Values are uint8 in range [0, 255].
        """
        if img_gray is None:
            raise ValueError("img_gray must not be None")
        if img_gray.ndim != 2:
            raise ValueError("img_gray must be a 2D grayscale image")
        
        if lambdas is None:
            lambdas = [8.0, 16.0, 32.0]
        if psis is None:
            psis = [0.0, np.pi / 2]
        if thetas is None:
            thetas = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        
        # Ensure kernel size is odd
        ksize = int(ksize)
        if (ksize % 2) == 0:
            ksize += 1
        
        # Convert to float32
        img = img_gray.astype(np.float32, copy=False)
        h, w = img.shape
        
        # Build Gabor bank
        bank = []
        for theta in thetas:
            for lambd in lambdas:
                for psi in psis:
                    kernel = cv.getGaborKernel(
                        ksize=(ksize, ksize),
                        sigma=sigma,
                        theta=theta,
                        lambd=lambd,
                        gamma=gamma,
                        psi=psi,
                        ktype=cv.CV_32F,
                    )
                    knorm = np.sum(np.abs(kernel))
                    #print(f"Created Gabor kernel with theta={theta:.2f}, lambda={lambd:.2f}, psi={psi:.2f}, norm={knorm:.2f}")
                    bank.append(kernel/knorm)
        
        # Apply all filters and collect responses
        responses = []
        for kernel in bank:
            filtered = cv.filter2D(img, cv.CV_32F, kernel)
            # Normalize to uint8 [0, 255]
            #filtered_abs = np.abs(filtered)
            #filtered_norm = cv.normalize(filtered_abs, None, 0, 255, cv.NORM_MINMAX)
            #responses.append(filtered_norm.astype(np.uint8))
            responses.append(filtered)
        
        # Stack responses into (H, W, C) array
        output = np.stack(responses, axis=2)
        
        return output
        
    def normalize_kernel(self,kernel: np.ndarray) -> np.ndarray:
        kernel      = kernel.astype(np.float32)
        kernel_mean = kernel - np.mean(kernel)
        kernel_norm = kernel_mean/cv.norm(kernel_mean, cv.NORM_L2)
        return kernel_norm

    def gabor_bank_init(
        self,
        ksize=21,
        sigma=4.0,
        lambdas=None,
        gamma=0.5,
        psis=None,
        thetas=None,
    ):
        """
        Process an image through a Gabor filter bank and return multi-channel output.
        
        Applies a bank of Gabor filters to an input image and returns all responses
        stacked as a single (H, W, C) array where C is the number of filters.
        
        Parameters
        ----------
        img_gray : np.ndarray
            Input grayscale image with shape (H, W). Any numeric dtype is accepted.
        ksize : int, default=21
            Gabor kernel size (will be made odd).
        sigma : float, default=4.0
            Gaussian standard deviation.
        lambdas : list of float, default=[8.0, 16.0, 32.0]
            Wavelengths for the Gabor filters.
        gamma : float, default=0.5
            Aspect ratio of the Gabor envelope.
        psis : list of float, default=[0.0, π/2]
            Phase offsets for the Gabor filters.
        thetas : list of float, default=[0, π/4, π/2, 3π/4]
            Orientations for the Gabor filters.
            
        Returns
        -------
        np.ndarray
            Multi-channel response array of shape (H, W, C) where C is the total
            number of Gabor filters applied. Values are uint8 in range [0, 255].
        """

        
        if lambdas is None:
            lambdas = [8.0, 16.0, 32.0]
        if psis is None:
            psis = [0.0, np.pi / 2]
        if thetas is None:
            thetas = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        
        # Ensure kernel size is odd
        ksize = int(ksize)
        if (ksize % 2) == 0:
            ksize += 1

        
        # Build Gabor bank
        bank = []; params = []
        for theta in thetas:
            for lambd in lambdas:
                for psi in psis:
                    kernel = cv.getGaborKernel(
                        ksize=(ksize, ksize),
                        sigma=sigma,
                        theta=theta,
                        lambd=lambd,
                        gamma=gamma,
                        psi=psi,
                        ktype=cv.CV_32F,
                    )
                    #knorm = np.sum(np.abs(kernel))
                    kernel = self.normalize_kernel(kernel)
                    bank.append(kernel)
                    #print(f"Created Gabor kernel with theta={theta:.2f}, lambda={lambd:.2f}, psi={psi:.2f}, norm={knorm:.2f}")
                    params.append(
                    {
                        "theta": theta,
                        "lambda": lambd,
                        "psi": psi,
                    }
                )
        
        return bank, params
        
    def gabor_bank_filter(self, img_gray, bank):    

        """
        Process an image through a Gabor filter bank and return multi-channel output.
        
        Applies a bank of Gabor filters to an input image and returns all responses
        stacked as a single (H, W, C) array where C is the number of filters.
        
        Parameters
        ----------
        img_gray : np.ndarray
            Input grayscale image with shape (H, W). Any numeric dtype is accepted.
        bank : list of np.ndarray
            Precomputed Gabor filter bank.
            
        Returns
        -------
        np.ndarray
            Multi-channel response array of shape (H, W, C) where C is the total
            number of Gabor filters applied. Values are uint8 in range [0, 255].
        """
        if img_gray is None:
            raise ValueError("img_gray must not be None")
        if img_gray.ndim != 2:
            raise ValueError("img_gray must be a 2D grayscale image")
        
        
        # Convert to float32
        img = img_gray.astype(np.float32, copy=False)
        h, w = img.shape
        
        
        # Apply all filters and collect responses
        responses = []
        for kernel in bank:
            filtered = cv.filter2D(img, cv.CV_32F, kernel)
            # Normalize to uint8 [0, 255]
            #filtered_abs = np.abs(filtered)
            #filtered_norm = cv.normalize(filtered_abs, None, 0, 255, cv.NORM_MINMAX)
            #responses.append(filtered_norm.astype(np.uint8))
            responses.append(filtered)
        
        # Stack responses into (H, W, C) array
        output = np.stack(responses, axis=2)
        
        return output

    #%% -----------------------------------------
    def calculate_dispartity_difference(self, img_left, img_right, min_disparity = 1, max_disparity = 64):
        "shifts the right image to the left and for each shift computes the difference between the left image and the shifted "
        "right image"
        #img_array = np.zeros((img_right.shape[0], img_right.shape[1], max_disparity), dtype=np.float32)
        log.debug(f'Calculating disparity difference with max_disparity {max_disparity}')
        h, w            = img_right.shape[:2]
        img_array       = np.full((h, w, max_disparity), 80, dtype=np.float32)
        
        for shift in range(min_disparity, max_disparity):
            img_array[:, shift:, shift] = img_right[:, :w - shift] 

        img_array       = img_array - img_left[:,:,np.newaxis]  # broadcasting to compute difference for each shift
        #img_array = np.abs(img_array)  # take absolute value of the difference

        # for shift in range(min_disparity,max_disparity):
        #     img_array[:,:,shift]      = cv.boxFilter(img_array[:,:,shift], -1,   (kernel_size, kernel_size), normalize=True) 

        #img_array      = cv.boxFilter(img_array, -1,   (kernel_size, kernel_size), normalize=True) 
        log.debug('Done')
        return img_array    

    def gabor_dispartity(self, bank_left, bank_right, max_disparity = 64):
        "bank_left, bank_right are NxMxC arrays of gabor responses for the left and right images, respectively. "
        "This function computes a disparity cost volume by comparing the left and right responses across a range of disparities. "
        "For each disparity, it shifts the right bank to the left and for each shift computes the difference between the left bank and the shifted "
        "right bank"

        log.debug(f'Calculating disparity difference with max_disparity {max_disparity}')
        bank_left  = bank_left[..., np.newaxis]  if bank_left.ndim == 2  else bank_left
        bank_right = bank_right[..., np.newaxis] if bank_right.ndim == 2 else bank_right
        max_disparity = int(max_disparity)

        h, w            = bank_right.shape[:2]
        diff_array       = np.full((h, w, max_disparity), 1, dtype=np.float32)
        
        for shift in range(max_disparity):
            bank_diff   = bank_left[:, shift:, :] - bank_right[:, :w - shift,:] 
            diff_array[:, shift:, shift] = np.mean(np.abs(bank_diff), axis=2)

        log.debug('Done')
        return diff_array   
    
    def gabor_normalize_responses(self, gabor_left):
         # This can help improve the correlation by making it more about the pattern of responses across channels rather 
         # than their absolute strength.  SQRT is important, we want to differentiate between strong and no strong responses, but we don't want the strongest responses to dominate the correlation.
        gabor_left_norm          = np.sqrt(np.linalg.norm(gabor_left, axis=2, keepdims=True)) + 1e-6
        #gabor_left_norm          = np.linalg.norm(gabor_left, axis=2, keepdims=True) + 1e-6
        gabor_left               = gabor_left / gabor_left_norm

        return gabor_left

    def gabor_normalize_responses_with_energy(self, gabor_left):
         # This can help improve the correlation by making it more about the pattern of responses across channels rather 
         # than their absolute strength.  SQRT is important, we want to differentiate between strong and no strong responses, but we don't want the strongest responses to dominate the correlation.
        gabor_left_norm          = np.sqrt(np.linalg.norm(gabor_left, axis=2, keepdims=True)) + 1e-6
        #gabor_left_norm          = np.linalg.norm(gabor_left, axis=2, keepdims=True) + 1e-6
        gabor_left               = gabor_left / gabor_left_norm

        return gabor_left, gabor_left_norm.squeeze()   

    def gabor_decomposition_variability(self, gaborL, kernel_size = 7):
        "compute variability of the gabor responses across channels for each pixel, which can be used as a measure of local information content. This can help identify pixels that are more likely to provide reliable matches, so we can use this information to weight the correlation scores."
        gaborL_aver         = cv.boxFilter(gaborL, -1,   (kernel_size, kernel_size), normalize=True) 
        gaborL_diff         = gaborL - gaborL_aver
        variabilityL        = cv.boxFilter(gaborL_diff**2, -1,   (kernel_size, kernel_size), normalize=True)  
        informationL        = np.mean(variabilityL, axis=2)  # shape (N, M)
        return informationL

    def debug_gabor_image_disparity_multiscale(self, debug_levels: list[dict], prob_total: np.ndarray, row_index: int):
        "display per-level debug views for a selected row, similar to test_gabor_line_correlation_multiscale"
        if row_index is None or len(debug_levels) == 0:
            return

        row_full       = int(np.clip(row_index, 0, prob_total.shape[0] - 1))


        for k, level_data in enumerate(debug_levels):
            gabor_left    = level_data["gabor_left"]
            gabor_right   = level_data["gabor_right"]
            dist_left     = level_data["distance_left"]  # shape (M, M, D)
            info_left     = level_data["info_left"]
            prob_left    = level_data["prob_left"]   # shape (N, M, D)
            prob_filt     = level_data["prob_context"] 
            img_left_lvl  = level_data.get("img_left")
            img_right_lvl = level_data.get("img_right")

            prob_so_far   = prob_total[row_full, :, :(k+1), :]
            final_prob_row = np.sum(prob_so_far, axis=1)   # shape (M, D)
            final_disp_row = np.argmax(final_prob_row, axis=1)  # shape (M,)

            row_level  = int(np.clip(row_full // (2 ** k), 0, gabor_left.shape[0] - 1))

            rowL       = gabor_left[row_level, :, :].astype(np.float32)   # shape (M, C)
            rowR       = gabor_right[row_level, :, :].astype(np.float32)  # shape (M, C)
            dist_disp  = dist_left[row_level, :, :].T                    # shape (D, M)
            prob_row   = prob_left[row_level, :, :].T                    # shape (D, M)
            prob_filt_row = prob_filt[row_level, :, :].T
            info_row   = info_left[row_level, :]                          # shape (M,)
            confidence = np.max(prob_left[row_level, :, :], axis=1)      # shape (M,)

            # 1. Stacked images with highlighted row

            img_show = np.concatenate((img_left_lvl, img_right_lvl), axis=0).copy()
            img_show[[row_level, row_level + img_left_lvl.shape[0]], :] = 255
            plt.figure(figsize=(8, 10))
            plt.imshow(img_show, cmap='gray')
            plt.title(f'Image L-R : Level {k}, row {row_level}')

            # 2. Gabor row features as line plots
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(rowL, '-.')
            plt.title(f'Gabor row features: Left image, Level {k}, row {row_level}')
            plt.subplot(1, 2, 2)
            plt.plot(rowR, '-.')
            plt.title(f'Gabor row features: Right image, Level {k}, row {row_level}')
            plt.tight_layout()

            plt.figure(figsize=(8, 4))
            plt.imshow(dist_disp, cmap='viridis', aspect='auto')
            plt.colorbar(label='Inner difference over C channels')
            plt.xlabel('Right image column index')
            plt.ylabel('Left image column index')
            plt.title(f'Gabor row correlation matrix (diagonal band) at row Level {k}, row {row_level}')
            plt.tight_layout()
            plt.show(block=False)              

            # 3. Confidence and information
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(confidence, '-o', markersize=3)
            plt.title(f'Confidence: Level {k}, row {row_level}')
            plt.xlabel('Left image column index')
            plt.ylabel('Confidence')
            plt.subplot(1, 2, 2)
            plt.plot(info_row, '-o', markersize=3)
            plt.title(f'Information: Level {k}, row {row_level}')
            plt.xlabel('Left image column index')
            plt.ylabel('Information')
            plt.tight_layout()

            # 4. Probability diagonal band (disparity x column) at this row
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(prob_row, cmap='viridis', aspect='auto')
            plt.colorbar(label='Probability current level')
            plt.title(f'Probability band at Level {k}, row {row_level}')
            plt.subplot(1, 2, 2)
            plt.imshow(prob_filt_row, cmap='viridis', aspect='auto') 
            plt.colorbar(label='Probability current level')           
            plt.title(f'Probability filtered at Level {k}, row {row_level}')
            plt.tight_layout()
            plt.show(block=False)

            # 5. Probability current level vs accumulated final
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(prob_row, cmap='viridis', aspect='auto')
            plt.colorbar(label='Probability current level')
            plt.title(f'Probability current level, Level {k}, row {row_level}')
            plt.subplot(1, 2, 2)
            plt.imshow(final_prob_row.T, cmap='viridis', aspect='auto')
            plt.colorbar(label='Probability accumulated')
            plt.title(f'Probability accumulated (full row {row_full})')
            plt.tight_layout()

        # Final: best disparity at the debug row
        plt.figure(figsize=(12, 3))
        plt.plot(final_disp_row)
        plt.title(f'Final best disparity - full row {row_full}')
        plt.xlabel('Column')
        plt.ylabel('Disparity index')
        plt.tight_layout()
        plt.show()

    def anisotropic_filter_not_memory_efficient(self, prob_distance, img_left):
        """
        Perona-Malik Anisotropic Diffusion.
        img: 2D grayscale float array
        kappa: Edge threshold (higher = smoother, lower = preserves sharper edges)
        """        
        num_iter = 8
        delta    = 0.2
        kappa    = 0.03
        log.debug(f'Starting anisotropic diffusion filtering with num_iter={num_iter}, delta={delta}, kappa={kappa}')

        out       = prob_distance #.astype(np.float32)
        
        for i in range(num_iter):
            # Calculate gradients in North, South, East, West directions
            grad_N = np.roll(out, -1, axis=0) - out
            grad_S = np.roll(out,  1, axis=0) - out
            grad_E = np.roll(out, -1, axis=1) - out
            grad_W = np.roll(out,  1, axis=1) - out
            grad_U = np.roll(out, -1, axis=2) - out
            grad_D = np.roll(out,  1, axis=2) - out            

            # # add spatial filtering to the gradients to make them more robust to noise
            # grad_N = cv.GaussianBlur(grad_N, (3, 3), 0)
            # grad_S = cv.GaussianBlur(grad_S, (3, 3), 0)
            # grad_E = cv.GaussianBlur(grad_E, (3, 3), 0)
            # grad_W = cv.GaussianBlur(grad_W, (3, 3), 0)

            # grad_N = cv.Sobel(out, cv.CV_32F, 1, 0, ksize=3)
            # grad_S = cv.Sobel(out, cv.CV_32F, 1, 0, ksize=3)
            # grad_E = cv.Sobel(out, cv.CV_32F, 0, 1, ksize=3)
            # grad_W = cv.Sobel(out, cv.CV_32F, 0, 1, ksize=3)
            
            # Exponential conduction function
            c_N = np.exp(-(grad_N / kappa)**2)
            c_S = np.exp(-(grad_S / kappa)**2)
            c_E = np.exp(-(grad_E / kappa)**2)
            c_W = np.exp(-(grad_W / kappa)**2)
            c_U = np.exp(-(grad_U / kappa)**2)
            c_D = np.exp(-(grad_D / kappa)**2)
            
            # Update image
            out += delta * (c_N * grad_N + c_S * grad_S + c_E * grad_E + c_W * grad_W + c_U * grad_U + c_D * grad_D)
            
        log.debug('Finished anisotropic diffusion filtering after %d iterations', num_iter)
        return out

    def anisotropic_filter_3d(self, prob_distance, kappa = 0.1, img_left=None):
        """
        Perona-Malik Anisotropic Diffusion.
        img: 2D grayscale float array
        kappa: Edge threshold (higher = smoother, lower = preserves sharper edges)
        """        
        num_iter = 4
        delta    = 0.1
        
        log.debug(f'Starting anisotropic diffusion filtering with num_iter={num_iter}, delta={delta}, kappa={kappa}')

        out_new    = prob_distance.copy() #.astype(np.float32)
        directions = [0,1]
        shift      = [-1,1]
        
        for i in range(num_iter):
            out         = out_new.copy()
            #out_new[:]  = 0
            for d in directions:
                for s in shift:
                    grad = np.roll(out, s, axis=d) - out
                    c = np.exp(-(grad / kappa)**2)
                    out_new += delta * c * grad

            
        log.debug(f'Finished anisotropic diffusion filtering after {num_iter} iterations')
        return out_new
    
    def anisotropic_filter_prob(self, prob, img_left=None):
        """
        Perona-Malik Anisotropic Diffusion adapted for probability function.
        img: 2D grayscale float array
        kappa: Edge threshold (higher = smoother, lower = preserves sharper edges)
        """        
        num_iter = 8
        delta    = 0.25
        kappa    = 0.1
        log.debug(f'Starting anisotropic diffusion filtering with num_iter={num_iter}, delta={delta}, kappa={kappa}')

        out_new    = prob #.astype(np.float32)
        directions = [0,1]
        shift      = [-1,1]
        
        for i in range(num_iter):
            out         = out_new.copy()
            out_new[:]  = 0
            for d in directions:
                for s in shift:
                    prob_shift = np.roll(out, s, axis=d) 
                    c          = prob_shift*(1 - prob) # if prob is high, we want to diffuse less, if prob is low, we want to diffuse more
                    out_new += delta * c 

            
        log.debug(f'Finished anisotropic diffusion filtering after {num_iter} iterations')
        return out_new    

    def anisotropic_filter_with_edges(self, prob_distance, img_left, num_iter = 8):
        """
        Perona-Malik Anisotropic Diffusion.
        img: 2D grayscale float array
        kappa: Edge threshold (higher = smoother, lower = preserves sharper edges)
        """        
        num_iter    = np.maximum(4,num_iter)
        delta       = 0.24
        kappa       = 80
        s          = 1
        log.debug(f'Starting anisotropic diffusion filtering with num_iter={num_iter}, delta={delta}, kappa={kappa}')

        grad_img_S  = cv.Sobel(img_left, cv.CV_32F, 0, 1, ksize=3)
        grad_img_E  = cv.Sobel(img_left, cv.CV_32F, 1, 0, ksize=3)  
        img_edges   = np.hypot(grad_img_S, grad_img_E)  # more accurate magnitude

        #img_edges   = self.compute_edges(img_left)
        grad_img    = np.exp(-img_edges/ kappa)[:, :, np.newaxis]  # gradient magnitude as edge strength

        # c_N         = np.abs(np.roll(grad_img, -s, axis=0) - grad_img)
        # c_S         = np.abs(np.roll(grad_img,  s, axis=0) - grad_img)
        # c_E         = np.abs(np.roll(grad_img, -s, axis=1) - grad_img)
        # c_W         = np.abs(np.roll(grad_img,  s, axis=1) - grad_img)

        out         = prob_distance.copy() #.astype(np.float32)
        #directions = [0,1,2]
       
        
        for i in range(num_iter):
                    
            grad_N = np.roll(out, -s, axis=0) - out
            grad_S = np.roll(out,  s, axis=0) - out
            grad_E = np.roll(out, -s, axis=1) - out
            grad_W = np.roll(out,  s, axis=1) - out      

            # c_N = np.exp(-(grad_N / kappa)**2)
            # c_S = np.exp(-(grad_S / kappa)**2)
            # c_E = np.exp(-(grad_E / kappa)**2)
            # c_W = np.exp(-(grad_W / kappa)**2)
            
            # Update image
            #out += delta * (c_N * grad_N + c_S * grad_S + c_E * grad_E + c_W * grad_W)    
            out += delta * grad_img * (grad_N + grad_S + grad_E + grad_W)   
            out = np.clip(out, 0, 5)  # Ensure the output remains in the valid range [0, 1]          


        log.debug('Finished anisotropic diffusion filtering after %d iterations', num_iter)
        return out

    def anisotropic_filter_avergaing(self, prob_distance, img_left):
        """
        Perona-Malik Anisotropic Diffusion.
        img: 2D grayscale float array
        kappa: Edge threshold (higher = smoother, lower = preserves sharper edges)
        """   
        #from scipy.ndimage import uniform_filter     
        num_iter    = 8
        delta       = 0.25
        kappa       = 10
        log.debug(f'Starting anisotropic diffusion filtering ....')

        if prob_distance.ndim == 2:
            prob_distance = prob_distance[:, :, np.newaxis]  # expand to 3d for broadcasting        

        grad_img_S  = cv.Sobel(img_left, cv.CV_32F, 0, 1, ksize=3)
        grad_img_E  = cv.Sobel(img_left, cv.CV_32F, 1, 0, ksize=3)  
        grad_img    = np.exp(-(grad_img_S**2 + grad_img_E**2)/ kappa**2) #[:, :, np.newaxis]  # gradient magnitude as edge strength
        #c_S         = np.exp(-(grad_img_S / kappa)**2)[:, :, np.newaxis] # expand to 3d if needed
        #c_E         = np.exp(-(grad_img_E / kappa)**2)[:, :, np.newaxis]  
        

        # Apply the 3D uniform (box) filter
        kernel_size     = 11
        
        # expend to 3d for broadcasting according to the shape of prob_distance
        #if prob_distance.ndim == 3:
        #    grad_img        = grad_img[:, :, np.newaxis]  # expand to 3d for broadcasting
        #    #filtered_sum    = filtered_sum[:, :, np.newaxis]  # expand to 3d for broadcasting

        #filtered_sum    = uniform_filter(grad_img, size=kernel_size, mode='reflect') + 1e-6
        filtered_sum    = cv.boxFilter(grad_img, -1,   (kernel_size, kernel_size), normalize=False)[:, :, np.newaxis] + 1e-6

        out             = prob_distance.copy()
        for i in range(num_iter):
                    
            #out          = out * grad_img
            #filtered_3d  = uniform_filter(out, size=kernel_size, mode='reflect')
            #out          = out / filtered_sum 
            for c in range(out.shape[2]):
                out[:, :, c] = cv.boxFilter(out[:, :, c] * grad_img, -1,   (kernel_size, kernel_size), normalize=False)

            out          = out / filtered_sum 

        log.debug('Finished anisotropic diffusion filtering after %d iterations', num_iter)
        return out.squeeze()

    def anisotropic_filter(self, prob_distance, img_left):
        """
        Perona-Malik Anisotropic Diffusion.
        img: 2D grayscale float array
        kappa: Edge threshold (higher = smoother, lower = preserves sharper edges)
        """        
        num_iter    = 8
        delta       = 0.25
        kappa       = 7
        log.debug(f'Starting anisotropic diffusion ...')

        if prob_distance.ndim == 2:
            prob_distance = prob_distance[:, :, np.newaxis]  # expand to 3d for broadcasting

        out         = img_left #.astype(np.float32)
        grad_N      = np.roll(out, -1, axis=0) - out
        grad_S      = np.roll(out,  1, axis=0) - out
        grad_E      = np.roll(out, -1, axis=1) - out
        grad_W      = np.roll(out,  1, axis=1) - out

        # Exponential conduction function
        c_N         = np.exp(-(grad_N / kappa)**2)[ :, :, np.newaxis]  # expand to 3d for broadcasting
        c_S         = np.exp(-(grad_S / kappa)**2)[ :, :, np.newaxis]  # expand to 3d for broadcasting
        c_E         = np.exp(-(grad_E / kappa)**2)[ :, :, np.newaxis]  # expand to 3d for broadcasting
        c_W         = np.exp(-(grad_W / kappa)**2)[ :, :, np.newaxis]  # expand to 3d for broadcasting
              
        out       = prob_distance #.astype(np.float32)
        
        for i in range(num_iter):
            # Calculate gradients in North, South, East, West directions
            grad_N = np.roll(out, -1, axis=0) - out
            grad_S = np.roll(out,  1, axis=0) - out
            grad_E = np.roll(out, -1, axis=1) - out
            grad_W = np.roll(out,  1, axis=1) - out
          
            # Update image
            out += delta * (c_N * grad_N + c_S * grad_S + c_E * grad_E + c_W * grad_W)
            
        log.debug(f'Finished anisotropic diffusion filtering after {num_iter} iterations')
        return out.squeeze()

    def adaptive_filter_3d(self, prob_3d, kernel_size = 3, kappa = 0.1):
        "finds similar pixels in a local neighborhood and computes a weighted average of the depth values, where the weights are based on the similarity of the pixel values. This can help reduce noise and improve the quality of the depth map."

        nr, nc, nd              = prob_3d.shape
        border                  = kernel_size >> 1
        border_d                = 3
        
        # downscale depth maps 
        prob_3d_filt            = np.zeros_like(prob_3d)  # make a copy to avoid modifying the original array
        
        for r in range(border, nr - border):
            for c in range(border, nc - border):
                for d in range(border_d, nd - border_d):

                    # compute adaptive mask - only pixels that have close values are used
                    neighborhood_volume  = prob_3d[r-border:r+border+1, c-border:c+border+1, d-border_d:d+border_d+1]
                    #neighborhood_pixels   = np.abs(neighborhood_volume - prob_3d[r,c,d])
                    # the prob_3d[r,c,d] is local minima => neighborhood_pixels are positive
                    neighborhood_pixels   = neighborhood_volume - prob_3d[r,c,d]
                    #neighborhood_mask     = 1.0 - np.clip(neighborhood_pixels/kappa, 0, 1)
                    neighborhood_mask     = np.exp(-neighborhood_pixels/kappa)
                    prob_3d_filt[r,c,d]   = np.sum(neighborhood_mask * neighborhood_volume) / (np.sum(neighborhood_mask) + 1e-5)

        return prob_3d_filt

    def adaptive_filter_3d_fast(self, prob_3d, kernel_size = 3, kappa = 0.1):
        "Vectorized equivalent of adaptive_filter_3d. Accumulates weighted contributions over all kernel offsets using shifted views, avoiding the per-voxel Python loops. Borders are left as zeros to match adaptive_filter_3d."

        nr, nc, nd      = prob_3d.shape
        border          = kernel_size >> 1
        p               = prob_3d.astype(np.float32, copy=False)
        border_d        = 3

        weight_sum      = np.zeros_like(p)
        value_sum       = np.zeros_like(p)

        # iterate over all (dr, dc, dd) offsets in the kernel window and accumulate
        # using shifted volumes. For interior voxels (those at least `border` away
        # from every face) the shifted view contains the exact neighborhood values
        # the original loop visits.
        for dr in range(-border, border + 1):
            for dc in range(-border, border + 1):
                for dd in range(-border_d, border_d + 1):
                    if dr == 0 and dc == 0 and dd == 0:
                        shifted = p
                    else:
                        shifted = np.roll(p, shift=(dr, dc, dd), axis=(0, 1, 2))
                    diff        = np.abs(shifted - p)
                    w           = 1.0 - np.clip(diff / kappa, 0.0, 1.0)
                    weight_sum += w
                    value_sum  += w * shifted

        filtered            = value_sum / (weight_sum + 1e-5)

        prob_3d_filt        = np.zeros_like(prob_3d)
        if nr > 2 * border and nc > 2 * border and nd > 2 * border_d:
            prob_3d_filt[border:nr - border, border:nc - border,  border_d:nd - border_d] = filtered[border:nr - border,
                                                            border:nc - border,
                                                            border_d:nd - border_d].astype(prob_3d.dtype, copy=False)

        return prob_3d_filt

    def estimate_disparity_from_prob(self, prob_total, estim_type=1):
        "The simple way: estimate disparity from the probability volume by taking the argmax over the disparity dimension"
        N,M,D           = prob_total.shape[:3]
        if estim_type == 1:
            disparity_map = np.argmax(prob_total, axis=2)  # shape (N, M, L)
        elif estim_type == 2:            
            # "More advanced way is to create a disparity index np.arange(0,L) and compute the expected value of the disparity for each pixel, which can help reduce noise and provide a more robust estimate of the disparity. This can be done by multiplying the probability volume by the disparity index and summing over the disparity dimension, then normalizing by the sum of probabilities."
            disparity_index = np.arange(D, dtype=np.float32)
            prob_sum        = np.sum(prob_total, axis=2) + 1e-2
            disparity_map   = np.sum(prob_total * disparity_index[np.newaxis, np.newaxis, :], axis=2) /  prob_sum # shape (N, M)  
        elif estim_type == 3: 
            prob_total       = self.softmax_with_threshold(prob_total**2, dim=2, T=0.05)  # shape (N, M, D), higher is more similar
            disparity_index = np.arange(D, dtype=np.float32)
            disparity_map   = np.sum(prob_total * disparity_index[np.newaxis, np.newaxis, :], axis=2) #/  prob_sum # shape (N, M)  
        elif estim_type == 4:            
            # "More advanced way is to create a disparity index np.arange(0,L) and compute the expected value of the disparity for each pixel, which can help reduce noise and provide a more robust estimate of the disparity. This can be done by multiplying the probability volume by the disparity index and summing over the disparity dimension, then normalizing by the sum of probabilities."
            disparity_index = np.arange(D, dtype=np.float32)
            prob_total      = prob_total**2
            prob_sum        = np.sum(prob_total, axis=2) + 1e-2
            disparity_map   = np.sum(prob_total * disparity_index[np.newaxis, np.newaxis, :], axis=2) /  prob_sum
            disparity_map[prob_sum < 0.1] = 0  # replace NaN values with 0
            # shape (N, M)
        elif estim_type == 5:
            # Hard argmax with a local 3-tap parabola refinement around the peak only.
            # estim_type 2-4 take a probability-weighted average over the *whole* disparity
            # axis, which blends separate matching modes together - exactly what smears
            # disparity across a depth edge, where the distribution is bimodal (one peak for
            # the foreground, one for the background). Fitting a parabola through only the
            # peak and its two immediate neighbors gives sub-pixel precision without ever
            # mixing in a distant, unrelated mode.
            d_peak      = np.argmax(prob_total, axis=2)              # (N, M)
            border      = (d_peak == 0) | (d_peak == D - 1)
            d_peak_c    = np.clip(d_peak, 1, D - 2)

            p0 = np.take_along_axis(prob_total, (d_peak_c - 1)[:, :, np.newaxis], axis=2)[:, :, 0]
            p1 = np.take_along_axis(prob_total,  d_peak_c[:, :, np.newaxis],      axis=2)[:, :, 0]
            p2 = np.take_along_axis(prob_total, (d_peak_c + 1)[:, :, np.newaxis], axis=2)[:, :, 0]

            denom           = p0 - 2 * p1 + p2
            offset          = np.zeros_like(denom)
            valid           = np.abs(denom) > 1e-6
            offset[valid]   = 0.5 * (p0[valid] - p2[valid]) / denom[valid]
            offset          = np.clip(offset, -0.5, 0.5)  # parabola fit is only trustworthy this close to the peak

            disparity_map           = d_peak.astype(np.float32) + offset
            disparity_map[border]   = d_peak[border].astype(np.float32)  # no margin to fit a parabola at the border

        return disparity_map

    def disparity_peak_ambiguity(self, prob_volume, suppress_radius=2):
        "Best-peak / second-peak ratio along the disparity axis. A pixel whose distribution has two comparably strong peaks (e.g. straddling a depth edge, where both the foreground and background match reasonably well) shows a ratio close to 1; an unambiguous match shows a ratio close to 0."
        N, M, D     = prob_volume.shape
        best_idx    = np.argmax(prob_volume, axis=2)                       # (N, M)
        best_val    = np.max(prob_volume, axis=2)                          # (N, M)

        d_range     = np.arange(D)[np.newaxis, np.newaxis, :]              # (1, 1, D)
        near_peak   = np.abs(d_range - best_idx[:, :, np.newaxis]) <= suppress_radius
        suppressed  = np.where(near_peak, -np.inf, prob_volume)
        second_val  = np.max(suppressed, axis=2)
        second_val[~np.isfinite(second_val)] = 0.0

        ratio       = second_val / (best_val + 1e-6)                       # (N, M)
        return ratio, best_val

    #%% ------------------------------------------
    # Upscaling

    def upscale_img_array(self, low_res_array, img_ref, upscale_type = 1):
        "Upscale the disparity map to the target shape using interpolation. This can be useful when the disparity map is computed at a lower resolution and needs to be upscaled to match the original image size."
        log.debug(f'Upscaling disparity map from {low_res_array.shape} to {img_ref.shape}')
        ndims = low_res_array.ndim
        if ndims == 2:
            low_res_array = low_res_array[:, :, np.newaxis]  # expand to 3d for broadcasting
        nr_lr, nc_lr, nd_lr = low_res_array.shape
        nr_hr, nc_hr        = img_ref.shape[:2]
        scale_factor        = int(nc_hr/nr_lr)
        nd_hr               = nd_lr * scale_factor
        high_res_array      = np.zeros((nr_hr, nc_hr, nd_hr), dtype=low_res_array.dtype)

        if upscale_type == 1:
            for d in range(nd_lr):
                high_res_array[:,:,d] = cv.resize(low_res_array[:,:,d], (nc_hr, nr_hr), interpolation=cv.INTER_LINEAR)
        elif upscale_type == 2:
            for d in range(nd_lr):
                high_res_array[:,:,d] = cv.resize(low_res_array[:,:,d], (nc_hr, nr_hr), interpolation=cv.INTER_CUBIC)
        elif upscale_type == 11:
            for d in range(nd_lr):
                high_res_array[:,:,d] = zoom(low_res_array[:,:,d], zoom=(scale_factor, scale_factor), order=1)
        elif upscale_type == 12:
                high_res_array = zoom(low_res_array, zoom=(scale_factor, scale_factor, scale_factor), order=1)                
        elif upscale_type == 22:
            guide_u8        = cv.normalize(img_ref, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
            for d in range(nd_lr):
                baseline        = low_res_array[:,:,d]
                src_u8          = cv.normalize(baseline, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
                # 4) Guided filter
                context_guided  = cv.ximgproc.guidedFilter(
                    guide=guide_u8,
                    src=src_u8,
                    radius=4,
                    eps=25.0
                    )
                high_res_array[:,:,d] = context_guided.astype(high_res_array.dtype)

        elif upscale_type == 23:
            guide_u8        = cv.normalize(img_ref, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
            for d in range(nd_lr):
                baseline        = low_res_array[:,:,d]
                src_u8          = cv.normalize(baseline, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
                context_fgs = cv.ximgproc.fastGlobalSmootherFilter(
                guide=guide_u8,
                src=src_u8,
                lambda_=12.0,
                sigma_color=8.0
                )
                high_res_array[:,:,d] = context_fgs.astype(high_res_array.dtype)

        if ndims == 2:
            high_res_array = high_res_array.squeeze()
        return high_res_array

    #%% ------------------------------------------

    def gabor_image_disparity_multiscale(self, img_left, img_right, debug_row=None):
        "compute row-wise left/right gabor channel inner products and show MxM matrix"
        row_index               = debug_row if debug_row is not None else 400
        debug                   = debug_row is not None

        level_num               = 3
        max_disparity           = 128
        
        kernel_size             = 9
        row_num,col_num        = img_left.shape[:2]

        img_left, img_right = img_left.astype(np.float32), img_right.astype(np.float32)

        # Gabor bank parameters.
        rot_left, rot_right     = np.pi/16*0, np.pi/10*0  # rotate left image Gabors slightly counterclockwise and right image Gabors slightly clockwise to better match floor pattern    
        #rot_left, rot_right   = 0, 0 
        thetas                  = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        lambdas                 = [8.0, 16.0, 32.0];  psis            = [0.0, np.pi / 2];  ksize           = 13; sigma           = 4.0;      gamma = 0.5
        thetas_left             = [t + rot_left for t in thetas]  # rotate left image Gabors slightly counterclockwise
        thetas_right            = [t - rot_right for t in thetas]
        bank_left, p_left        = self.gabor_bank_init(ksize=ksize, sigma=sigma, lambdas=lambdas, gamma=gamma, psis=psis, thetas=thetas_left)
        bank_right, p_right      = self.gabor_bank_init(ksize=ksize, sigma=sigma, lambdas=lambdas, gamma=gamma, psis=psis, thetas=thetas_right)         


        prob_total              = np.zeros((row_num,col_num,level_num,max_disparity), dtype=np.float32)
        conf_total              = np.zeros((row_num,col_num,level_num), dtype=np.float32) # left-right confidence
        info_total              = np.zeros((row_num,col_num,level_num), dtype=np.float32) # left spatial information content
        debug_levels            = []
        for k in range(level_num):

            # Build NxMxC responses for both images.
            gabor_left               = self.gabor_bank_filter(img_left, bank=bank_left)
            gabor_right              = self.gabor_bank_filter(img_right, bank=bank_right) 

            # normalize the responses across channels for each pixel to have zero mean but not variance. 
            # This can help improve the correlation by making it more about the pattern of responses across channels rather than their absolute strength.  
            gabor_left_norm          = np.sqrt(np.linalg.norm(gabor_left, axis=2, keepdims=True)) + 1e-6
            gabor_left               = gabor_left / gabor_left_norm
            gabor_right_norm         = np.sqrt(np.linalg.norm(gabor_right, axis=2, keepdims=True)) + 1e-6
            gabor_right              = gabor_right / gabor_right_norm

            # information spatial content for the left image, which can be used to weight the correlation scores. This can help identify pixels that are more likely to provide reliable matches, so we can use this information to weight the correlation scores.
            info_left                = self.gabor_decomposition_variability(gabor_left, kernel_size=kernel_size)  # shape (N, M)

            # 3d volume distance between left and right gabor responses across channels for each pixel and disparity. This can be used as a cost volume for stereo matching, where lower values indicate more similar responses and thus more likely matches.
            distance_left            = self.gabor_dispartity(gabor_left, gabor_right, max_disparity=max_disparity) # shape (N, M, D)

            # convert to probability using soft max over the disparity dimension, which can help normalize the scores and make them more interpretable as probabilities. The temperature parameter T can be tuned to control the sharpness of the distribution, with lower values leading to a more peaked distribution and higher values leading to a softer distribution.
            prob_distance           = self.softmax_with_threshold(-distance_left, dim=2, T=0.05)  # shape (N, M, D), higher is more similar

            # confidence of the best match for each pixel, which can be used to weight the depth estimation. This can help improve the depth estimation by giving more weight to pixels that have a more confident best match, and less weight to pixels that have a more ambiguous match. The confidence can be computed as the maximum probability across disparities for each pixel, or as a measure of how much better the best match is compared to the second-best match.
            conf_left                = np.max(prob_distance, axis=2)  # shape (N, M)
            conf_current             = cv.resize(conf_left, (col_num, row_num))
            conf_total[:,:,k]        = conf_current            

            # for multi level processing interpolate the information content to the original image size and accumulate it in a 3d volume, which can be used to weight the probability volume. This can help improve the depth estimation by giving more weight to pixels that are more likely to provide reliable matches.  
            info_current             = cv.resize(info_left, (col_num, row_num))
            info_total[:,:,k]        = info_current

            # upscale 3D volume to the original image size and accumulate it in a 4D volume, which can be used to compute a final depth estimation by taking a weighted average over the disparity dimension. This can help improve the depth estimation by giving more weight to disparities that are more likely to be correct based on the accumulated information content across levels.  
            #prob_current             = cv.resize(prob_distance, (col_num, row_num))  # shape (N, M, D)

            # perorm anisotropic filtering based on left image information
            prob_distance             = self.anisotropic_filter(prob_distance, img_left)  # shape (N, M, D)


            # Zooming a 3D volume by a factor of 2 in all dimensions
            # Spline order 1 = Trilinear, Order 3 = Tricubic
            prob_current            = zoom(prob_distance, zoom=2**k, order=1)
            prob_total[:,:,k,:]     = prob_current
            #prob_current            = cv.resize(prob_distance, (col_num, row_num))  # shape (N, M, D)
            # interpolate prob_current from (N, M, D) to (N, M, D) by resizing each disparity slice and then stacking them back together
            # prob_current            = np.zeros((row_num, col_num, max_disparity), dtype=np.float32)
            # for d in range(max_disparity):
            #     prob_current[:,:,d] = cv.resize(prob_distance[:,:,d], (col_num, row_num))
            # prob_total[:,:,k,:]     = prob_current
            #prob_total[:,:,k,:] = grid_interpolation(prob_distance, factor=2**k)

            if debug:
                debug_levels.append(
                    {
                        "gabor_left": gabor_left,
                        "gabor_right": gabor_right,
                        "distance_left": distance_left,
                        "info_left": info_left,
                        "prob_distance": prob_distance,
                        "img_left": img_left.copy(),
                        "img_right": img_right.copy(),
                    }
                )

            # downscale
            img_left                = cv.pyrDown(img_left)
            img_right               = cv.pyrDown(img_right)
            max_disparity           = max_disparity // 2

                
        # normalize to probability by multiplying the spatial information and then normalizeing accross lev el dimensions
        # prob_total                = prob_total * info_total[:,:,:,np.newaxis]  # weight by information content, shape (N, M, D)
        # info_sum                  = np.sum(info_total, axis=2) + 1e-6  # shape (N, M)
        # prob_total                = np.sum(prob_total, axis=2) / info_sum[:,:,np.newaxis]  # shape (N, M)
        prob_total_final                  = np.sum(prob_total,axis=2)   # weight by confidence, shape (N, M, D)

        # imshow the argmax value of the final probability volume as a debug view of the estimated disparity map, which can help visualize the results and identify any issues with the estimation. The argmax value represents the disparity with the highest probability for each pixel, which can be interpreted as the most likely depth estimate based on the computed probabilities. By visualizing this disparity map, we can gain insights into how well the algorithm is performing and where it may be struggling to make accurate estimates.
        disp_index                  = np.argmax(prob_total_final, axis=2)
        plt.figure(figsize=(12, 6)) 
        plt.imshow(disp_index, cmap='viridis')
        plt.colorbar(label='Disparity Index')
        plt.title('Estimated Disparity Map (argmax of probability volume)')
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.show(block=False)

        if debug and row_index is not None:
            self.debug_gabor_image_disparity_multiscale(debug_levels, prob_total, row_index=row_index)

        return prob_total
    
    def gabor_image_disparity_down_up(self, img_left, img_right, debug_row=None):
        "compute row-wise left/right gabor channel inner products and show MxM matrix"
        row_index               = debug_row if debug_row is not None else 400
        debug                   = debug_row is not None

        level_num               = 3
        max_disparity           = 128
        
        kernel_size             = 9
        row_num,col_num        = img_left.shape[:2]
        T_weights               = [0.05, 0.1, 0.2]  # example weights for each level, should sum to 1

        img_left, img_right     = img_left.astype(np.float32), img_right.astype(np.float32)
        img_left_ref            = img_left.copy()

        # Gabor bank parameters.
        rot_left, rot_right     = np.pi/16*0, np.pi/10*0  # rotate left image Gabors slightly counterclockwise and right image Gabors slightly clockwise to better match floor pattern    
        #rot_left, rot_right   = 0, 0 
        thetas                  = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        lambdas                 = [8.0, 16.0, 32.0];  psis = [0.0, np.pi / 2];  ksize = 13; sigma  = 4.0;  gamma = 0.5
        thetas_left             = [t + rot_left for t in thetas]  # rotate left image Gabors slightly counterclockwise
        thetas_right            = [t - rot_right for t in thetas]
        bank_left, p_left        = self.gabor_bank_init(ksize=ksize, sigma=sigma, lambdas=lambdas, gamma=gamma, psis=psis, thetas=thetas_left)
        bank_right, p_right      = self.gabor_bank_init(ksize=ksize, sigma=sigma, lambdas=lambdas, gamma=gamma, psis=psis, thetas=thetas_right)         


        prob_total              = np.zeros((row_num,col_num,level_num,max_disparity), dtype=np.float32)
        conf_total              = np.zeros((row_num,col_num,level_num), dtype=np.float32) # left-right confidence
        info_total              = np.zeros((row_num,col_num,level_num), dtype=np.float32) # left spatial information content
        debug_levels            = []
        info_left_all, prob_left_all, prob_context_all = [], [], []
        for k in range(level_num):

            # Build NxMxC responses for both images.
            gabor_left               = self.gabor_bank_filter(img_left, bank=bank_left)
            gabor_right              = self.gabor_bank_filter(img_right, bank=bank_right) 

            # normalize the responses across channels for each pixel to have zero mean but not variance. 
            gabor_left               = self.gabor_normalize_responses(gabor_left)
            gabor_right              = self.gabor_normalize_responses(gabor_right)

            # information spatial content for the left image, which can be used to weight the correlation scores. This can help identify pixels that are more likely to provide reliable matches, so we can use this information to weight the correlation scores.
            info_left                = self.gabor_decomposition_variability(gabor_left, kernel_size=kernel_size)  # shape (N, M)

            # 3d volume distance between left and right gabor responses across channels for each pixel and disparity. This can be used as a cost volume for stereo matching, where lower values indicate more similar responses and thus more likely matches.
            distance_left            = self.gabor_dispartity(gabor_left, gabor_right, max_disparity=max_disparity) # shape (N, M, D)

            # apply contextual filtering to the distance volume using the left image as guidance, which can help improve the matching by taking into account the local structure of the left image. This can be done using a guided filter or a bilateral filter, which can help preserve edges and reduce noise in the distance volume.
            #distance_left            = self.anisotropic_filter(distance_left, img_left)

            # convert to probability using soft max over the disparity dimension, which can help normalize the scores and make them more interpretable as probabilities. The temperature parameter T can be tuned to control the sharpness of the distribution, with lower values leading to a more peaked distribution and higher values leading to a softer distribution.
            prob_left               = self.softmax_with_threshold(-distance_left, dim=2, T=0.05) #T_weights[k])  # shape (N, M, D), higher is more similar

            # perorm anisotropic filtering based on left image information
            conf_left                = np.max(prob_left, axis=2)  # shape (N, M)
            #prob_context             = prob_left * conf_left[:,:,np.newaxis] #
            #prob_context            = self.anisotropic_filter(prob_left, img_left)
            prob_context            = self.anisotropic_filter_avergaing(prob_left, 64*conf_left.astype(np.float32))  # shape (N, M, D)
            #prob_context            = self.anisotropic_filter_3d(prob_left)  # shape (N, M, D), higher is more similar
            #prob_context            = self.anisotropic_filter_prob(prob_left)  # shape (N, M, D), higher is more similar
            #distance_left_2, tmp_P  = self.kalman_neighborhood_fusion(distance_left, prob_left, kernel_size=9)  # shape (N, M, D)
            #prob_context             = self.softmax_with_threshold(-distance_left_2, dim=2, T=0.05)  # shape (N, M, D)

            if debug:
                debug_levels.append({
                    "img_left": img_left,  "img_right": img_right, "gabor_left": gabor_left,  "gabor_right": gabor_right,
                    "distance_left": distance_left,  "info_left": info_left,  "prob_left": prob_left, "prob_context": prob_context}
                    )                   

            # save 
            info_left_all.append(info_left)
            prob_left_all.append(prob_left)
            prob_context_all.append(prob_context)

            # downscale
            img_left                = zoom(img_left, zoom=0.5, order=1)
            img_right               = zoom(img_right, zoom=0.5, order=1)
            max_disparity           = max_disparity // 2

        for k in reversed(range(level_num)):

            # recover the saved information and probabilities for this level
            info_left                = info_left_all[k]
            prob_left                = prob_left_all[k]
            prob_context             = prob_context_all[k]

            # confidence of the best match for each pixel, which can be used to weight the depth estimation. This can help improve the depth estimation by giving more weight to pixels that have a more confident best match, and less weight to pixels that have a more ambiguous match. The confidence can be computed as the maximum probability across disparities for each pixel, or as a measure of how much better the best match is compared to the second-best match.
            conf_left                = np.max(prob_left, axis=2)  # shape (N, M)
            conf_current             = zoom(conf_left, zoom=2**k, order=1)
            conf_total[:,:,k]        = conf_current            

            # for multi level processing interpolate the information content to the original image size and accumulate it in a 3d volume, which can be used to weight the probability volume. This can help improve the depth estimation by giving more weight to pixels that are more likely to provide reliable matches.  
            info_current             = zoom(info_left, zoom=2**k, order=1)
            info_total[:,:,k]        = info_current      

            # Zooming a 3D volume by a factor of 2 in all dimensions
            # Spline order 1 = Trilinear, Order 3 = Tricubic
            #prob_current            = zoom(prob_left, zoom=2**k, order=1)
            prob_current            = zoom(prob_context, zoom=2**k, order=1)
            prob_total[:,:,k,:]     = prob_current   
            #prob_total[:,:,k,:]     = np.roll(prob_current, shift=-k, axis=[2])  

                        # 5. Probability current level vs accumulated final
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(conf_current, cmap='viridis', aspect='auto')
            plt.colorbar(label='Confidence current level')
            plt.title(f'Confidence current, Level {k}')
            plt.subplot(1, 2, 2)
            plt.imshow(info_current, cmap='viridis', aspect='auto')
            plt.colorbar(label='Information content')
            plt.title(f'Information content, Level {k}')
            plt.tight_layout()             

                
        # normalize to probability by multiplying the spatial information and then normalizeing accross level dimensions
        # prob_total_final            = prob_total * info_total[:,:,:,np.newaxis]  # weight by information content, shape (N, M, D)
        # info_sum                    = np.sum(info_total, axis=2) + 1e-6  # shape (N, M)
        # prob_total_final            = np.sum(prob_total_final, axis=2) / info_sum[:,:,np.newaxis]  # shape (N, M)
        # prob_total_final            = prob_total * conf_total[:,:,:,np.newaxis]  # weight by confidence, shape (N, M, D)
        # conf_sum                    = np.sum(conf_total, axis=2) + 1e-6  # shape (N, M)
        # prob_total_final            = np.sum(prob_total_final, axis=2) / conf_sum[:,:,np.newaxis]  # shape (N, M)        
        #prob_total_final             = np.sum(prob_total,axis=2)   # weight by confidence, shape (N, M, D)
        prob_total_final             = prob_total[:,:,0,:]*4/7 + prob_total[:,:,1,:]*2/7 + prob_total[:,:,2,:]*1/7  # weight by confidence, shape (N, M, D)
        #prob_total_final              = self.softmax_with_threshold(prob_total,dim=2, T=0.05)

        #prob_total_final             = anisotropic_diffusion(prob_total_final)  # shape (N, M, D)

        # imshow the argmax value of the final probability volume as a debug view of the estimated disparity map, which can help visualize the results and identify any issues with the estimation. The argmax value represents the disparity with the highest probability for each pixel, which can be interpreted as the most likely depth estimate based on the computed probabilities. By visualizing this disparity map, we can gain insights into how well the algorithm is performing and where it may be struggling to make accurate estimates.
        disp_index                   = self.estimate_disparity_from_prob(prob_total_final) # ensure values are within the valid range
        #disp_index[conf_sum < 0.3]   = 0  # mask out low confidence areas
        #disp_index                   = self.anisotropic_filter(disp_index, img_left_ref) 

        plt.figure(figsize=(12, 6)) 
        plt.imshow(disp_index, cmap='viridis')
        plt.colorbar(label='Disparity Index')
        plt.title('Estimated Disparity Map (argmax of probability volume)')
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.show(block=False)

        if debug and row_index is not None:
            self.debug_gabor_image_disparity_multiscale(debug_levels, prob_total, row_index=row_index)

        plt.show()
        return prob_total    

    def gabor_image_disparity_down_up_on_volume(self, img_left, img_right, debug_row=None):
        "compute row-wise left/right gabor channel inner products and show MxM matrix"
        row_index               = debug_row if debug_row is not None else 400
        debug                   = debug_row is not None

        level_num               = 3
        max_disparity           = 128
        
        kernel_size             = 9
        row_num,col_num        = img_left.shape[:2]
        T_weights               = [0.05, 0.1, 0.2]  # example weights for each level, should sum to 1

        img_left, img_right     = img_left.astype(np.float32), img_right.astype(np.float32)
        img_left_ref            = img_left.copy()

        # Gabor bank parameters.
        rot_left, rot_right     = np.pi/16*0, np.pi/10*0  # rotate left image Gabors slightly counterclockwise and right image Gabors slightly clockwise to better match floor pattern    
        #rot_left, rot_right   = 0, 0 
        thetas                  = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        lambdas                 = [8.0, 16.0, 32.0];  psis = [0.0, np.pi / 2];  ksize = 13; sigma  = 4.0;  gamma = 0.5
        thetas_left             = [t + rot_left for t in thetas]  # rotate left image Gabors slightly counterclockwise
        thetas_right            = [t - rot_right for t in thetas]
        bank_left, p_left        = self.gabor_bank_init(ksize=ksize, sigma=sigma, lambdas=lambdas, gamma=gamma, psis=psis, thetas=thetas_left)
        bank_right, p_right      = self.gabor_bank_init(ksize=ksize, sigma=sigma, lambdas=lambdas, gamma=gamma, psis=psis, thetas=thetas_right)         

        max_disparity_4         = max_disparity // 4
        prob_total              = np.zeros((row_num // 4, col_num // 4, level_num, max_disparity_4), dtype=np.float32)
        
        # processing one by one
        gabor_left               = self.gabor_bank_filter(img_left, bank=bank_left)
        gabor_right              = self.gabor_bank_filter(img_right, bank=bank_right) 

        # download the gabor resolution
        gabor_left               = zoom(gabor_left,  zoom=(0.25,0.25,1), order=1)
        gabor_right              = zoom(gabor_right, zoom=(0.25,0.25,1), order=1)

        # normalize the responses across channels for each pixel to have zero mean but not variance. 
        gabor_left               = self.gabor_normalize_responses(gabor_left)
        gabor_right              = self.gabor_normalize_responses(gabor_right)        

        # 3d volume distance between left and right gabor responses across channels for each pixel and disparity. This can be used as a cost volume for stereo matching, where lower values indicate more similar responses and thus more likely matches.
        distance_left            = self.gabor_dispartity(gabor_left, gabor_right, max_disparity=max_disparity_4) # shape (N, M, D)   

        # store the distance volume for the first level
        prob_total[:,:,0,:]      = distance_left

        # downscale images
        img_left                = zoom(img_left,  zoom=0.5, order=1)
        img_right               = zoom(img_right, zoom=0.5, order=1) 

        gabor_left               = self.gabor_bank_filter(img_left, bank=bank_left)
        gabor_right              = self.gabor_bank_filter(img_right, bank=bank_right)

        # download the gabor resolution
        gabor_left               = zoom(gabor_left,  zoom=(0.5, 0.5, 1), order=1)
        gabor_right              = zoom(gabor_right, zoom=(0.5, 0.5, 1), order=1)  

        # normalize the responses across channels for each pixel to have zero mean but not variance. 
        gabor_left               = self.gabor_normalize_responses(gabor_left)
        gabor_right              = self.gabor_normalize_responses(gabor_right)        

        # 3d volume distance between left and right gabor responses across channels for each pixel and disparity. This can be used as a cost volume for stereo matching, where lower values indicate more similar responses and thus more likely matches.
        distance_left            = self.gabor_dispartity(gabor_left, gabor_right, max_disparity=max_disparity_4) # shape (N, M, D)   

        # store the distance volume for the first level
        prob_total[:,:,1,:]      = distance_left     

        # downscale images
        img_left                = zoom(img_left,  zoom=0.5, order=1)
        img_right               = zoom(img_right, zoom=0.5, order=1) 

        gabor_left               = self.gabor_bank_filter(img_left, bank=bank_left)
        gabor_right              = self.gabor_bank_filter(img_right, bank=bank_right)        

        # normalize the responses across channels for each pixel to have zero mean but not variance. 
        gabor_left               = self.gabor_normalize_responses(gabor_left)
        gabor_right              = self.gabor_normalize_responses(gabor_right)              

        # 3d volume distance between left and right gabor responses across channels for each pixel and disparity. This can be used as a cost volume for stereo matching, where lower values indicate more similar responses and thus more likely matches.
        distance_left            = self.gabor_dispartity(gabor_left, gabor_right, max_disparity=max_disparity_4) # shape (N, M, D)   

        # store the distance volume for the first level
        prob_total[:,:,2,:]      = distance_left

        debug_row_4              = debug_row // 4
        img_list                 = [prob_total[debug_row_4,:,m,:].squeeze().T for m in range(3)]
        ttl_list                 = [f'Level {m} Probability Volume (row {debug_row_4})' for m in range(3)]
        self.show_subset(img_list, ttl_list, col_num=1)

        # convert to probability using soft max over the disparity dimension, which can help normalize the scores and make them more interpretable as probabilities. The temperature parameter T can be tuned to control the sharpness of the distribution, with lower values leading to a more peaked distribution and higher values leading to a softer distribution.
        prob_total               = self.softmax_with_threshold(-prob_total, dim=3, T=0.2) #T_weights[k])  # shape (N, M, D), higher is more similar
        
        img_list                 = [prob_total[debug_row_4,:,m,:].squeeze().T for m in range(3)]
        ttl_list                 = [f'Level {m} Probability Volume (row {debug_row_4})' for m in range(3)]
        self.show_subset(img_list, ttl_list, col_num=1)

        img_list                 = [prob_total[debug_row_4+1,:,m,:].squeeze().T for m in range(3)]
        ttl_list                 = [f'Level {m} Probability Volume (row {debug_row_4+1})' for m in range(3)]
        self.show_subset(img_list, ttl_list, col_num=1)

        img_list                 = [prob_total[debug_row_4-1,:,m,:].squeeze().T for m in range(3)]
        ttl_list                 = [f'Level {m} Probability Volume (row {debug_row_4-1})' for m in range(3)]
        self.show_subset(img_list, ttl_list, col_num=1)   

        # collaps
        prob_total_final         = np.sum(prob_total, axis=2).squeeze()  # weight by confidence, shape (N, M, D)     

        # do some filtering on the final probability volume to smooth out noise and improve the disparity estimation. This can be done using a Gaussian filter, a median filter, or an anisotropic diffusion filter, depending on the characteristics of the data and the desired level of smoothing. The goal is to reduce spurious peaks in the probability distribution while preserving important features such as edges and texture.
        prob_total_filter         = self.adaptive_filter_3d(prob_total_final, kernel_size = 9, kappa = 0.1)  # shape (N, M, D)

        img_list                 = [prob_total_final[debug_row_4,:,:].squeeze().T , prob_total_filter[debug_row_4,:,:].squeeze().T]
        ttl_list                 = [f'Final Probability Volume (row {debug_row_4})', f'Filtered Final Probability Volume (row {debug_row_4})']
        self.show_subset(img_list, ttl_list, col_num=1)        

        disp_index               = self.estimate_disparity_from_prob(prob_total_filter) # ensure values are within the valid range
                

        plt.figure(figsize=(12, 6)) 
        plt.imshow(disp_index, cmap='viridis')
        plt.colorbar(label='Disparity Index')
        plt.title('Estimated Disparity Map (argmax of probability volume)')
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.show(block=False)

        # if debug and row_index is not None:
        #     self.debug_gabor_image_disparity_multiscale(debug_levels, prob_total, row_index=row_index)

        plt.show()
        return prob_total    

    def gabor_image_disparity_down_up_full_volume(self, img_left, img_right, debug_row=None):
        "compute row-wise left/right gabor channel inner products and show MxM matrix"
        row_index               = debug_row if debug_row is not None else 400
        debug                   = debug_row is not None

        level_num               = 3
        max_disparity           = 128
        debug_on                = False
        
        kernel_size             = 9
        row_num,col_num        = img_left.shape[:2]
        T_weights               = [0.05, 0.1, 0.2]  # example weights for each level, should sum to 1

        img_left, img_right     = img_left.astype(np.float32), img_right.astype(np.float32)
        img_left_ref            = img_left.copy()

        # Gabor bank parameters.
        rot_left, rot_right     = np.pi/16*0, np.pi/10*0  # rotate left image Gabors slightly counterclockwise and right image Gabors slightly clockwise to better match floor pattern    
        #rot_left, rot_right   = 0, 0 
        thetas                  = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        lambdas                 = [8.0, 16.0, 32.0];  psis = [0.0, np.pi / 2];  ksize = 9; sigma  = 4.0;  gamma = 0.5
        thetas_left             = [t + rot_left for t in thetas]  # rotate left image Gabors slightly counterclockwise
        thetas_right            = [t - rot_right for t in thetas]
        bank_left, p_left        = self.gabor_bank_init(ksize=ksize, sigma=sigma, lambdas=lambdas, gamma=gamma, psis=psis, thetas=thetas_left)
        bank_right, p_right      = self.gabor_bank_init(ksize=ksize, sigma=sigma, lambdas=lambdas, gamma=gamma, psis=psis, thetas=thetas_right)         

        prob_total              = np.zeros((row_num, col_num ,level_num, max_disparity), dtype=np.float32)
        
        # processing one by one
        gabor_left               = self.gabor_bank_filter(img_left, bank=bank_left)
        gabor_right              = self.gabor_bank_filter(img_right, bank=bank_right) 

        # normalize the responses across channels for each pixel to have zero mean but not variance. 
        gabor_left               = self.gabor_normalize_responses(gabor_left)
        gabor_right              = self.gabor_normalize_responses(gabor_right)        

        # 3d volume distance between left and right gabor responses across channels for each pixel and disparity. This can be used as a cost volume for stereo matching, where lower values indicate more similar responses and thus more likely matches.
        distance_left            = self.gabor_dispartity(gabor_left, gabor_right, max_disparity=max_disparity) # shape (N, M, D)   

        # store the distance volume for the first level
        #distance_left_filtered   = self.joint_bilateral_upsampling(distance_left, img_left_ref, spatial_sigma=3.0, range_sigma=7.1, radius=4)
        #distance_left             = self.joint_bilateral_filtering(img_left, distance_left, spatial_sigma=3.0, range_sigma=5.1, radius=2, iter_num=3)
        prob_total[:,:,0,:]       = distance_left

        #distance_left_filtered   = self.anisotropic_filter_3d(distance_left, kappa = 0.1)  # shape (N, M, D)

        # downscale images
        img_left                = zoom(img_left,  zoom=0.5, order=1)
        img_right               = zoom(img_right, zoom=0.5, order=1) 

        gabor_left               = self.gabor_bank_filter(img_left, bank=bank_left)
        gabor_right              = self.gabor_bank_filter(img_right, bank=bank_right)

        # normalize the responses across channels for each pixel to have zero mean but not variance. 
        gabor_left               = self.gabor_normalize_responses(gabor_left)
        gabor_right              = self.gabor_normalize_responses(gabor_right)        

        # 3d volume distance between left and right gabor responses across channels for each pixel and disparity. This can be used as a cost volume for stereo matching, where lower values indicate more similar responses and thus more likely matches.
        distance_left            = self.gabor_dispartity(gabor_left, gabor_right, max_disparity=max_disparity//2) # shape (N, M, D)   

        # store the distance volume for the first level
        #distance_left_filtered  = self.joint_bilateral_upsampling(distance_left, img_left_ref, spatial_sigma=3.0, range_sigma=7.1, radius=4)
        #distance_left             = self.joint_bilateral_filtering(img_left, distance_left, spatial_sigma=3.0, range_sigma=5.1, radius=2, iter_num=3)
        #prob_total[:,:,1,:]      = distance_left
        prob_total[:,:,1,:]      = zoom(distance_left,  zoom=(2, 2, 2), order=1) 
        #prob_total[:,:,1,:]      = np.roll(zoom(distance_left,  zoom=(2, 2, 2), order=1), shift=[ksize//2, -ksize//2], axis=[0,1])
        #prob_total[:,:,1,:]      = zoom(distance_left_filtered,  zoom=(1, 1, 2), order=1)    

        # downscale images
        img_left                = zoom(img_left,  zoom=0.5, order=1)
        img_right               = zoom(img_right, zoom=0.5, order=1) 

        gabor_left               = self.gabor_bank_filter(img_left, bank=bank_left)
        gabor_right              = self.gabor_bank_filter(img_right, bank=bank_right)        

        # normalize the responses across channels for each pixel to have zero mean but not variance. 
        gabor_left               = self.gabor_normalize_responses(gabor_left)
        gabor_right              = self.gabor_normalize_responses(gabor_right) 
              
        # 3d volume distance between left and right gabor responses across channels for each pixel and disparity. This can be used as a cost volume for stereo matching, where lower values indicate more similar responses and thus more likely matches.
        distance_left            = self.gabor_dispartity(gabor_left, gabor_right, max_disparity=max_disparity//4) # shape (N, M, D)   

        # upsample
        #distance_left_filtered  = self.joint_bilateral_upsampling(distance_left, img_left_ref, spatial_sigma=3.0, range_sigma=7.1, radius=4)

        # store the distance volume for the first level
        #distance_left            = self.joint_bilateral_filtering(img_left, distance_left, spatial_sigma=3.0, range_sigma=5.1, radius=2, iter_num=3)
        prob_total[:,:,2,:]      = zoom(distance_left,  zoom=(4, 4, 4), order=1)    
        #prob_total[:,:,2,:]      = np.roll(zoom(distance_left,  zoom=(4, 4, 4), order=1), shift=[ksize,-ksize*3//2], axis=[0,1]) 
        #prob_total[:,:,2,:]      = zoom(distance_left_filtered,  zoom=(1, 1, 4), order=1)      

        if debug_on:
            img_list                 = [prob_total[debug_row,:,m,:].squeeze().T for m in range(3)]
            ttl_list                 = [f'Level {m} Probability Volume (row {debug_row})' for m in range(3)]
            self.show_subset(img_list, ttl_list, col_num=1)

        # convert to probability using soft max over the disparity dimension, which can help normalize the scores and make them more interpretable as probabilities. The temperature parameter T can be tuned to control the sharpness of the distribution, with lower values leading to a more peaked distribution and higher values leading to a softer distribution.
        prob_total               = self.softmax_with_threshold(-prob_total, dim=3, T=0.05) #T_weights[k])  # shape (N, M, D), higher is more similar

        if debug_on:        
            #img_list                 = [prob_total[debug_row,:,m,:].squeeze().T for m in range(3)]
            img_list                 = [prob_total[:,debug_row,m,:].squeeze().T for m in range(3)]
            ttl_list                 = [f'Level {m} Probability Volume (row {debug_row})' for m in range(3)]
            self.show_subset(img_list, ttl_list, col_num=1)

            img_list                 = [prob_total[debug_row+15,:,m,:].squeeze().T for m in range(3)]
            ttl_list                 = [f'Level {m} Probability Volume (row {debug_row+15})' for m in range(3)]
            self.show_subset(img_list, ttl_list, col_num=1)

            img_list                 = [prob_total[debug_row-15,:,m,:].squeeze().T for m in range(3)]
            ttl_list                 = [f'Level {m} Probability Volume (row {debug_row-15})' for m in range(3)]
            self.show_subset(img_list, ttl_list, col_num=1)   

        # collaps
        prob_total_final         = np.sum(prob_total, axis=2).squeeze()  # weight by confidence, shape (N, M, D)   
        #prob_total_final         = np.sum(prob_total / (np.sum(prob_total, axis=2, keepdims=True) + 0.01), axis=2).squeeze()
        #prob_total_filter        = self.probability_bilateral_filtering(img_left_ref, prob_total_final, spatial_sigma=3.0, range_sigma=5.1, radius=3, iter_num=1) 
        prob_total_filter        = self.joint_bilateral_filtering(img_left_ref, prob_total_final, spatial_sigma=3.0, range_sigma=5.1, radius=2, iter_num=4) 
        

        # do some filtering on the final probability volume to smooth out noise and improve the disparity estimation. This can be done using a Gaussian filter, a median filter, or an anisotropic diffusion filter, depending on the characteristics of the data and the desired level of smoothing. The goal is to reduce spurious peaks in the probability distribution while preserving important features such as edges and texture.
        #prob_total_filter         = self.adaptive_filter_3d_fast(prob_total_final, kernel_size = 15, kappa = 0.1)  # shape (N, M, D)
        #prob_total_filter         = self.anisotropic_filter_avergaing(prob_total_final, img_left_ref)  # shape (N, M, D)
        #prob_total_filter        = self.joint_bilateral_upsampling(prob_total_final, img_left_ref, spatial_sigma=3.0, range_sigma=7.1, radius=4)
        # prob_total_filter        = prob_total_final.copy()
        # for i in range(8):
        #     prob_total_filter    = self.joint_bilateral_filtering(img_left_ref, prob_total_filter, spatial_sigma=3.0, range_sigma=5.1, radius=3)
        #prob_total_filter        = self.joint_bilateral_filtering(img_left_ref, prob_total_final, spatial_sigma=3.0, range_sigma=5.1, radius=3, iter_num=8) 

        if debug_on:
            img_list                 = [prob_total_final[debug_row,:,:].squeeze().T , prob_total_filter[debug_row,:,:].squeeze().T]
            ttl_list                 = [f'Final Probability Volume (row {debug_row})', f'Filtered Final Probability Volume (row {debug_row})']
            self.show_subset(img_list, ttl_list, col_num=1)        

        disp_index               = self.estimate_disparity_from_prob(prob_total_filter, estim_type = 4) # ensure values are within the valid range
        disp_confidence          = np.max(prob_total_filter, axis=2)  # shape (N, M)
        disp_index[disp_confidence < 0.1]   = 0  # mask out low confidence areas

        # disp_confidence_filter    = disp_confidence.copy()
        # for i in range(8):
        #     disp_confidence_filter    = self.joint_bilateral_filtering(img_left_ref, disp_confidence_filter, spatial_sigma=3.0, range_sigma=5.1, radius=2)
        disp_confidence_filter    = self.joint_bilateral_filtering(img_left_ref, disp_confidence, spatial_sigma=3.0, range_sigma=5.1, radius=2, iter_num=4)


        if debug_on:
            img_list                 = [img_left_ref, disp_index , disp_confidence, disp_confidence_filter]
            ttl_list                 = ['Left Image', 'Estimated Disparity Map', 'Estimated Disparity Confidence Map','Filtered Confidence Map']
            self.show_subset(img_list, ttl_list, col_num=2) 

        # if debug and row_index is not None:
        #     self.debug_gabor_image_disparity_multiscale(debug_levels, prob_total, row_index=row_index)

        #plt.show()
        return prob_total    

    #%% -----------------------------------------
    # Pixel disparity

    def pixel_two_image_disparity(self, img_left, img_right, spatial_sigma=3.0, range_sigma=0.1, radius=2):
        # =====================================================================
        # 1. High-Performance NumPy JBU Implementation
        # =====================================================================
        # if img_left.ndim == 2:
        #     img_left = img_left[..., np.newaxis]
        # if img_right.ndim == 2:
        #     img_right = img_right[..., np.newaxis]
            
        H_lr, W_lr          = img_left.shape
        H_hr, W_hr          = img_right.shape

        lr_diff             = np.abs(img_left - img_right)

        
        # Precompute spatial Gaussian weights
        y_coords, x_coords  = np.mgrid[-radius:radius+1, -radius:radius+1]
        spatial_dist_sq     = y_coords**2 + x_coords**2
        spatial_weights     = np.exp(-spatial_dist_sq / (2 * spatial_sigma**2))
        
        upsampled           = np.zeros_like(img_left, dtype=np.float32) #if C_lr == C_hr else np.zeros((H_hr, W_hr, C_lr), dtype=np.float32)
        norm_factor         = np.ones_like(img_left, dtype=np.float32)
        #hr_y, hr_x     = np.meshgrid(np.arange(H_hr), np.arange(W_hr), indexing='ij')
        y_index             = np.arange(radius, H_hr - radius - 1).reshape(-1, 1)
        x_index             = np.arange(radius, W_hr - radius - 1).reshape(1, -1)
        
        # Neighborhood match loop
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                s_w             = spatial_weights[dy + radius, dx + radius]
                if s_w < 1e-4:
                    continue

                #lr_val          = np.abs(img_left[y_index, x_index] - img_right[y_index + dy, x_index + dx])
                lr_val          = lr_diff[y_index + dy, x_index + dx]
                guide_diff      = img_left[y_index, x_index] - img_left[y_index + dy, x_index + dx]
                range_weights   = np.exp(-guide_diff**2 / (2 * range_sigma**2))
                
                #weight          = s_w * range_weights
                #upsampled[y_index, x_index]      += weight #lr_val * weight
                #norm_factor    += weight

                weight                            = s_w * range_weights
                upsampled[y_index, x_index]      += lr_val * weight
                norm_factor[y_index, x_index]    += weight                
                
        upsampled /= (norm_factor + 1e-8)
        return np.squeeze(upsampled)  

    def pixel_image_disparity(self, img_left, img_right, max_disparity = 64):
        "left and right pixel matcjing with bilateral filtering"

        log.debug(f'Calculating disparity difference with max_disparity {max_disparity}')
        h, w             = img_right.shape[:2]
        diff_array       = np.full((h, w, max_disparity), 0, dtype=np.float32)
        img_left, img_right = img_left.astype(np.float32), img_right.astype(np.float32)
        
        for shift in range(max_disparity):
            img_pixel_match = self.pixel_two_image_disparity(img_left[:, shift:], img_right[:, :w - shift], spatial_sigma=3.0, range_sigma=7.1, radius=4)
            diff_array[:, shift:, shift] = img_pixel_match

        log.debug('Done')
        return diff_array        

    #%% -----------------------------------------
    # Multiscale disparity

    def multiscale_disparity(self, img_left, img_right, debug_row=None):
        "compute row-wise left/right gabor channel inner products and show MxM matrix"

        row_index               = debug_row if debug_row is not None else 400
        debug_on                = debug_row is not None

        level_num               = 4
        max_disparity           = 64
        
        row_num,col_num        = img_left.shape[:2]
        T_weights               = [0.05, 0.1, 0.2]  # example weights for each level, should sum to 1

        img_left, img_right     = img_left.astype(np.float32), img_right.astype(np.float32)
        img_left_ref            = img_left.copy()

        # Gabor bank parameters.
        rot_left, rot_right     = np.pi/16*0, np.pi/10*0  # rotate left image Gabors slightly counterclockwise and right image Gabors slightly clockwise to better match floor pattern    
        #rot_left, rot_right   = 0, 0 
        thetas                  = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        lambdas                 = [8.0, 16.0, 32.0];  psis = [0.0, np.pi / 2];  ksize = 7; sigma  = 3.0;  gamma = 0.5
        thetas_left             = [t + rot_left for t in thetas]  # rotate left image Gabors slightly counterclockwise
        thetas_right            = [t - rot_right for t in thetas]
        bank_left, p_left        = self.gabor_bank_init(ksize=ksize, sigma=sigma, lambdas=lambdas, gamma=gamma, psis=psis, thetas=thetas_left)
        bank_right, p_right      = self.gabor_bank_init(ksize=ksize, sigma=sigma, lambdas=lambdas, gamma=gamma, psis=psis, thetas=thetas_right)         

        distance_total           = np.zeros((row_num, col_num ,level_num, max_disparity), dtype=np.float32)

        # calculate simple features
        feature_left            = self.pixel_features(img_left)  # shape (N, M, C)
        feature_right           = self.pixel_features(img_right) # shape (N, M, C)

        # normalize the responses across channels for each pixel to have zero mean but not variance. 
        feature_left             = self.gabor_normalize_responses(feature_left)
        feature_right            = self.gabor_normalize_responses(feature_right)         

        # only pixels in the valid range are considered for disparity estimation, 
        distance_left            = self.gabor_dispartity(feature_left, feature_right, max_disparity=max_disparity) # shape (N, M, D) 
        distance_total[:,:,0,:]      = distance_left

        for level in range(1, level_num):

            # scale factor
            scale_factor             = 2**(level-1)
        
            # processing one by one
            gabor_left               = self.gabor_bank_filter(img_left, bank=bank_left)
            gabor_right              = self.gabor_bank_filter(img_right, bank=bank_right) 

            # normalize the responses across channels for each pixel to have zero mean but not variance. 
            gabor_left               = self.gabor_normalize_responses(gabor_left)
            gabor_right              = self.gabor_normalize_responses(gabor_right)        

            # 3d volume distance between left and right gabor responses across channels for each pixel and disparity. This can be used as a cost volume for stereo matching, where lower values indicate more similar responses and thus more likely matches.
            distance_left            = self.gabor_dispartity(gabor_left, gabor_right, max_disparity=max_disparity//scale_factor) # shape (N, M, D)   

            # filter
            #distance_left           = self.anisotropic_filter_with_edges(distance_left, img_left, num_iter=8)
            #distance_left           = self.inverse_filter_with_edges(distance_left, img_left)
            #distance_left_interpolated   = self.joint_bilateral_upsampling(distance_left, img_left_ref, spatial_sigma=3.0, range_sigma=0.1, radius=7)
            #distance_left_interpolated   = zoom(distance_left_interpolated,zoom=(1,1,scale_factor))

            # interpolate to original 
            #cv2.resize(low_res_feat, (img_left_ref.shape[1], img_left_ref.shape[0]), interpolation=cv2.INTER_NEAREST)
            distance_left_interpolated    = zoom(distance_left,  zoom=(scale_factor, scale_factor, scale_factor), order=1)  
            distance_total[:,:,level,:]   = distance_left_interpolated

            # filter
            distance_total[:,:,level,:]   = self.anisotropic_filter_with_edges(distance_total[:,:,level,:], img_left_ref, num_iter=8)
            #distance_total[:,:,level,:]    = cv.ximgproc.guidedFilter( guide=img_left_ref, src=distance_total[:,:,level,:],  radius=8,  eps=1e-3 )  

            # # Edge-preserving sharpening
            # bilateral                   = cv.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
            # edge_mask                   = cv.subtract(img, bilateral)
            # sharpened_robust            = cv.add(img, cv.multiply(edge_mask, 1.5))        

            # downscale images
            img_left                = zoom(img_left,  zoom=0.5, order=1)
            img_right               = zoom(img_right, zoom=0.5, order=1) 

    
        # show the difference data
        if debug_on:
            img_list                 = [distance_total[debug_row,:,m,:].squeeze().T for m in range(level_num)]
            ttl_list                 = [f'Level {m} Distance Volume (row {debug_row})' for m in range(level_num)]
            self.show_subset(img_list, ttl_list, col_num=2)

        # convert to probability using soft max over the disparity dimension, which can help normalize the scores and make them more interpretable as probabilities. The temperature parameter T can be tuned to control the sharpness of the distribution, with lower values leading to a more peaked distribution and higher values leading to a softer distribution.
        prob_total               = self.softmax_with_threshold(-distance_total, dim=3, T=0.1, x_thr=-1) #T_weights[k])  # shape (N, M, D), higher is more similar

        if debug_on:        
            #img_list                 = [prob_total[debug_row,:,m,:].squeeze().T for m in range(3)]
            img_list                 = [prob_total[debug_row,:,m,:].squeeze().T for m in range(level_num)]
            ttl_list                 = [f'Level {m} Probability Volume (row {debug_row})' for m in range(level_num )]
            self.show_subset(img_list, ttl_list, col_num=2)

        # do edge filtering
        prob_filtered              = prob_total.copy()
        # for m in range(level_num):
        #     prob_filtered[:,:,m,:]  = self.anisotropic_filter_with_edges(prob_total[:,:,m,:], img_left_ref, num_iter = 8)

        if debug_on:
            img_list                 = [prob_filtered[debug_row,:,m,:].squeeze().T for m in range(level_num)]
            ttl_list                 = [f'Level {m} Probability Filtered (row {debug_row})' for m in range(level_num )]
            self.show_subset(img_list, ttl_list, col_num=2)            


        # collaps
        #prob_total_final         = np.sum(prob_filtered, axis=2).squeeze()/2  # weight by confidence, shape (N, M, D)   
        prob_total_final         = prob_filtered[:,:,0,:] 
        for m in range(1,level_num):
            prob_total_final      = prob_total_final + (1-prob_total_final) * prob_filtered[:,:,m,:]
        # prob_total_final         = prob_total_final + (1-prob_total_final) * prob_filtered[:,:,2,:]
        # prob_total_final         = prob_total_final + (1-prob_total_final) * prob_filtered[:,:,3,:]
        # prob_total_final         = np.clip(prob_total_final.squeeze(), 0, 1)
        #prob_total_final         = np.sum(prob_total / (np.sum(prob_total, axis=2, keepdims=True) + 0.01), axis=2).squeeze()
        #prob_total_filter        = self.probability_bilateral_filtering(img_left_ref, prob_total_final, spatial_sigma=3.0, range_sigma=5.1, radius=3, iter_num=1) 
        
        #prob_total_filter        = self.joint_bilateral_filtering(img_left_ref, prob_total_final, spatial_sigma=3.0, range_sigma=5.1, radius=5, iter_num=4) 
        #prob_total_filter        = self.anisotropic_filter_with_edges(prob_total_final, img_left_ref, num_iter = 8)
        prob_total_filter         = prob_total_final

        # compute edges
        #img_edges                = self.compute_edges(img_left_ref)

        # do some filtering on the final probability volume to smooth out noise and improve the disparity estimation. This can be done using a Gaussian filter, a median filter, or an anisotropic diffusion filter, depending on the characteristics of the data and the desired level of smoothing. The goal is to reduce spurious peaks in the probability distribution while preserving important features such as edges and texture.
        #prob_total_filter         = self.adaptive_filter_3d_fast(prob_total_final, kernel_size = 15, kappa = 0.1)  # shape (N, M, D)
        #prob_total_filter         = self.anisotropic_filter_avergaing(prob_total_final, img_left_ref)  # shape (N, M, D)
        #prob_total_filter        = self.joint_bilateral_upsampling(prob_total_final, img_left_ref, spatial_sigma=3.0, range_sigma=7.1, radius=4)
        # prob_total_filter        = prob_total_final.copy()
        # for i in range(8):
        #     prob_total_filter    = self.joint_bilateral_filtering(img_left_ref, prob_total_filter, spatial_sigma=3.0, range_sigma=5.1, radius=3)
        #prob_total_filter        = self.joint_bilateral_filtering(img_left_ref, prob_total_final, spatial_sigma=3.0, range_sigma=5.1, radius=3, iter_num=8) 

        if debug_on:
            img_list                 = [prob_total_final[debug_row,:,:].squeeze().T , prob_total_filter[debug_row,:,:].squeeze().T]
            ttl_list                 = [f'Final Probability Volume (row {debug_row})', f'Filtered Final Probability Volume (row {debug_row})']
            self.show_subset(img_list, ttl_list, col_num=1)        

        disp_index               = self.estimate_disparity_from_prob(prob_total_filter, estim_type = 4) # ensure values are within the valid range
        disp_confidence          = np.max(prob_total_filter, axis=2)  # shape (N, M)
        disp_index[disp_confidence < 0.1]   = 0  # mask out low confidence areas

        # disp_confidence_filter    = disp_confidence.copy()
        # for i in range(8):
        #     disp_confidence_filter    = self.joint_bilateral_filtering(img_left_ref, disp_confidence_filter, spatial_sigma=3.0, range_sigma=5.1, radius=2)
        # disp_confidence_filter   = self.joint_bilateral_filtering(img_left_ref, disp_confidence, spatial_sigma=3.0, range_sigma=5.1, radius=2, iter_num=4)

        # img_list                 = [img_left_ref, disp_index , disp_confidence, disp_confidence_filter]
        # ttl_list                 = ['Left Image', 'Estimated Disparity Map', 'Estimated Disparity Confidence Map','Filtered Confidence Map']
        # self.show_subset(img_list, ttl_list, col_num=2) 

        if debug_on:
            img_list                 = [img_left_ref, disp_index , disp_confidence]
            ttl_list                 = ['Left Image', 'Estimated Disparity Map', 'Estimated Confidence Map']
            self.show_subset(img_list, ttl_list, col_num=1)         

        # if debug and row_index is not None:
        #     self.debug_gabor_image_disparity_multiscale(debug_levels, prob_total, row_index=row_index)

        #plt.show(block=False)
        return disp_index    

    def multiscale_disparity_pixel_features(self, img_left, img_right, debug_row=None):
        "compute row-wise left/right pixel features disparity"
        row_index               = debug_row if debug_row is not None else 400
        debug                   = debug_row is not None

        feature_types           = ['center','left','right','up','down','center_big']
        level_num               = len(feature_types)
        max_disparity           = 64
        row_num,col_num        = img_left.shape[:2]


        img_left, img_right     = img_left.astype(np.float32), img_right.astype(np.float32)
        img_left_ref            = img_left.copy()

        distance_total           = np.zeros((row_num, col_num ,level_num, max_disparity), dtype=np.float32)


        for level in range(0, level_num):

            # scale factor
            feature_type             = feature_types[level]
        
            # calculate simple features
            feature_left            = self.pixel_features(img_left, feat_type=feature_type)  # shape (N, M, C)
            feature_right           = self.pixel_features(img_right, feat_type=feature_type) # shape (N, M, C)

            # normalize the responses across channels for each pixel to have zero mean but not variance. 
            feature_left             = self.gabor_normalize_responses(feature_left)
            feature_right            = self.gabor_normalize_responses(feature_right)        

            # 3d volume distance between left and right gabor responses across channels for each pixel and disparity. This can be used as a cost volume for stereo matching, where lower values indicate more similar responses and thus more likely matches.
            distance_left            = self.gabor_dispartity(feature_left, feature_right, max_disparity=max_disparity) # shape (N, M, D)   

            # filter
            #distance_left            = self.anisotropic_filter_with_edges(distance_left, img_left_ref, num_iter=8)
            #distance_left            = self.joint_bilateral_filtering(img_left_ref, distance_left, spatial_sigma=3.0, range_sigma=4.1, radius=3, iter_num=3) 

            # interpolate to original   
            distance_total[:,:,level,:]   = distance_left


        # show the difference data
        img_list                 = [distance_total[debug_row,:,m,:].squeeze().T for m in range(level_num)]
        ttl_list                 = [f'Level {m} Distance Volume (row {debug_row})' for m in range(level_num)]
        self.show_subset(img_list, ttl_list, col_num=2)

        # convert to probability using soft max over the disparity dimension, which can help normalize the scores and make them more interpretable as probabilities. The temperature parameter T can be tuned to control the sharpness of the distribution, with lower values leading to a more peaked distribution and higher values leading to a softer distribution.
        prob_total               = self.softmax_with_threshold(-distance_total, dim=3, T=0.1, x_thr=-1) #T_weights[k])  # shape (N, M, D), higher is more similar
        
        #img_list                 = [prob_total[debug_row,:,m,:].squeeze().T for m in range(3)]
        img_list                 = [prob_total[debug_row,:,m,:].squeeze().T for m in range(level_num)]
        ttl_list                 = [f'Level {m} Probability Volume (row {debug_row})' for m in range(level_num )]
        self.show_subset(img_list, ttl_list, col_num=2)

        # do edge filtering
        prob_filtered              = prob_total.copy()
        for m in range(level_num):
            prob_filtered[:,:,m,:]  = self.anisotropic_filter_with_edges(prob_total[:,:,m,:], img_left_ref, num_iter = 4)

        img_list                 = [prob_filtered[debug_row,:,m,:].squeeze().T for m in range(level_num)]
        ttl_list                 = [f'Level {m} Probability Filtered (row {debug_row})' for m in range(level_num )]
        self.show_subset(img_list, ttl_list, col_num=2)            


        # collaps
        prob_total_final         = np.mean(prob_filtered, axis=2).squeeze()  # weight by confidence, shape (N, M, D)   
        # prob_total_final         = prob_filtered[:,:,0,:] 
        # prob_total_final         = prob_total_final + (1-prob_total_final) * prob_filtered[:,:,1,:]
        # prob_total_final         = prob_total_final + (1-prob_total_final) * prob_filtered[:,:,2,:]
        # prob_total_final         = prob_total_final + (1-prob_total_final) * prob_filtered[:,:,3,:]
        # prob_total_final         = np.clip(prob_total_final.squeeze(), 0, 1)
        #prob_total_final         = np.sum(prob_total / (np.sum(prob_total, axis=2, keepdims=True) + 0.01), axis=2).squeeze()
        #prob_total_filter        = self.probability_bilateral_filtering(img_left_ref, prob_total_final, spatial_sigma=3.0, range_sigma=5.1, radius=3, iter_num=1) 
        #prob_total_filter        = self.joint_bilateral_filtering(img_left_ref, prob_total_final, spatial_sigma=3.0, range_sigma=5.1, radius=2, iter_num=8) 
        prob_total_filter        = prob_total_final #self.anisotropic_filter_with_edges(prob_total_final, img_left_ref, num_iter = 8)

        # compute edges
        #img_edges                = self.compute_edges(img_left_ref)

        # do some filtering on the final probability volume to smooth out noise and improve the disparity estimation. This can be done using a Gaussian filter, a median filter, or an anisotropic diffusion filter, depending on the characteristics of the data and the desired level of smoothing. The goal is to reduce spurious peaks in the probability distribution while preserving important features such as edges and texture.
        #prob_total_filter         = self.adaptive_filter_3d_fast(prob_total_final, kernel_size = 15, kappa = 0.1)  # shape (N, M, D)
        #prob_total_filter         = self.anisotropic_filter_avergaing(prob_total_final, img_left_ref)  # shape (N, M, D)
        #prob_total_filter        = self.joint_bilateral_upsampling(prob_total_final, img_left_ref, spatial_sigma=3.0, range_sigma=7.1, radius=4)
        # prob_total_filter        = prob_total_final.copy()
        # for i in range(8):
        #     prob_total_filter    = self.joint_bilateral_filtering(img_left_ref, prob_total_filter, spatial_sigma=3.0, range_sigma=5.1, radius=3)
        #prob_total_filter        = self.joint_bilateral_filtering(img_left_ref, prob_total_final, spatial_sigma=3.0, range_sigma=5.1, radius=3, iter_num=8) 

        img_list                 = [prob_total_final[debug_row,:,:].squeeze().T , prob_total_filter[debug_row,:,:].squeeze().T]
        ttl_list                 = [f'Final Probability Volume (row {debug_row})', f'Filtered Final Probability Volume (row {debug_row})']
        self.show_subset(img_list, ttl_list, col_num=1)        

        disp_index               = self.estimate_disparity_from_prob(prob_total_filter, estim_type = 2) # ensure values are within the valid range
        disp_confidence          = np.max(prob_total_filter, axis=2)  # shape (N, M)
        disp_index[disp_confidence < 0.1]   = 0  # mask out low confidence areas

        # disp_confidence_filter    = disp_confidence.copy()
        # for i in range(8):
        #     disp_confidence_filter    = self.joint_bilateral_filtering(img_left_ref, disp_confidence_filter, spatial_sigma=3.0, range_sigma=5.1, radius=2)
        #disp_confidence_filter   = self.joint_bilateral_filtering(img_left_ref, disp_confidence, spatial_sigma=3.0, range_sigma=5.1, radius=2, iter_num=4)

        img_list                 = [img_left_ref, disp_index , disp_confidence, disp_confidence]
        ttl_list                 = ['Left Image', 'Estimated Disparity Map', 'Estimated Disparity Confidence Map','Filtered Confidence Map']
        self.show_subset(img_list, ttl_list, col_num=2) 

        # if debug and row_index is not None:
        #     self.debug_gabor_image_disparity_multiscale(debug_levels, prob_total, row_index=row_index)

        plt.show()
        return prob_total    

    def multiscale_disparity_with_energy(self, img_left, img_right, debug_row=None):
        "compute row-wise left/right gabor channel inner products and show MxM matrix"

        from scipy.ndimage import correlate1d
        #kernel                  = np.array([0.1, 0.2, 0.4, 0.2, 0.1])  # must sum to 1 for smoothing
        kernel                  = np.array([0.2, 0.6, 0.2]) 

        row_index               = debug_row if debug_row is not None else 400
        debug_on                = debug_row is not None

        level_num               = 4
        max_disparity           = 128
        
        row_num,col_num        = img_left.shape[:2]
        T_weights               = [0.05, 0.1, 0.2]  # example weights for each level, should sum to 1

        img_left, img_right     = img_left.astype(np.float32), img_right.astype(np.float32)
        img_left_ref            = img_left.copy()

        # Gabor bank parameters.
        rot_left, rot_right     = np.pi/16*0, np.pi/10*0  # rotate left image Gabors slightly counterclockwise and right image Gabors slightly clockwise to better match floor pattern    
        #rot_left, rot_right   = 0, 0 
        thetas                  = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        lambdas                 = [8.0, 16.0, 32.0];  psis = [0.0, np.pi / 2];  ksize = 7; sigma  = 3.0;  gamma = 0.5
        thetas_left             = [t + rot_left for t in thetas]  # rotate left image Gabors slightly counterclockwise
        thetas_right            = [t - rot_right for t in thetas]
        bank_left, p_left        = self.gabor_bank_init(ksize=ksize, sigma=sigma, lambdas=lambdas, gamma=gamma, psis=psis, thetas=thetas_left)
        bank_right, p_right      = self.gabor_bank_init(ksize=ksize, sigma=sigma, lambdas=lambdas, gamma=gamma, psis=psis, thetas=thetas_right)         

        distance_total           = np.zeros((row_num, col_num ,level_num, max_disparity), dtype=np.float32)
        energy_total             = np.zeros((row_num, col_num ,level_num), dtype=np.float32)

        # # calculate simple features
        # feature_left            = self.pixel_features(img_left)  # shape (N, M, C)
        # feature_right           = self.pixel_features(img_right) # shape (N, M, C)

        # # normalize the responses across channels for each pixel to have zero mean but not variance. 
        # feature_left,  energy_left            = self.gabor_normalize_responses_with_energy(feature_left)
        # feature_right, energy_right           = self.gabor_normalize_responses_with_energy(feature_right)         

        # # only pixels in the valid range are considered for disparity estimation, 
        # distance_left                = self.gabor_dispartity(feature_left, feature_right, max_disparity=max_disparity) # shape (N, M, D) 
        # distance_total[:,:,0,:]      = distance_left
        # energy_total[:,:,0]          = energy_left

        # # upscale images
        # img_left                = zoom(img_left,  zoom=2, order=1)
        # img_right               = zoom(img_right, zoom=2, order=1) 

        for level in range(0, level_num):

            # scale factor
            scale_factor             = 2**(level-0)
        
            # processing one by one
            gabor_left               = self.gabor_bank_filter(img_left, bank=bank_left)
            gabor_right              = self.gabor_bank_filter(img_right, bank=bank_right) 

            # normalize the responses across channels for each pixel to have zero mean but not variance. 
            gabor_left,  energy_left    = self.gabor_normalize_responses_with_energy(gabor_left)
            gabor_right, energy_right   = self.gabor_normalize_responses_with_energy(gabor_right)        

            # 3d volume distance between left and right gabor responses across channels for each pixel and disparity. This can be used as a cost volume for stereo matching, where lower values indicate more similar responses and thus more likely matches.
            distance_left            = self.gabor_dispartity(gabor_left, gabor_right, max_disparity=max_disparity//scale_factor) # shape (N, M, D)   

            # filter
            #distance_left           = self.anisotropic_filter_with_edges(distance_left, img_left, num_iter=8)
            #distance_left           = self.inverse_filter_with_edges(distance_left, img_left)
            #distance_left_interpolated   = self.joint_bilateral_upsampling(distance_left, img_left_ref, spatial_sigma=3.0, range_sigma=0.1, radius=7)
            #distance_left_interpolated   = zoom(distance_left_interpolated,zoom=(1,1,scale_factor))

            # interpolate to original 
            #cv2.resize(low_res_feat, (img_left_ref.shape[1], img_left_ref.shape[0]), interpolation=cv2.INTER_NEAREST)
            distance_total[:,:,level,:]   = zoom(distance_left,  zoom=(scale_factor, scale_factor, scale_factor), order=1)  
            energy_total[:,:,level]       = zoom(energy_left,    zoom=(scale_factor, scale_factor), order=1) 
            #energy_total[:,:,level]       = cv.resize(energy_left, (img_left_ref.shape[1], img_left_ref.shape[0]), interpolation=cv.INTER_LINEAR)             

            # compensate
            distance_total[:,:,level,:]   = np.roll(distance_total[:,:,level,:],axis=2,shift=-level)

            # # smoooth
            # for s in range(level):
            #     distance_total[:,:,level,:]   = correlate1d(distance_total[:,:,level,:], kernel, axis=2, mode='nearest')

            #distance_total[:,:,level,:]   = self.upscale_img_array(distance_left, img_left_ref, upscale_type = 2)
            #energy_total[:,:,level]       = self.upscale_img_array(energy_left, img_left_ref, upscale_type = 2)
            # filter
            #distance_total[:,:,level,:]   = self.anisotropic_filter_with_edges(distance_total[:,:,level,:], img_left_ref, num_iter=8)
            #distance_total[:,:,level,:]    = cv.ximgproc.guidedFilter( guide=img_left_ref, src=distance_total[:,:,level,:],  radius=8,  eps=1e-3 )  

            # # Edge-preserving sharpening
            # bilateral                   = cv.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
            # edge_mask                   = cv.subtract(img, bilateral)
            # sharpened_robust            = cv.add(img, cv.multiply(edge_mask, 1.5))        

            # downscale images
            img_left                = zoom(img_left,  zoom=0.5, order=1)
            img_right               = zoom(img_right, zoom=0.5, order=1) 

    
        # show the difference data
        if debug_on:
            img_list                 = [distance_total[debug_row,:,m,:].squeeze().T for m in range(level_num)]
            ttl_list                 = [f'Level {m} Distance Volume (row {debug_row})' for m in range(level_num)]
            self.show_subset(img_list, ttl_list, col_num=2)

            # show the energy data
            img_list                 = [energy_total[:,:,m] for m in range(level_num)]
            ttl_list                 = [f'Level {m} Energy Features (row {debug_row})' for m in range(level_num)]
            self.show_subset(img_list, ttl_list, col_num=2)        

        # convert to probability using soft max over the disparity dimension, which can help normalize the scores and make them more interpretable as probabilities. The temperature parameter T can be tuned to control the sharpness of the distribution, with lower values leading to a more peaked distribution and higher values leading to a softer distribution.
        prob_total               = self.softmax_with_threshold(-distance_total, dim=3, T=0.1, x_thr=-3) #T_weights[k])  # shape (N, M, D), higher is more similar

        if debug_on:        
            #img_list                 = [prob_total[debug_row,:,m,:].squeeze().T for m in range(3)]
            img_list                 = [prob_total[debug_row,:,m,:].squeeze().T for m in range(level_num)]
            ttl_list                 = [f'Level {m} Probability Volume (row {debug_row})' for m in range(level_num )]
            self.show_subset(img_list, ttl_list, col_num=2)

        # debug_row               += 1
        # img_list                 = [prob_total[debug_row,:,m,:].squeeze().T for m in range(level_num)]
        # ttl_list                 = [f'Level {m} Probability Volume (row {debug_row})' for m in range(level_num )]
        # self.show_subset(img_list, ttl_list, col_num=2)      

        prob_energy              = energy_total[:,:,:]
        prob_energy              = prob_energy/(np.sum(prob_energy, axis=2, keepdims=True) + 1e-1)

        if debug_on:
            img_list                 = [prob_energy[:,:,m] for m in range(level_num)]
            ttl_list                 = [f'Level {m} Probability Energy (row {debug_row})' for m in range(level_num )]
            self.show_subset(img_list, ttl_list, col_num=2)         

        # do edge filtering
        prob_filtered              = prob_total.copy()#*prob_energy[:,:,:,np.newaxis]
        # for m in range(level_num):
        #     prob_filtered[:,:,m,:]  = self.anisotropic_filter_with_edges(prob_total[:,:,m,:], img_left_ref, num_iter = 8)

        if debug_on:
            img_list                 = [prob_filtered[debug_row,:,m,:].squeeze().T for m in range(level_num)]
            ttl_list                 = [f'Level {m} Probability Filtered (row {debug_row})' for m in range(level_num )]
            self.show_subset(img_list, ttl_list, col_num=2)            

  

        # collaps
        #prob_total_final         = np.sum(prob_filtered, axis=2).squeeze()/2  # weight by confidence, shape (N, M, D)   
        prob_total_final         = prob_filtered[:,:,0,:] 
        for m in range(1,level_num):
            #prob_total_final      = prob_total_final + (1-prob_total_final) * prob_filtered[:,:,m,:]
            prob_max              = np.max(prob_total_final,axis=2)[:,:,np.newaxis]
            prob_total_final      = prob_total_final + (1-prob_max) * prob_filtered[:,:,m,:]

        # prob_total_final         = np.clip(prob_total_final.squeeze(), 0, 1)
        #prob_total_final         = np.sum(prob_total / (np.sum(prob_total, axis=2, keepdims=True) + 0.01), axis=2).squeeze()
        #prob_total_filter        = self.probability_bilateral_filtering(img_left_ref, prob_total_final, spatial_sigma=3.0, range_sigma=5.1, radius=3, iter_num=1) 
        
        #prob_total_filter        = self.joint_bilateral_filtering(img_left_ref, prob_total_final, spatial_sigma=3.0, range_sigma=5.1, radius=5, iter_num=4) 
        #prob_total_filter        = self.anisotropic_filter_with_edges(prob_total_final, img_left_ref, num_iter = 8)
        prob_total_filter         = prob_total_final
        #prob_total_filter         = self.softmax_local_maxima(prob_total_final, kernel_size=7, T=0.1, x_thr=-1) #T_weights[k])  # shape (N, M, D), higher is more similar
   

        # compute edges
        #img_edges                = self.compute_edges(img_left_ref)

        # do some filtering on the final probability volume to smooth out noise and improve the disparity estimation. This can be done using a Gaussian filter, a median filter, or an anisotropic diffusion filter, depending on the characteristics of the data and the desired level of smoothing. The goal is to reduce spurious peaks in the probability distribution while preserving important features such as edges and texture.
        #prob_total_filter         = self.adaptive_filter_3d_fast(prob_total_final, kernel_size = 15, kappa = 0.1)  # shape (N, M, D)
        #prob_total_filter         = self.anisotropic_filter_avergaing(prob_total_final, img_left_ref)  # shape (N, M, D)
        #prob_total_filter        = self.joint_bilateral_upsampling(prob_total_final, img_left_ref, spatial_sigma=3.0, range_sigma=7.1, radius=4)
        # prob_total_filter        = prob_total_final.copy()
        # for i in range(8):
        #     prob_total_filter    = self.joint_bilateral_filtering(img_left_ref, prob_total_filter, spatial_sigma=3.0, range_sigma=5.1, radius=3)
        #prob_total_filter        = self.joint_bilateral_filtering(img_left_ref, prob_total_final, spatial_sigma=3.0, range_sigma=5.1, radius=3, iter_num=8) 

        if debug_on:
            img_list                 = [prob_total_final[debug_row,:,:].squeeze().T , prob_total_filter[debug_row,:,:].squeeze().T]
            ttl_list                 = [f'Final Probability Volume (row {debug_row})', f'Filtered Final Probability Volume (row {debug_row})']
            self.show_subset(img_list, ttl_list, col_num=1)        

        disp_index               = self.estimate_disparity_from_prob(prob_total_filter, estim_type = 4) # ensure values are within the valid range
        disp_confidence          = np.max(prob_total_filter, axis=2)  # shape (N, M)
        disp_index[disp_confidence < 0.1]   = 0  # mask out low confidence areas

        # disp_confidence_filter    = disp_confidence.copy()
        # for i in range(8):
        #     disp_confidence_filter    = self.joint_bilateral_filtering(img_left_ref, disp_confidence_filter, spatial_sigma=3.0, range_sigma=5.1, radius=2)
        # disp_confidence_filter   = self.joint_bilateral_filtering(img_left_ref, disp_confidence, spatial_sigma=3.0, range_sigma=5.1, radius=2, iter_num=4)
        #disp_confidence_filter    = self.softmax_local_maxima(disp_confidence, kernel_size=7, T=0.1, x_thr=-1) #T_weigh

        # img_list                 = [img_left_ref, disp_index , disp_confidence, disp_confidence_filter]
        # ttl_list                 = ['Left Image', 'Estimated Disparity Map', 'Estimated Disparity Confidence Map','Filtered Confidence Map']
        # self.show_subset(img_list, ttl_list, col_num=2) 

        if debug_on:
            img_list                 = [img_left_ref, disp_index , disp_confidence]
            ttl_list                 = ['Left Image', 'Estimated Disparity Map', 'Estimated Confidence Map']
            self.show_subset(img_list, ttl_list, col_num=1)         

            plt.show()
        return disp_index

    def multiscale_disparity_edge_aware(self, img_left, img_right, debug_row=None):
        """
        Edge-aware variant of multiscale_disparity_with_energy, aimed at removing the
        disparity smearing that shows up around well-defined, high-contrast image edges.

        The smearing has three separate causes in the original pipeline, each addressed here:
          1. Pyramid fusion upsamples each coarse level with a bilinear `zoom`, which turns any
             sharp step at that level into a multi-pixel ramp once it reaches full resolution.
             Fix: upsample the spatial axes with nearest-neighbor (order=0) instead of linear
             interpolation - it never averages across a step.
          2. The raw per-pixel Gabor cost has no spatial regularization, so noisy/ambiguous
             pixels are not distinguished from pixels that are genuinely near a depth edge.
             Fix: guided-filter the whole probability volume (all disparity channels in one
             call) using the left image as guide - this aggregates cost within regions of
             similar color and stops at color edges, instead of blending across them.
          3. estimate_disparity_from_prob(estim_type=2-4) takes a probability-weighted average
             over the *entire* disparity axis. At a depth edge the distribution is genuinely
             bimodal (a foreground peak and a background peak), so averaging them produces an
             intermediate disparity that does not exist in the scene.
             Fix: estim_type=5 - hard argmax with a local 3-tap parabola refinement, which
             gives sub-pixel precision without ever blending in a distant mode.
        Ambiguous pixels (low confidence, or a strong runner-up peak - the signature of a pixel
        straddling an edge) are then cleaned up with a targeted joint-bilateral pass guided by
        the left image, instead of a global filter that would blur confident regions too.

        Not implemented here (out of scope for a drop-in post-processing variant):
          - Adaptive-support-weight matching / narrower cost-aggregation windows, which would
            require reworking gabor_dispartity itself.
          - True left-right consistency checking (would need a second, mirrored matching pass);
            the peak-ambiguity ratio below is used as a practical proxy instead.
        """
        row_index               = debug_row if debug_row is not None else 400
        debug                   = debug_row is not None

        level_num               = 4
        max_disparity           = 128

        row_num, col_num        = img_left.shape[:2]

        img_left, img_right     = img_left.astype(np.float32), img_right.astype(np.float32)
        img_left_ref            = img_left.copy()

        # Gabor bank parameters - same bank as multiscale_disparity_with_energy.
        thetas                  = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        lambdas                 = [8.0, 16.0, 32.0];  psis = [0.0, np.pi / 2];  ksize = 11; sigma = 3.0;  gamma = 0.5
        bank_left, _            = self.gabor_bank_init(ksize=ksize, sigma=sigma, lambdas=lambdas, gamma=gamma, psis=psis, thetas=thetas)
        bank_right, _           = self.gabor_bank_init(ksize=ksize, sigma=sigma, lambdas=lambdas, gamma=gamma, psis=psis, thetas=thetas)

        distance_total          = np.zeros((row_num, col_num, level_num, max_disparity), dtype=np.float32)
        energy_total            = np.zeros((row_num, col_num, level_num), dtype=np.float32)

        for level in range(level_num):

            scale_factor            = 2 ** level

            gabor_left              = self.gabor_bank_filter(img_left, bank=bank_left)
            gabor_right             = self.gabor_bank_filter(img_right, bank=bank_right)

            gabor_left,  energy_left    = self.gabor_normalize_responses_with_energy(gabor_left)
            gabor_right, energy_right   = self.gabor_normalize_responses_with_energy(gabor_right)

            distance_left            = self.gabor_dispartity(gabor_left, gabor_right, max_disparity=max_disparity // scale_factor)  # (row_lvl, col_lvl, D_lvl)

            if scale_factor == 1:
                distance_full            = distance_left
                energy_full              = energy_left
            else:
                # 1) disparity axis: a simple re-indexing to full-resolution disparity units,
                #    not a spatial resize - linear interpolation here is fine.
                distance_full            = zoom(distance_left, zoom=(1, 1, scale_factor), order=1)
                # 2) spatial axes: nearest-neighbor, so a sharp step at this coarse level stays
                #    a step instead of turning into a multi-pixel ramp at full resolution.
                distance_full            = zoom(distance_full, zoom=(scale_factor, scale_factor, 1), order=0)
                energy_full              = zoom(energy_left, zoom=(scale_factor, scale_factor), order=0)

            distance_total[:, :, level, :]  = distance_full
            energy_total[:, :, level]       = energy_full

            # compensate for the level shift, same as multiscale_disparity_with_energy
            distance_total[:, :, level, :]  = np.roll(distance_total[:, :, level, :], axis=2, shift=-level)

            img_left                = zoom(img_left,  zoom=0.5, order=1)
            img_right               = zoom(img_right, zoom=0.5, order=1)

        # show the difference data
        if debug:
            img_list                 = [distance_total[debug_row,:,m,:].squeeze().T for m in range(level_num)]
            ttl_list                 = [f'Level {m} Distance Volume (row {debug_row})' for m in range(level_num)]
            self.show_subset(img_list, ttl_list, col_num=2)

            # show the energy data
            img_list                 = [energy_total[:,:,m] for m in range(level_num)]
            ttl_list                 = [f'Level {m} Energy Features (row {debug_row})' for m in range(level_num)]
            self.show_subset(img_list, ttl_list, col_num=2)              

        prob_total               = self.softmax_with_threshold(-distance_total, dim=3, T=0.1, x_thr=-2).astype(np.float32)  # shape (N, M, level, D); softmax_with_threshold upcasts to float64, but guidedFilter only accepts CV_32F/CV_8U

        # Edge-aware cost aggregation: guided-filter every disparity channel of each level's
        # probability volume in one call, guided by the full-resolution left image. This is the
        # direct replacement for the commented-out anisotropic_filter_with_edges stub in
        # multiscale_disparity_with_energy.
        prob_filtered            = prob_total.copy()
        # prob_filtered            = np.empty_like(prob_total)
        # for level in range(level_num):
        #     prob_filtered[:, :, level, :] = cv.ximgproc.guidedFilter(
        #         guide=img_left_ref, src=prob_total[:, :, level, :], radius=5, eps=50.0
        #     )

        if debug:
            img_list = [prob_total[debug_row, :, m, :].squeeze().T for m in range(level_num)]
            ttl_list = [f'Level {m} Probability Volume (row {debug_row})' for m in range(level_num)]
            self.show_subset(img_list, ttl_list, col_num=2)

            img_list = [prob_filtered[debug_row, :, m, :].squeeze().T for m in range(level_num)]
            ttl_list = [f'Level {m} Edge-Filtered Probability (row {debug_row})' for m in range(level_num)]
            self.show_subset(img_list, ttl_list, col_num=2)

        # combine levels
        prob_total_final          = prob_filtered[:, :, 0, :]
        for m in range(1, level_num):
            prob_max                 = np.max(prob_total_final, axis=2)[:, :, np.newaxis]
            prob_total_final         = prob_total_final + (1 - prob_max) * prob_filtered[:, :, m, :]

        # hard argmax + local parabola sub-pixel refinement - never blends two separated modes
        disp_index                = self.estimate_disparity_from_prob(prob_total_final, estim_type=5)

        # flag pixels whose match is ambiguous (low confidence, or a strong runner-up peak -
        # the signature of a pixel straddling a depth edge) and clean up only those, guided by
        # the left image, so confident regions are left untouched.
        ratio, disp_confidence     = self.disparity_peak_ambiguity(prob_total_final)
        ambiguous                  = (ratio > 0.6) | (disp_confidence < 0.1)

        disp_index_clean           = self.joint_bilateral_filtering(
            img_left_ref, disp_index, spatial_sigma=3.0, range_sigma=5.0, radius=3, iter_num=2
        )
        disp_index_final             = disp_index.copy()
        disp_index_final[ambiguous]  = disp_index_clean[ambiguous]
        disp_index_final[disp_confidence < 0.05] = 0  # mask out very low confidence areas

        if debug:
            img_list = [img_left_ref, disp_index, ambiguous.astype(np.float32), disp_index_final]
            ttl_list = ['Left Image', 'Disparity (pre-cleanup)', 'Ambiguous / Edge Pixels', 'Disparity (edge-aware)']
            self.show_subset(img_list, ttl_list, col_num=2)
            plt.show()

        return disp_index_final

    #%% -----------------------------------------
    # Functional blocks
    #
    def convert_depth_to_volume_and_back(self, img_left, img_right, img_depth):
        "test function to convert depth image to 3d volume and back"
        # convert depth image to 3d volume by stacking shifted versions of the image along a new dimension, where each shift corresponds to a different disparity level. This can be useful for stereo matching algorithms that operate on 3D cost volumes, where the cost for each pixel is computed across a range of disparities. By converting the depth image into a 3D volume, we can more easily compare it to the cost volume and compute metrics such as mean squared error or cross-entropy loss for training or evaluation purposes.
        max_disparity = 128

        # reinit ROI
        width, height       = img_depth.shape[1], img_depth.shape[0]

        # recover disparity         
        img_disparity       = self.convert_depth_to_disparity(img_depth, 500) 

        # reproduce volume from matching features
        vol_disparity_new   = disparity_to_volume(img_disparity, D=max_disparity)  # convert to 3D volume by stacking shifted versions of the image along a new dimension, where each shift corresponds to a different disparity level
        



        # downscale and back
        scale_factor        = 0.5  # Downsample to 25% of original size (16x16x16)
        downsampled         = zoom(vol_disparity_new, zoom=scale_factor, order=1)

        img_left, img_right = zoom(img_left, zoom=scale_factor, order=1), zoom(img_right, zoom=scale_factor, order=1)

        # 2. UPSAMPLE
        # To return to original size, invert the scale factor (e.g., 1 / 0.5 = 2.0)
        upsample_factor     = 1.0 / scale_factor
        vol_disparity_new   = zoom(downsampled, zoom=upsample_factor, order=1)

        # get the disparity back
        img_disparity_new   = np.argmax(vol_disparity_new, axis=2).astype(np.float32)  # convert back to 2D disparity map   
          
        # return back
        img_depth_new       = self.convert_disparity_to_depth(img_disparity_new, 500)

        # measure inside the roi - only for vertical walls
        #ret                 = self.compute_roi_mean_std(img_depth, img_depth_new)

        return img_depth_new

    def convert_depth_from_left_and_right(self, img_left, img_right, img_depth):
        "function to compute depth from images"
        # convert depth image to 3d volume by stacking shifted versions of the image along a new dimension, where each shift corresponds to a different disparity level. This can be useful for stereo matching algorithms that operate on 3D cost volumes, where the cost for each pixel is computed across a range of disparities. By converting the depth image into a 3D volume, we can more easily compare it to the cost volume and compute metrics such as mean squared error or cross-entropy loss for training or evaluation purposes.
        max_disparity       = 64


        # recover disparity         
        img_disparity       = self.convert_depth_to_disparity(img_depth, 500) 

        # reproduce volume from matching features
        img_disparity_new   = self.multiscale_disparity(img_left, img_right)

        # return back
        img_depth_new       = self.convert_disparity_to_depth(img_disparity_new, 500)

        # measure inside the roi - only for vertical walls
        #ret                 = self.compute_roi_mean_std(img_depth, img_depth_new)

        return img_depth_new    

    #%% -----------------------------------------

    def show_images_left_right(self, imgL = None, imgR = None, ttl = 'Left and Right'):
        "draw left right results"
        if imgL is None:
            imgL, imgR = self.imgL, self.imgR

        if imgL is None or imgR is None:
            log.error('No images found')
            return False
            
        # deal with black and white
        img_show = np.concatenate((imgL, imgR ), axis = 0)

        # --------------------------
        # working ok
        if not self.real_time_on:
        
            plt.figure()
            plt.imshow(img_show, cmap='gray')#, vmin=vmin, vmax=vmax)
            plt.title(ttl)
            #plt.show()   
          
        else:
            # --------------------------
            # real time
            if self.show_info is None:
                plt.ion()  # Turn on interactive mode
                fig, ax = plt.subplots()
                imh = ax.imshow(img_show, cmap='gray')
                plt.title(ttl)
                self.show_info = {"fig":fig,"ax":ax,"imh":imh}
            else:
                self.show_info["imh"].set_data(img_show)
                self.show_info["ax"].set_title(ttl)

            self.show_info["fig"].canvas.draw_idle()
            self.show_info["fig"].canvas.flush_events()

            
        return True
    
    def show_keypoints(self, imgL, kpL, ttl = 'Image and Keypoints'):
        "draw image and detected keypoints"
        if imgL is None:
            imgL = self.imgL

        if imgL is None:
            log.error('No images found')
            return False
            
        if imgL.dtype is not np.uint8:
            imgL = imgL.astype(np.uint8)



        # Draw matches
        img_show = cv.drawKeypoints(imgL, kpL, imgL, color=(0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
      
        # cv.imshow('Image & Keypoints (q-exit)', img_show)
        #log.info('show done')
        # ch = cv.waitKey(5)
        # ret = ch == ord('q')

        plt.figure()
        plt.imshow(img_show, cmap='gray')#, vmin=vmin, vmax=vmax)
        plt.title(ttl)
        plt.show(block = False)          
        return True    

    def show_images_and_keypoints(self, imgL, imgR, kpL, kpR, matches, ttl = 'Left and Right + Matches'):
        "draw left right results"
        if imgL is None:
            imgL, imgR = self.imgL, self.imgR

        if imgL is None or imgR is None:
            log.error('No images found')
            return False
            
        if imgL.dtype is not np.uint8:
            imgL = imgL.astype(np.uint8)
            imgR = imgR.astype(np.uint8)


        # Draw matches
        img_show     = cv.drawMatches(imgL, kpL, imgR, kpR, matches, None, flags=2)        
        
        cv.imshow('Image D-C (q-exit)', img_show)
        #log.info('show done')
        ch = cv.waitKey(5)
        ret = ch == ord('q')

        # plt.figure()
        # plt.imshow(img_show, cmap='gray')#, vmin=vmin, vmax=vmax)
        # plt.title(ttl)
        # plt.show()          
        return ret    

    def show_images_depth(self, imgD = None, do_show = True, fig_num = 1, fig_name = 'Depth Image'):
        "draw results of depth estimation"
        if imgD is not None:
            self.imgD = imgD

        if self.imgD is None and self.imgC is None:
            log.error('No images found')
            return False
        
        elif self.imgD is None: # no data acquired
            img_show = self.imgC

        elif self.imgC is None: # no data is processed
            img_show = self.imgD      
            #img_show = cv.applyColorMap(self.imgD, cv.COLORMAP_TURBO)     

        elif np.all(self.imgD.shape == self.imgC.shape):
            img_show = np.concatenate((self.imgD, self.imgC ), axis = 1)

        # deal with 16 uint    
        if img_show.dtype == 'uint16':
            img_show    = cv.convertScaleAbs(img_show, alpha=0.03)
            img_show    = cv.applyColorMap(img_show, cv.COLORMAP_TURBO) #   
        else:
            #self.imgD = np.repeat(self.imgD[:,:,np.newaxis], 3, axis = 2)
            #img_show = np.concatenate((self.imgD, self.imgC ), axis = 1)
            #img_show = cv.applyColorMap(img_show.astype(np.uint8), cv.COLORMAP_TURBO) 
            #img_show = self.imgC #np.concatenate((self.imgD, self.imgC ), axis = 1)
            pass
            
        if not do_show:
            return img_show

        # deal with black and white
        if img_show.shape[1] > 2400:
            img_show = cv.pyrDown(img_show)
                

        #cv.imshow('Image D-C (q-exit)', img_show)
        #log.info('show done')
        #ch = cv.waitKey(5)
        # ret = ch == ord('q')

        if self.rect is None:
            x0,y0,x1,y1 = 0, 0, img_show.shape[1], img_show.shape[0]
        else:
            x0,y0,x1,y1 = self.rect
        
        vmean       = img_show[y0:y1,x0:x1].mean()
        #vmean       = img_show.mean()
        #vmean       = 1000 #3100 # 1000
        #vmin, vmax  = 100, 1200 #
        vmin, vmax  = vmean - 10, vmean + 10

        # plt.figure(fig_num)
        # plt.imshow(img_show, cmap='gray', vmin=vmin, vmax=vmax) # vmin=img_show.min(), vmax=img_show.max())  # vmin=3000, vmax=4000)
        # plt.title(fig_name)
        # #plt.colorbar()
        # plt.show(block = False)   

        if not self.real_time_on:
        
            plt.figure(fig_num)
            plt.imshow(img_show, cmap='gray', vmin=vmin, vmax=vmax)
            plt.title(fig_name)
            plt.show(block = False)  
            #plt.show()   
          
        else:
            # --------------------------
            # real time
            if self.show_info is None:
                plt.ion()  # Turn on interactive mode
                #fig, ax = plt.subplots(fig_num = fig_num)
                fig = plt.figure(fig_num)
                ax  = fig.add_subplot(1, 1, 1)
                imh = ax.imshow(img_show, cmap='gray')
                plt.title(fig_name)
                self.show_info = {"fig":fig,"ax":ax,"imh":imh}
            else:
                self.show_info["imh"].set_data(img_show)
                self.show_info["imh"].set_clim(vmin, vmax)
                self.show_info["ax"].set_title(fig_name)

            self.show_info["fig"].canvas.draw_idle()
            self.show_info["fig"].canvas.flush_events()        
        
        ret = False
        return ret

    def show_images_depth_3d(self, imgD = None, do_show = True, fig_num = 1, fig_name = 'Depth Image'):
        "draw results of depth estimation"
        if imgD is None:
            log.error('No images found')
            return False
        
        x0,y0,x1,y1 = self.rect
        img_show    = imgD[y0:y1,x0:x1]

        # # deal with black and white
        # img_show = np.uint8(img_show) #.copy()
        # if len(img_show.shape) < 3:
        #     img_show = cv.applyColorMap(img_show, cv.COLORMAP_JET)        

        # cv.imshow('Image D-C (q-exit)', img_show)
        # #log.info('show done')
        # ch = cv.waitKey(50)
        # ret = ch == ord('q')

        vmean       = img_show.mean()
        vmin, vmax  = vmean - 600, vmean + 200

        # remove dead
        #img_show[img_show < 2] = np.nan
        img_show = np.ma.masked_where(img_show < 2, img_show)

        # set up the Axes for the first plot
        fig         = plt.figure(fig_num) #, figsize=plt.figaspect(0.5))
        ax          = fig.add_subplot(1, 1, 1, projection='3d')

        # plot a 3D surface like in the example mplot3d/surface3d_demo
        X           = np.arange(x0,x1)
        Y           = np.arange(y0,y1)
        X, Y        = np.meshgrid(X, Y)
        Z           = img_show

        #surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=True)
        surf        = ax.plot_surface(X, Y, Z,  cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_zlim(vmin, vmax)
        #fig.colorbar(surf, shrink=0.5, aspect=10)
        ax.set_xlabel('X pix')
        ax.set_ylabel('Y pix')
        ax.set_zlabel('Z mm')     
        ax.yaxis.set_inverted(True) 
        ax.set_aspect('equal')  
        plt.title(fig_name)
        plt.show(block = False)    
        ret = False

        return ret

    def show_roi_keypoints_3d(self, xyz_roi, xyz_kp = None):
        "display in 3D"
        fig             = plt.figure(31)
        ax              = fig.add_subplot(projection='3d')

        #xs,ys,zs       = img3d[:,:,0].reshape((-1,1)), img3d[:,:,1].reshape((-1,1)), img3d[:,:,2].reshape((-1,1))
        
        xs,ys,zs       = xyz_roi[:,0].reshape((-1,1)), xyz_roi[:,1].reshape((-1,1)), xyz_roi[:,2].reshape((-1,1))
        ax.scatter(xs, ys, zs, marker='.')
        
        xs,ys,zs       = xyz_kp[:,0].reshape((-1,1)), xyz_kp[:,1].reshape((-1,1)), xyz_kp[:,2].reshape((-1,1))
        ax.scatter(xs, ys, zs, marker='o', s=20, label='Keypoints')


        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        ax.set_aspect('equal', 'box')
        plt.title('Depth Converted and Keypoints in 3D')

        max_dim = xyz_kp.max(axis=0) * 1.2  # assuming the board is square
        min_dim = xyz_kp.min(axis=0) * 1.2
        ax.set_xlim([min_dim[0], max_dim[0]]) # Allow space for camera
        ax.set_ylim([min_dim[1], max_dim[1]])
        ax.set_zlim([-10, max_dim[2]]) # Assuming Z is positive into the scene   
        ax.yaxis.set_inverted(True)      
        plt.show(block = False)

    def show_images_depth_3d_rt(self, xyz_roi, xyz_kp = None):
        "draw results of depth estimation in real time"
        if xyz_roi is None:
            log.error('No images found')
            return False
        
        #x0,y0,x1,y1 = self.rect
        #xs,ys,zs       = xyz_roi[:,0].reshape((-1,1)), xyz_roi[:,1].reshape((-1,1)), xyz_roi[:,2].reshape((-1,1))
        #ax.scatter(xs, ys, zs, marker='.')
    
        # plot a 3D surface like in the example mplot3d/surface3d_demo
        xs,ys,zs    = xyz_kp[:,0].reshape((-1,1)), xyz_kp[:,1].reshape((-1,1)), xyz_kp[:,2].reshape((-1,1))
       

        # set up the Axes for the first plot
        if self.show_info is None:
            fig         = plt.figure(41) #, figsize=plt.figaspect(0.5))
            ax          = fig.add_subplot(1, 1, 1, projection='3d')

            points      = ax.scatter(xs, ys, zs, marker='o', s=20, label='Keypoints')    

            #fig.colorbar(surf, shrink=0.5, aspect=10)
            ax.set_xlabel('X pix')
            ax.set_ylabel('Y pix')
            ax.set_zlabel('Z mm')     
            ax.yaxis.set_inverted(True) 
            ax.set_aspect('equal')  

            max_dim = xyz_kp.max(axis=0) * 1.5  # assuming the board is square
            min_dim = xyz_kp.min(axis=0) * 1.5
            ax.set_xlim([min_dim[0], max_dim[0]]) # Allow space for camera
            ax.set_ylim([min_dim[1], max_dim[1]])
            ax.set_zlim([-10, max_dim[2]]) # Assuming Z is positive into the scene   

            plt.title('Keypoints in 3D')            

            self.show_info = {"fig":fig,"ax":ax,"points":points}
            
        else:
            # 4. Define the update function
            # Create the new vertex array (reshape to (N, 3))
            #self.show_info['points'].set_xdata(xs)
            #self.show_info['points'].set_ydata(ys)
            #self.show_info['points'].set_3d_properties(zs)
            self.show_info['points']._offsets3d = (xs.ravel(), ys.ravel(), zs.ravel())
            #self.show_info['points']._offsets3d = (xs.T, ys.T, zs.T)
            #self.show_dict['ax'].set_ylim(low_limit, high_limit)      


        self.show_info['fig'].canvas.draw_idle()
        self.show_info['fig'].canvas.flush_events()

        ret = False

        return ret

    def show_disparity_plot(self, r=0,c=0):
        "plots disparity line for a specifc pixel"
        if self.search_pmm is None:
            return
        
        plot_num = self.search_pmm.shape[0]
        plt.figure(figsize=(12, 4))

        # for k in range(plot_num):
        #     plt.subplot(plot_num, 1, k+1)
        #     plt.plot(self.search_pmm[k,:], label=str(k))
        #     plt.title('Disparity Line %d' %k)

        plt.plot(self.search_pmm.T,'-o')            
        plt.legend({str(k) for k in range(plot_num)})
        plt.tight_layout()
        plt.title('Disparity Line row: %d, col: %d' %(r,c))
        #plt.show()          

    def show_noise_data(self, vis):
        "show noise related data"

        # init
        if self.noise_rs is None or self.noise_fft is None:
            return

        # do
        vis            = self.noise_rs.show_time_line(vis,'b')
        vis            = self.noise_fft.show_time_line(vis, 'r')

        return vis  

    def show_subset(self, img_list, ttl_list, vmin=None, vmax=None, save_path='', fig_name='', col_num=3):
        "show some images"
        if not self.debug_show: 
            return

        img_num  = len(img_list)
        row_num  = int(img_num/col_num) 
        col_num  = int(np.ceil(img_num/row_num))
        fig, axes = plt.subplots(row_num, col_num, sharey=True, sharex=True)
        axes      = axes.reshape((row_num,col_num))
        do_save   = os.path.exists(save_path)
        for k in range(img_num):
            ri, ci = int(k / col_num), k % col_num
            pcm = axes[ri, ci].imshow(img_list[k], vmin=vmin, vmax=vmax)
            axes[ri, ci].set_title(ttl_list[k])     
            #fig.colorbar(pcm, ax=axes[ri, ci])  
        
        if do_save:
            fig.savefig(os.path.join(save_path, fig_name + ".png"))
        
        plt.show(block=False)
        #plt.show()

    def show_outputs(self, params : list[dict], bank: list[dict], responses: list[np.ndarray], cols: int = 4, ttl: str = "Gabor Filter Bank Decomposition") -> None:
        total = len(bank)
        rows = int(np.ceil(total / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), sharey=True, sharex=True)
        axes = np.array(axes).reshape(rows, cols)

        # axes[0, 0].imshow(image_gray, cmap="gray")
        # axes[0, 0].set_title("Original", fontsize=10)
        # axes[0, 0].axis("off")

        for i in range(total):  # (item, response) in enumerate(zip(bank, responses), start=0):
            item, response = params[i], responses[:,:,i]
            r = i // cols
            c = i % cols
            axes[r, c].imshow(response, cmap="gray")
            axes[r, c].set_title(
                f"θ={item['theta'] / np.pi:.2f}π, λ={item['lambda']:.1f}, ψ={item['psi']:.2f}",
                fontsize=9,
            )
            axes[r, c].axis("off")

        for i in range(total, rows * cols):
            r = i // cols
            c = i % cols
            axes[r, c].axis("off")

        fig.suptitle(ttl, fontsize=14)
        fig.tight_layout()

    def show_kernels(self, params : list[dict], bank: list[dict], cols: int = 4, ttl: str = "Gabor Kernels") -> None:
        total = len(bank)
        rows = int(np.ceil(total / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
        axes = np.array(axes).reshape(rows, cols)

        for i in range(total): # item in enumerate(bank):
            item = params[i]
            kernel = bank[i]
            r = i // cols
            c = i % cols
            axes[r, c].imshow(kernel, cmap="gray")
            axes[r, c].set_title(
                f"θ={item['theta'] / np.pi:.2f}π, λ={item['lambda']:.1f}, ψ={item['psi']:.2f}",
                fontsize=9,
            )
            axes[r, c].axis("off")

        for i in range(total, rows * cols):
            r = i // cols
            c = i % cols
            axes[r, c].axis("off")

        fig.suptitle(ttl, fontsize=14)
        fig.tight_layout()        

    def show_disparity_per_pixel(self, img_array, rc_list, ttl_list = []):
        "show all the disparity values per specific pixel"
        min_disp, max_disp  = 1, img_array.shape[2]
        x_disp = np.arange(min_disp, max_disp)
        ttl_list = ttl_list if len(ttl_list) == len(rc_list) else [f'{i}' for i in range(len(rc_list))]
        plt.figure(figsize=(8,4))
        for rc, ttl in zip(rc_list, ttl_list):
            r, c = rc
            disp_line = img_array[r,c,:].squeeze()
            plt.plot(x_disp, disp_line[min_disp:max_disp], label = f'Pixel {rc} - {ttl}')
        plt.legend()
        plt.title('Disparity values per pixel')
        plt.show(block=False)
        return True

# ----------------------
#%% Tests
class TestShazamDepthEstimator():

    def __init__(self):
        "init test"
        log.info('ShazamDepthEstimator tests started')

    def assertTrue(self, isOk = True):
        "assert true"
        if not isOk:
            raise AssertionError("Test failed")

    def assertFalse(self, isOk = False):
        "assert false"
        if isOk:
            raise AssertionError("Test failed")

    def test_show_images_left_right(self):
        "left right test"
        p = DataSource() #DepthEstimator()
        p.init_image(11)
        p.show_images_left_right()
        plt.show()
        self.assertFalse(p.imgD is None)

    def test_show_images_depth(self):
        "depth show"
        p = DataSource()
        p.init_image(11)
        p.show_images_depth()
        cv.waitKey()
        self.assertFalse(p.imgD is None)  

    def test_convert_disparity_to_depth(self):
        "try to see if there are steps in the function"
        p                   = ShazamDepthEstimator()
        factor              = 1
        depth_orig          = np.linspace(3000,4000,100)
        disparity_orig      = p.convert_depth_to_disparity(depth_orig, 121) 
        disparity_scale     = disparity_orig*factor
        #disparity           = disparity_scale.astype(np.uint16) #np.round(disparity,3)
        disparity           = np.round(disparity_scale,0) 
        disparity_quant     = disparity.astype(np.float32)
        disparity_quant     = disparity_quant/factor
        depth               = p.convert_disparity_to_depth(disparity_quant, 121)

        plt.figure(121)
        plt.plot(depth_orig, depth - depth_orig,'-ob',depth_orig, disparity_orig*10,'g', depth_orig, disparity_quant*10,'r')
        plt.legend(['depth error','disparity orig*10','disparity quant*10'])
        plt.title('Error versus depth')
        plt.show()

    def test_grid_interpolation(self):
        "verify 3D grid interpolation shape and values for a linear field"
        nx, ny, nz = 4, 5, 6
        factor = 2

        x = np.linspace(0.0, 1.0, nx)
        y = np.linspace(0.0, 1.0, ny)
        z = np.linspace(0.0, 1.0, nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # Linear field: trilinear interpolation should be exact up to fp tolerance.
        data = X + 2.0 * Y + 3.0 * Z

        interp = grid_interpolation(data, factor=factor)

        self.assertTrue(interp is not None)
        self.assertTrue(interp.shape == (nx * factor, ny * factor, nz * factor))

        # Validate exactness against analytic ground truth on the denser grid.
        xi = np.linspace(0.0, 1.0, nx * factor)
        yi = np.linspace(0.0, 1.0, ny * factor)
        zi = np.linspace(0.0, 1.0, nz * factor)
        Xi, Yi, Zi = np.meshgrid(xi, yi, zi, indexing='ij')
        expected = Xi + 2.0 * Yi + 3.0 * Zi

        # 3D visualization of source data vs interpolated result.
        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        sc1 = ax1.scatter(
            X.ravel(), Y.ravel(), Z.ravel(),
            c=data.ravel(), cmap='viridis', s=45
        )
        ax1.set_title('Input 3D Grid Values')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')
        fig.colorbar(sc1, ax=ax1, shrink=0.75, label='value')

        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        sc2 = ax2.scatter(
            Xi.ravel(), Yi.ravel(), Zi.ravel(),
            c=interp.ravel(), cmap='viridis', s=10
        )
        ax2.set_title('Interpolated 3D Grid Values')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('z')
        fig.colorbar(sc2, ax=ax2, shrink=0.75, label='value')

        plt.tight_layout()
        plt.show()

        self.assertTrue(np.allclose(interp, expected, atol=1e-6))


    #%% ---------------------------------------------------

    def test_gabor_bank_channels(self):
        "test Gabor filter bank multi-channel output with visualization of all responses"
        d       = DataSource()
        ret     = d.init_image(11) # 91-ok
        self.assertTrue(ret)

        p       = ShazamDepthEstimator()
        
        # Gabor parameters
        thetas = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        lambdas = [8.0, 16.0, 32.0]
        psis = [0.0, np.pi / 2]
        ksize = 21
        sigma = 4.0
        gamma = 0.5
        
        # Apply Gabor bank to left image
        output = p.gabor_bank_channels(
            d.imgL,
            ksize=ksize,
            sigma=sigma,
            lambdas=lambdas,
            gamma=gamma,
            psis=psis,
            thetas=thetas
        )
        
        # Validate output shape
        h, w = d.imgL.shape[:2]
        expected_channels = len(thetas) * len(lambdas) * len(psis)
        #self.assertEqual(output.shape, (h, w, expected_channels))
        #self.assertEqual(output.dtype, np.uint8)
        
        # Create visualization grid showing all channels
        fig, axes = plt.subplots(
            len(thetas), 
            len(lambdas) * len(psis), 
            figsize=(15, 12),
            sharex=True,
            sharey=True
        )
        
        ch_idx = 0
        for t_idx, theta in enumerate(thetas):
            for l_idx, lambd in enumerate(lambdas):
                for p_idx, psi in enumerate(psis):
                    row = t_idx
                    col = l_idx * len(psis) + p_idx
                    
                    axes[row, col].imshow(output[:, :, ch_idx], cmap='hot')
                    
                    theta_deg = np.degrees(theta)
                    psi_deg = np.degrees(psi)
                    title = f'θ={theta_deg:.0f}°\nλ={lambd:.0f}, ψ={psi_deg:.0f}°'
                    axes[row, col].set_title(title, fontsize=9)
                    axes[row, col].axis('off')
                    
                    ch_idx += 1
        
        plt.suptitle(f'Gabor Filter Bank Response Channels (Total: {expected_channels} filters)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Print channel information
        print(f"\nGabor Filter Bank Configuration:")
        print(f"  Image shape: {d.imgL.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Orientations (θ): {len(thetas)} ({', '.join([f'{np.degrees(t):.0f}°' for t in thetas])})")
        print(f"  Wavelengths (λ): {len(lambdas)} ({', '.join([str(int(l)) for l in lambdas])})")
        print(f"  Phase offsets (ψ): {len(psis)} ({', '.join([f'{np.degrees(p):.0f}°' for p in psis])})")
        print(f"  Total channels: {expected_channels}")
        print(f"  Kernel size: {ksize}, σ={sigma}, γ={gamma}")

    def test_gabor_line_correlation(self):
        "compute row-wise left/right gabor channel inner products and show MxM matrix"
        d = DataSource()
        ret = d.init_image(11)
        self.assertTrue(ret)
        d.show_images_left_right()

        p = ShazamDepthEstimator()

        # Gabor bank parameters.
        thetas = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        lambdas = [8.0, 16.0, 32.0]
        psis = [0.0, np.pi / 2]
        ksize = 21
        sigma = 4.0
        gamma = 0.5

        # Build NxMxC responses for both images.
        gabor_L = p.gabor_bank_channels(
            d.imgL,
            ksize=ksize,
            sigma=sigma,
            lambdas=lambdas,
            gamma=gamma,
            psis=psis,
            thetas=thetas,
        )
        gabor_R = p.gabor_bank_channels(
            d.imgR,
            ksize=ksize,
            sigma=sigma,
            lambdas=lambdas,
            gamma=gamma,
            psis=psis,
            thetas=thetas,
        )

        self.assertTrue(gabor_L.ndim == 3)
        self.assertTrue(gabor_R.ndim == 3)
        self.assertTrue(gabor_L.shape == gabor_R.shape)

        n, m, c = gabor_L.shape
        self.assertTrue(m > 0 and c > 0)

        # Select row K (middle row by default).
        k = 200 #n // 2

        # Row feature matrices: (M, C).
        rowL = gabor_L[k, :, :].astype(np.float32)
        rowR = gabor_R[k, :, :].astype(np.float32)

        # Inner products across channels for all xL, xR pairs: (M, M).
        corr = rowL @ rowR.T

        # Absolute difference across channels for all xL, xR pairs: (M, M).
        # A: (N, C) and B: (M, C)
        A_expanded = rowL[:, np.newaxis, :]  # Shape (N, 1, C)
        B_expanded = rowR[np.newaxis, :, :]  # Shape (1, M, C)

        # Subtracting (N, 1, C) and (1, M, C) results in (N, M, C)
        # Then we sum along the last axis (C)
        corr = np.abs(A_expanded - B_expanded).sum(axis=2)               

        self.assertTrue(corr.shape == (m, m))

        # Plot correlation matrix.
        plt.figure(figsize=(8, 7))
        plt.imshow(corr, cmap='viridis', aspect='auto')
        plt.colorbar(label='Inner product over C channels')
        plt.xlabel('Right image column index')
        plt.ylabel('Left image column index')
        plt.title(f'Gabor row correlation matrix at row K={k} (size {m}x{m})')
        plt.tight_layout()
        plt.show()

        print(f"Gabor row correlation: N={n}, M={m}, C={c}, row K={k}, matrix shape={corr.shape}")
        
    def test_gabor_line_correlation_multiscale(self):
        "compute row-wise left/right gabor channel inner products and show MxM matrix"
        
        d               = DataSource()
        ret             = d.init_image(622) # 4,7,9,11,54,55,56-ok, ,62,66-nok, 622
        self.assertTrue(ret)
        #d.show_images_left_right()

        level_num       = 3
        max_disparity   = 128
        row_index       = 400 #128 #128 # Select row K (middle row by default).
        kernel_size     = 9
        p               = ShazamDepthEstimator()
        prob_weights    = 2**np.arange(0,level_num)  # e.g. for 4 levels: [8, 4, 2, 1]

        col_num        = d.imgL.shape[1]
        confidence_array = np.zeros((level_num, col_num), dtype=np.float32)

        # Gabor bank parameters.
        thetas          = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        lambdas         = [8.0, 16.0, 32.0];  psis            = [0.0, np.pi / 2];  ksize           = 7; sigma           = 4.0;      gamma = 0.5
        bank, params    = p.gabor_bank_init(ksize=ksize, sigma=sigma, lambdas=lambdas, gamma=gamma, psis=psis, thetas=thetas)

        # Per-channel weights of shape (C, 1); default = uniform 1/C (equivalent to mean).
        channel_num     = len(bank) 
        channel_weights = (np.ones((channel_num,), dtype=np.float32) / float(channel_num)).reshape(channel_num, 1)

        prob            = np.zeros((max_disparity,d.imgL.shape[1],level_num), dtype=np.float32)
        info            = np.zeros((1,d.imgL.shape[1],level_num), dtype=np.float32)
        img_left, img_right = d.imgL, d.imgR
        for k in range(level_num):
            # Build NxMxC responses for both images.
            gaborL              = p.gabor_bank_filter(img_left, bank=bank)
            gaborR              = p.gabor_bank_filter(img_right, bank=bank)

            # compute maximum variability/spatial information content for the left channel using spatial filter
            # This can be done by computing the local variance or entropy of the responses across channels for each pixel. Pixels with higher variability are more likely to provide reliable matches, so we can use this information to weight the correlation scores.

            gaborL_aver         = cv.boxFilter(gaborL, -1,   (kernel_size, kernel_size), normalize=True) 
            gaborL_diff         = gaborL - gaborL_aver
            variabilityL        = cv.boxFilter(gaborL_diff**2, -1,   (kernel_size, kernel_size), normalize=True)  
            informationL        = np.mean(variabilityL, axis=2)  # shape (N, M)
            # spatial averaging can be applied to the variability map to smooth it and reduce noise. This can be done using a Gaussian filter or a simple moving average filter, which can help improve the reliability of the variability estimates.
            # variabilityL        = cv.GaussianBlur(variabilityL, (5, 5), 0)

            # Row feature matrices: (M, C).
            rowL                = gaborL[row_index, :, :].astype(np.float32)
            rowR                = gaborR[row_index, :, :].astype(np.float32)
            infoL               = informationL[row_index, :] #.reshape(-1, 1)  # shape (M, 1)

            # clip noise - small values that are likely noise can be set to zero to reduce their influence on the correlation. This can be done by setting a threshold below which values are considered noise and set to zero.
            # noise_threshold     = 5.0
            # rowL[np.abs(rowL) < noise_threshold] = 0
            # rowR[np.abs(rowR) < noise_threshold] = 0

            # compute energy / norm
            normL               = np.linalg.norm(rowL, axis=1) 
            normR               = np.linalg.norm(rowR, axis=1) 

            # normize rows to unit vectors to focus on direction rather than magnitude. This can help improve the correlation by making it more about the pattern of responses across channels rather than their absolute strength.
            rowL                = rowL / np.sqrt(normL[:, np.newaxis] + 1e-6)  # shape (M, C)
            rowR                = rowR / np.sqrt(normR[:, np.newaxis] + 1e-6)  # shape (M, C)

            # Inner products across channels for all xL, xR pairs: (M, M).
            #corr                = rowL @ rowR.T

            # Absolute difference across channels for all xL, xR pairs: (M, M).
            # A: (N, C) and B: (M, C)
            expandedR           = rowR[:, np.newaxis, :]  # Shape (N, 1, C)
            expandedL           = rowL[np.newaxis, :, :]  # Shape (1, M, C)

            # Subtracting (N, 1, C) and (1, M, C) results in (N, M, C).
            # Then take a weighted sum along the last axis (C) using channel_weights of shape (C, 1).
            dist                 = (np.abs(expandedL - expandedR) @ channel_weights)[:, :, 0]  # shape (N, M)
            #corr_disp            = extract_diag_band_fast(corr, B=max_disparity) 
            dist_disp            = extract_diag_band(dist, B=max_disparity)  # clip negative values to zero

            # # 1. Error Gate (Closer to 1 when dist is small)
            # s_error             = np.exp(-np.square(dist / alpha))
            
            # # 2. Norm Gate (Sigmoid: Closer to 1 when average norm > tau)
            # avg_norm            = 0.5 * (norm_u + norm_v)
            # s_norm              = 1.0 / (1.0 + np.exp(-sigma * (avg_norm - tau)))
            # s_error * s_norm  # Combined similarity score that considers both error and norm
            
            #corr                 = np.exp(-corr/0.1)  # (N, M) similarity matrix with exponential decay
            corr                 = softmax_columns(-dist, dim=0, T=0.05) # (N, M) similarity matrix with exponential decay
            corr_disp            = softmax_columns(-dist_disp, dim=0, T=0.05)  # normalize to [0, 1], higher is more similar
             # keep only values close to the diagonal, which correspond to similar column indices in left and right images. This can help focus on likely matches and reduce noise from unrelated pairs.
            #corr_disp            = corr_disp/np.sum(corr_disp, axis=0, keepdims=True)  # normalize to [0, 1] for display
            
            row_num_current, col_num_current   = corr_disp.shape
            # remove no response columns and rows
            #corr[normL < 1e-1, :] = 0
            #corr[:, normR < 1e-1] = 0            

            # maxima suppression over matrix corr. For each column, find the maximum value and count how many values are above a certain percentage of that maximum. This gives a measure of how many strong matches there are for each column, which can be used as a confidence score.
            top_max_ind             = np.argmax(corr_disp, axis=0)  # shape (M,)
            # suppress the max index and close one pixel around it to avoid counting very close matches
            top_max_low_ind         = np.maximum(0, top_max_ind-1)
            top_max_high_ind        = np.minimum(row_num_current-1, top_max_ind+1) 
            corr2                   = corr_disp.copy()
            top_max_values          = corr_disp[top_max_ind, np.arange(col_num_current)]  # shape (M,)
            corr2[top_max_ind, np.arange(col_num_current)] = 0  # zero out the max values
            corr2[top_max_low_ind, np.arange(col_num_current)] = 0  # zero out the low neighbors
            corr2[top_max_high_ind, np.arange(col_num_current)] = 0  #


            # compute confidence by averaging over 95% of maximum for each column of the correlation matrix
            # second_max_values       = np.max(corr2, axis=0)  # shape (M,)
            # confidence              = np.zeros_like(top_max_values)
            # valid_mask             = top_max_values > 0.1
            # #confidence[valid_mask] = 1 - (second_max_values[valid_mask] / top_max_values[valid_mask] )  # shape (M,), higher confidence if second max is much lower than top max  
            # confidence[valid_mask] = 1 - np.exp(-(top_max_values[valid_mask]  - second_max_values[valid_mask]) / 1)  
            confidence              = top_max_values
            
            #pixels_above_threshold  = ((corr >= top_max_values*top_percentile) & (corr > 0.4)).sum(axis=0)  # shape (M,)
            # make confidence a matrix of the same shape as corr by repeating the confidence values for each row
            # confidence              = pixels_above_threshold #np.exp(-(pixels_above_threshold-1)/8.0)  # shape (1, M)
            #confidence_mtrx         = np.repeat(confidence[np.newaxis, :], corr.shape[0], axis=0)  # shape (N, M)
            #prob_current            = cv.resize(confidence_mtrx, (prob.shape[1], prob.shape[0]))  # shape (N, M)
            prob_current             = cv.resize(corr_disp, (prob.shape[1], prob.shape[0]))  #
            prob[:,:,k]              = prob_current
            interpolated_values      = np.interp(np.arange(col_num), np.linspace(0, col_num, col_num_current), infoL)
            info[:,:,k]              = interpolated_values
            # normalize to [0, 1]
            prob_total                = np.sum(prob*info, axis=2, keepdims=True) / (np.sum(info, axis=2, keepdims=True) + 1e-6)  # shape (N, M)

            #corr                    = corr * confidence_mtrx  # shape (N, M)
            #corr[:,confidence < 0.3] = 0  # zero out low confidence matches



            # show
            img_show = np.concatenate((img_left, img_right ), axis = 0)
            img_show[[row_index,row_index+img_left.shape[0]],:] = 255  # highlight the row in both images
            plt.figure(figsize=(8, 10))
            plt.imshow(img_show, cmap='gray')
            plt.title(f'Image L-R : Level {k}, row {row_index}')              

            # show image rows
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1) 
            plt.plot(rowL, '-.')
            plt.title(f'Gabor row features: Left image, Level {k}, row {row_index}')
            plt.subplot(1, 2, 2)        
            plt.plot(rowR, '-.')
            plt.title(f'Gabor row features: Right image, Level {k}, row {row_index}')
            plt.tight_layout()

            # show confidence
            plt.figure(figsize=(8, 4))
            plt.plot(confidence, '-o')
            plt.title(f'Confidence: Level {k}, row {row_index}')
            plt.xlabel('Left image column index')
            plt.ylabel('Confidence')
            plt.tight_layout()

            # show information
            plt.figure(figsize=(8, 4))
            plt.plot(infoL, '-o')
            plt.title(f'Information: Level {k}, row {row_index}')
            plt.xlabel('Left image column index')
            plt.ylabel('Information')
            plt.tight_layout()            

            # # Plot correlation matrix.
            # plt.figure(figsize=(8, 7))
            # plt.imshow(corr, cmap='viridis', aspect='auto')
            # plt.colorbar(label='Inner product over C channels')
            # plt.xlabel('Left image column index')
            # plt.ylabel('Right image column index')
            # plt.title(f'Gabor row correlation matrix at row Level {k}, row {row_index}')
            # plt.tight_layout()
            # plt.show(block=False)

            plt.figure(figsize=(8, 4))
            plt.imshow(corr_disp, cmap='viridis', aspect='auto')
            plt.colorbar(label='Inner product over C channels')
            plt.xlabel('Left image column index')
            plt.ylabel('Right image column index')
            plt.title(f'Gabor row correlation matrix (diagonal band) at row Level {k}, row {row_index}')
            plt.tight_layout()
            plt.show(block=False)

            # show image rows
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1) 
            plt.imshow(prob_current, cmap='viridis', aspect='auto')
            plt.colorbar(label='Probability current level')
            plt.title(f'Probability current level , Level {k}, row {row_index}')
            plt.subplot(1, 2, 2)        
            plt.imshow(prob_total, cmap='viridis', aspect='auto')
            plt.colorbar(label='Probability accumulated')
            plt.title(f'Probability accumulated , Level {k}, row {row_index}')
            plt.tight_layout()            

            # downscale
            img_left  = cv.pyrDown(img_left)
            img_right = cv.pyrDown(img_right)
            row_index = row_index // 2
            max_disparity = max_disparity // 2
            #prob      = prob + prob_current*prob_weights[k]
            #prob       = np.maximum(prob , prob_current)
            #prob      = (prob*k + prob_current*(level_num-k))/level_num  
            #prob       = prob * prob_current
            # prob       = softmax_columns(prob + prob_current,dim=0,T=0.01)
            # prob      = prob + prob_current
            # prob      = prob/np.max(prob, axis=0, keepdims=True)


            # save confidence for this level - interpolate to original size
            interpolated_values = np.interp(np.arange(col_num), np.linspace(0, col_num, col_num_current), confidence)
            confidence_array[k,:] = interpolated_values


        # Plot correlation matrix.
        #prob = prob / prob_weights.sum()  # average over levels
        plt.figure(figsize=(8, 7))
        plt.imshow(prob, cmap='viridis', aspect='auto')
        plt.colorbar(label='Probability of match (sum of levels)')
        plt.xlabel('Right image column index')
        plt.ylabel('Left image column index')
        plt.title(f'probability')
        plt.tight_layout()
        plt.show(block=False)

        # Plot confidence across columns.
        plt.figure(figsize=(8, 4))
        plt.plot(confidence_array.T, '-')
        plt.title('Confidence across columns for each level')
        plt.xlabel('Pixel column index')
        plt.ylabel('Confidence')
        plt.legend([f'Level {k}' for k in range(level_num)], loc='upper right')
        plt.tight_layout()


        plt.show()
        return True
        
    def test_gabor_multiline_correlation_multiscale(self):
        "compute row-wise left/right gabor channel inner products and show MxM matrix"
        
        d               = DataSource()
        ret             = d.init_image(622) # 4,7,9,11,54,55,56-ok, ,62,66-nok, 601-ok, 622-ok
        self.assertTrue(ret)
        #d.show_images_left_right()

        level_num       = 3
        max_disparity   = 128
        row_indexes     = [400,404] #128 #128 # Select row K (middle row by default).
        kernel_size     = 9
        row_num         = len(row_indexes)

        p               = ShazamDepthEstimator()
        prob_weights    = 2**np.arange(0,level_num)  # e.g. for 4 levels: [8, 4, 2, 1]

        col_num        = d.imgL.shape[1]
        confidence_array = np.zeros((level_num, col_num), dtype=np.float32)

        # Gabor bank parameters.
        thetas          = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        lambdas         = [8.0, 16.0, 32.0];  psis            = [0.0, np.pi / 2];  ksize           = 13; sigma           = 4.0;      gamma = 0.5
        bank, params    = p.gabor_bank_init(ksize=ksize, sigma=sigma, lambdas=lambdas, gamma=gamma, psis=psis, thetas=thetas)

        # Per-channel weights of shape (C, 1); default = uniform 1/C (equivalent to mean).
        channel_num     = len(bank) 
        channel_weights = (np.ones((channel_num,), dtype=np.float32) / float(channel_num)).reshape(channel_num, 1)

        # array of T
        T_weights       = [0.05, 0.05, 0.05] # for softmax temperature at each level

        prob            = np.zeros((max_disparity,d.imgL.shape[1],level_num, row_num), dtype=np.float32)
        info            = np.zeros((1,d.imgL.shape[1],level_num, row_num), dtype=np.float32)
        img_left, img_right = d.imgL, d.imgR
        for k in range(level_num):
            # Build NxMxC responses for both images.
            gabor_left               = p.gabor_bank_filter(img_left, bank=bank)
            gabor_right              = p.gabor_bank_filter(img_right, bank=bank)

            # normalize the responses across channels for each pixel to have zero mean but not variance. 
            # This can help improve the correlation by making it more about the pattern of responses across channels rather than their absolute strength.  
            gabor_left_norm          = np.sqrt(np.linalg.norm(gabor_left, axis=2, keepdims=True)) + 1e-6
            gabor_left               = gabor_left / gabor_left_norm
            gabor_right_norm         = np.sqrt(np.linalg.norm(gabor_right, axis=2, keepdims=True)) + 1e-6
            gabor_right              = gabor_right / gabor_right_norm

            # # information spatial content for the left image, which can be used to weight the correlation scores. This can help identify pixels that are more likely to provide reliable matches, so we can use this information to weight the correlation scores.
            # info_left                = self.gabor_decomposition_variability(gabor_left, kernel_size=kernel_size)             

            # row_num, col_num    = img_left.shape[:2]
            # confidence_array    = np.zeros((row_num, col_num), dtype=np.float32)

            # show
            img_show                = np.concatenate((img_left, img_right ), axis = 0)            

            for i, row_i in enumerate(row_indexes): #range(0, d.imgL.shape[0], 16): # loop over rows with a step of 16

                row_index               = int(row_i/2**k)

                # Row feature matrices: (M, C).
                rowL                    = gabor_left[row_index, :, :].astype(np.float32)
                rowR                    = gabor_right[row_index, :, :].astype(np.float32)

            # compute maximum variability/spatial information content for the left channel using spatial filter
                # This can be done by computing the local variance or entropy of the responses across channels for each pixel. Pixels with higher variability are more likely to provide reliable matches, so we can use this information to weight the correlation scores.

                gabor_left_aver         = cv.boxFilter(gabor_left, -1,   (kernel_size, kernel_size), normalize=True) 
                gabor_left_diff         = gabor_left - gabor_left_aver
                variability_left        = cv.boxFilter(gabor_left_diff**2, -1,   (kernel_size, kernel_size), normalize=True)  
                information_left        = np.mean(variability_left, axis=2) / 32 # shape (N, M)
                # spatial averaging can be applied to the variability map to smooth it and reduce noise. This can be done using a Gaussian filter or a simple moving average filter, which can help improve the reliability of the variability estimates.
                # variabilityL        = cv.GaussianBlur(variabilityL, (5, 5), 0)

                # Row feature matrices: (M, C).
                rowL                = gabor_left[row_index, :, :].astype(np.float32)
                rowR                = gabor_right[row_index, :, :].astype(np.float32)
                infoL               = information_left[row_index, :] #.reshape(-1, 1)  # shape (M, 1)

                # clip noise - small values that are likely noise can be set to zero to reduce their influence on the correlation. This can be done by setting a threshold below which values are considered noise and set to zero.
                # noise_threshold     = 5.0
                # rowL[np.abs(rowL) < noise_threshold] = 0
                # rowR[np.abs(rowR) < noise_threshold] = 0

                # compute energy / norm
                normL               = np.linalg.norm(rowL, axis=1) 
                normR               = np.linalg.norm(rowR, axis=1) 

                # normize rows to unit vectors to focus on direction rather than magnitude. This can help improve the correlation by making it more about the pattern of responses across channels rather than their absolute strength.
                rowL                = rowL / np.sqrt(normL[:, np.newaxis] + 1e-6)  # shape (M, C)
                rowR                = rowR / np.sqrt(normR[:, np.newaxis] + 1e-6)  # shape (M, C)

                # Inner products across channels for all xL, xR pairs: (M, M).
                #corr                = rowL @ rowR.T

                # Absolute difference across channels for all xL, xR pairs: (M, M).
                # A: (N, C) and B: (M, C)
                expandedR           = rowR[:, np.newaxis, :]  # Shape (N, 1, C)
                expandedL           = rowL[np.newaxis, :, :]  # Shape (1, M, C)

                # Subtracting (N, 1, C) and (1, M, C) results in (N, M, C).
                # Then take a weighted sum along the last axis (C) using channel_weights of shape (C, 1).
                dist                 = (np.abs(expandedL - expandedR) @ channel_weights)[:, :, 0]  # shape (N, M)
                #corr_disp            = extract_diag_band_fast(corr, B=max_disparity) 
                dist_disp            = extract_diag_band(dist, B=max_disparity)  # clip negative values to zero

                # # 1. Error Gate (Closer to 1 when dist is small)
                # s_error             = np.exp(-np.square(dist / alpha))
                
                # # 2. Norm Gate (Sigmoid: Closer to 1 when average norm > tau)
                # avg_norm            = 0.5 * (norm_u + norm_v)
                # s_norm              = 1.0 / (1.0 + np.exp(-sigma * (avg_norm - tau)))
                # s_error * s_norm  # Combined similarity score that considers both error and norm
                
                #corr                 = np.exp(-corr/0.1)  # (N, M) similarity matrix with exponential decay
                #corr                 = softmax_columns(-dist, dim=0, T=0.05) # (N, M) similarity matrix with exponential decay
                corr_disp            = softmax_columns(-dist_disp, dim=0, T=T_weights[k])  # normalize to [0, 1], higher is more similar
                # keep only values close to the diagonal, which correspond to similar column indices in left and right images. This can help focus on likely matches and reduce noise from unrelated pairs.
                #corr_disp            = corr_disp/np.sum(corr_disp, axis=0, keepdims=True)  # normalize to [0, 1] for display
                
                row_num_current, col_num_current   = corr_disp.shape
                # remove no response columns and rows
                #corr[normL < 1e-1, :] = 0
                #corr[:, normR < 1e-1] = 0            

                # maxima suppression over matrix corr. For each column, find the maximum value and count how many values are above a certain percentage of that maximum. This gives a measure of how many strong matches there are for each column, which can be used as a confidence score.
                top_max_ind             = np.argmax(corr_disp, axis=0)  # shape (M,)
                # suppress the max index and close one pixel around it to avoid counting very close matches
                top_max_low_ind         = np.maximum(0, top_max_ind-1)
                top_max_high_ind        = np.minimum(row_num_current-1, top_max_ind+1) 
                corr2                   = corr_disp.copy()
                top_max_values          = corr_disp[top_max_ind, np.arange(col_num_current)]  # shape (M,)
                corr2[top_max_ind, np.arange(col_num_current)] = 0  # zero out the max values
                corr2[top_max_low_ind, np.arange(col_num_current)] = 0  # zero out the low neighbors
                corr2[top_max_high_ind, np.arange(col_num_current)] = 0  #


                # compute confidence by averaging over 95% of maximum for each column of the correlation matrix
                # second_max_values       = np.max(corr2, axis=0)  # shape (M,)
                # confidence              = np.zeros_like(top_max_values)
                # valid_mask             = top_max_values > 0.1
                # #confidence[valid_mask] = 1 - (second_max_values[valid_mask] / top_max_values[valid_mask] )  # shape (M,), higher confidence if second max is much lower than top max  
                # confidence[valid_mask] = 1 - np.exp(-(top_max_values[valid_mask]  - second_max_values[valid_mask]) / 1)  
                confidence              = top_max_values
                
                #pixels_above_threshold  = ((corr >= top_max_values*top_percentile) & (corr > 0.4)).sum(axis=0)  # shape (M,)
                # make confidence a matrix of the same shape as corr by repeating the confidence values for each row
                # confidence              = pixels_above_threshold #np.exp(-(pixels_above_threshold-1)/8.0)  # shape (1, M)
                #confidence_mtrx         = np.repeat(confidence[np.newaxis, :], corr.shape[0], axis=0)  # shape (N, M)
                #prob_current            = cv.resize(confidence_mtrx, (prob.shape[1], prob.shape[0]))  # shape (N, M)
                #prob_current             = cv.resize(corr_disp, (prob.shape[1], prob.shape[0]))  #
                prob_current             = zoom(corr_disp, zoom=2**k, order=1)  #
                prob_current             = cv.resize(prob_current, (prob.shape[1], prob.shape[0]))  # 
                prob[:,:,k,i]            = prob_current
                interpolated_values      = np.interp(np.arange(col_num), np.linspace(0, col_num, col_num_current), infoL)
                info[:,:,k,i]            = interpolated_values
                # normalize to [0, 1]
                prob_total               = np.sum(prob*info, axis=2, keepdims=True) / (np.sum(info, axis=2, keepdims=True) + 1e-6)  # shape (N, M)



                # # compute confidence by averaging over 95% of maximum for each column of the correlation matrix
                # second_max_values       = np.max(corr2, axis=0)  # shape (M,)
                # confidence              = np.zeros_like(top_max_values)
                # valid_mask             = top_max_values > 0.1
                # confidence[valid_mask] = 1 - (second_max_values[valid_mask] / top_max_values[valid_mask] )  # shape (M,), higher confidence if second max is much lower than top max  
            


                img_show[[row_index,row_index+img_left.shape[0]],:] = 255  # highlight the row in both images
            
                # show image rows
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 2, 1) 
                plt.plot(rowL, '-.')
                plt.title(f'Gabor row features: Left image, Level {k}, row {row_index}')
                plt.subplot(1, 2, 2)        
                plt.plot(rowR, '-.')
                plt.title(f'Gabor row features: Right image, Level {k}, row {row_index}')
                plt.tight_layout()

                # show confidence
                plt.figure(figsize=(8, 4))
                plt.subplot(1, 2, 1) 
                plt.plot(confidence, '-o')
                plt.title(f'Confidence: Level {k}, row {row_index}')
                plt.subplot(1, 2, 2) 
                plt.plot(infoL, '-r')
                plt.title(f'Info: Level {k}, row {row_index}')
                plt.xlabel('Right image column index')
                plt.ylabel('Confidence')
                plt.tight_layout()     

                plt.figure(figsize=(8, 4))
                plt.imshow(dist_disp, cmap='viridis', aspect='auto')
                plt.colorbar(label='Inner difference over C channels')
                plt.xlabel('Right image column index')
                plt.ylabel('Left image column index')
                plt.title(f'Gabor row correlation matrix (diagonal band) at row Level {k}, row {row_index}')
                plt.tight_layout()
                plt.show(block=False)       

                # show image rows
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 2, 1) 
                plt.imshow(prob_current, cmap='viridis', aspect='auto')
                plt.colorbar(label='Probability current level')
                plt.title(f'Probability current level , Level {k}, row {row_index}')
                plt.subplot(1, 2, 2)        
                plt.imshow(prob_total[:, :, 0, i], cmap='viridis', aspect='auto')
                plt.colorbar(label='Probability accumulated')
                plt.title(f'Probability accumulated , Level {k}, row {row_index}')
                plt.tight_layout()     
            
            
            plt.figure(figsize=(8, 8))
            plt.imshow(img_show, cmap='gray')
            plt.title(f'Image L-R : Level {k}')              


            # # Plot correlation matrix.
            # plt.figure(figsize=(8, 7))
            # plt.imshow(corr, cmap='viridis', aspect='auto')
            # plt.colorbar(label='Inner product over C channels')
            # plt.xlabel('Right image column index')
            # plt.ylabel('Left image column index')
            # plt.title(f'Gabor row correlation matrix at row Level {k}, row {row_index}')
            # plt.tight_layout()
            # plt.show(block=False)



            # # show image rows
            # plt.figure(figsize=(12, 4))
            # plt.subplot(1, 2, 1) 
            # plt.imshow(prob_current, cmap='viridis', aspect='auto')
            # plt.colorbar(label='Probability current level')
            # plt.title(f'Probability current level , Level {k}, row {row_index}')
            # plt.subplot(1, 2, 2)        
            # plt.imshow(prob, cmap='viridis', aspect='auto')
            # plt.colorbar(label='Probability accumulated')
            # plt.title(f'Probability accumulated , Level {k}, row {row_index}')
            # plt.tight_layout()            

            # downscale
            #img_left  = cv.pyrDown(img_left)
            #img_right = cv.pyrDown(img_right)
            img_left  = zoom(img_left, zoom=0.5, order=1)
            img_right = zoom(img_right, zoom=0.5, order=1)            
            #row_index = row_index // 2
            max_disparity = max_disparity // 2
            #prob      = prob + prob_current*prob_weights[k]
            #prob       = np.maximum(prob , prob_current)
            #prob      = (prob*k + prob_current*(level_num-k))/level_num  
            #prob       = prob * prob_current
            #prob       = softmax_columns(prob + prob_current,dim=0,T=0.01)
            # prob      = prob + prob_current
            # prob      = prob/np.max(prob, axis=0, keepdims=True)


            # # save confidence for this level - interpolate to original size
            # interpolated_values = np.interp(np.arange(col_num), np.linspace(0, col_num, col_num_current), confidence)
            # confidence_array[k,:] = interpolated_values


        # Plot correlation matrix.
        #prob = prob / prob_weights.sum()  # average over levels
        for i, row_i in enumerate(row_indexes):
            plt.figure(figsize=(8, 7))
            plt.imshow(prob[:, :, :, i], cmap='viridis', aspect='auto')
            plt.colorbar(label='Probability of match (sum of levels)')
            plt.xlabel('Right image column index')
            plt.ylabel('Left image column index')
            plt.title(f'probability {row_i}')
            plt.tight_layout()
            plt.show(block=False)

        # # Plot confidence across columns.
        # plt.figure(figsize=(8, 4))
        # plt.plot(confidence_array.T, '-o')
        # plt.title('Confidence across columns for each level')
        # plt.xlabel('Pixel column index')
        # plt.ylabel('Confidence')
        # plt.legend([f'Level {k}' for k in range(level_num)], loc='upper right')
        # plt.tight_layout()


        plt.show()
        return True

    def test_gabor_bank_rotated(self):
        "compute row-wise left/right gabor channel inner products and show MxM matrix. Tricks gabor to rotate floor pattern according to the camera angle to improve floor disparity estimation"
        
        d               = DataSource()
        ret             = d.init_image(56) # 4,7,9,11,54,55,56-ok, ,62,66-nok
        self.assertTrue(ret)
        #d.show_images_left_right()

        p               = ShazamDepthEstimator()

        # Gabor bank parameters.
        thetas          = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        lambdas         = [8.0, 16.0, 32.0];  psis            = [0.0, np.pi / 2];  ksize           = 13; sigma           = 4.0;      gamma = 0.5
        
        rot_left, rot_right     = np.pi/16*0, np.pi/16
        thetas_left             = [t + rot_left for t in thetas]  # rotate left image Gabors slightly counterclockwise
        thetas_right            = [t - rot_right for t in thetas]
        bank_left, p_left        = p.gabor_bank_init(ksize=ksize, sigma=sigma, lambdas=lambdas, gamma=gamma, psis=psis, thetas=thetas_left)
        bank_right, p_right      = p.gabor_bank_init(ksize=ksize, sigma=sigma, lambdas=lambdas, gamma=gamma, psis=psis, thetas=thetas_right)        

        img_left, img_right = d.imgL, d.imgR

        # Build NxMxC responses for both images.
        gabor_left         = p.gabor_bank_filter(img_left, bank=bank_left)
        gabor_right        = p.gabor_bank_filter(img_right, bank=bank_right)

        p.show_outputs(p_left,  bank_left,  gabor_left,  cols=6, ttl = 'Left image: Gabor responses with rotated kernels')
        p.show_outputs(p_right, bank_right, gabor_right, cols=6, ttl = 'Right image: Gabor responses with rotated kernels')

        p.show_kernels(p_left,  bank_left,  cols=6, ttl='Gabor kernels - Left image')
        p.show_kernels(p_right, bank_right, cols=6, ttl='Gabor kernels - Right image')

        plt.show()

    def test_gabor_line_correlation_multiscale_floor(self):
        "compute row-wise left/right gabor channel inner products and show MxM matrix. Tricks gabor to rotate floor pattern according to the camera angle to improve floor disparity estimation"
        
        d               = DataSource()
        ret             = d.init_image(56) # 4,7,9,11,54,55,56-ok, ,62,66-nok
        self.assertTrue(ret)
        #d.show_images_left_right()

        level_num       = 3
        max_disparity   = 128
        row_index       = 280 #128 #128 # Select row K (middle row by default).
        p               = ShazamDepthEstimator()
        prob_weights    = 2**np.arange(0,level_num)  # e.g. for 4 levels: [8, 4, 2, 1]

        col_num                 = d.imgL.shape[1]
        confidence_array        = np.zeros((level_num, col_num), dtype=np.float32)

        # Gabor bank parameters.
        rot_left, rot_right     = np.pi/16*0, np.pi/10*0  # rotate left image Gabors slightly counterclockwise and right image Gabors slightly clockwise to better match floor pattern    
        #rot_left, rot_right   = 0, 0 
        thetas                  = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        lambdas                 = [8.0, 16.0, 32.0];  psis            = [0.0, np.pi / 2];  ksize           = 13; sigma           = 4.0;      gamma = 0.5
        thetas_left             = [t + rot_left for t in thetas]  # rotate left image Gabors slightly counterclockwise
        thetas_right            = [t - rot_right for t in thetas]
        bank_left, p_left        = p.gabor_bank_init(ksize=ksize, sigma=sigma, lambdas=lambdas, gamma=gamma, psis=psis, thetas=thetas_left)
        bank_right, p_right      = p.gabor_bank_init(ksize=ksize, sigma=sigma, lambdas=lambdas, gamma=gamma, psis=psis, thetas=thetas_right)         

        prob                     = np.zeros((max_disparity,d.imgL.shape[1]), dtype=np.float32)
        img_left, img_right = d.imgL, d.imgR
        for k in range(level_num):

            # Build NxMxC responses for both images.
            gaborL              = p.gabor_bank_filter(img_left, bank=bank_left)
            gaborR              = p.gabor_bank_filter(img_right, bank=bank_right)

            # Row feature matrices: (M, C).
            rowL                = gaborL[row_index, :, :].astype(np.float32)
            rowR                = gaborR[row_index, :, :].astype(np.float32)

            # clip noise - small values that are likely noise can be set to zero to reduce their influence on the correlation. This can be done by setting a threshold below which values are considered noise and set to zero.
            noise_threshold     = 5.0
            rowL[np.abs(rowL) < noise_threshold] = 0
            rowR[np.abs(rowR) < noise_threshold] = 0

            # compute energy / norm
            normL               = np.linalg.norm(rowL, axis=1) 
            normR               = np.linalg.norm(rowR, axis=1) 

            # normize rows to unit vectors to focus on direction rather than magnitude. This can help improve the correlation by making it more about the pattern of responses across channels rather than their absolute strength.
            rowL                = rowL / (normL[:, np.newaxis] + 1e-6)  # shape (M, C)
            rowR                = rowR / (normR[:, np.newaxis] + 1e-6)  # shape (M, C)

            # Inner products across channels for all xL, xR pairs: (M, M).
            #corr                = rowL @ rowR.T

            # Absolute difference across channels for all xL, xR pairs: (M, M).
            # A: (N, C) and B: (M, C)
            expandedL           = rowL[:, np.newaxis, :]  # Shape (N, 1, C)
            expandedR           = rowR[np.newaxis, :, :]  # Shape (1, M, C)

            # Subtracting (N, 1, C) and (1, M, C) results in (N, M, C)
            # Then we sum along the last axis (C)
            corr                 = np.abs(expandedL - expandedR).mean(axis=2)  # shape (N, M)
            corr_disp            = extract_diag_band_fast(corr, B=max_disparity) 
            corr                 = np.exp(-corr/0.1)  # (N, M) similarity matrix with exponential decay
            corr_disp            = softmax_columns(-corr_disp, dim=0, T=0.01)  # normalize to [0, 1], higher is more similar
             # keep only values close to the diagonal, which correspond to similar column indices in left and right images. This can help focus on likely matches and reduce noise from unrelated pairs.
            #corr_disp            = corr_disp/np.sum(corr_disp, axis=0, keepdims=True)  # normalize to [0, 1] for display
            
            row_num_current, col_num_current   = corr_disp.shape
            # remove no response columns and rows
            corr[normL < 1e-1, :] = 0
            corr[:, normR < 1e-1] = 0            

            # maxima suppression over matrix corr. For each column, find the maximum value and count how many values are above a certain percentage of that maximum. This gives a measure of how many strong matches there are for each column, which can be used as a confidence score.
            top_max_ind             = np.argmax(corr_disp, axis=0)  # shape (M,)
            # suppress the max index and close one pixel around it to avoid counting very close matches
            top_max_low_ind         = np.maximum(0, top_max_ind-1)
            top_max_high_ind        = np.minimum(row_num_current-1, top_max_ind+1) 
            corr2                   = corr_disp.copy()
            top_max_values          = corr_disp[top_max_ind, np.arange(col_num_current)]  # shape (M,)
            corr2[top_max_ind, np.arange(col_num_current)] = 0  # zero out the max values
            corr2[top_max_low_ind, np.arange(col_num_current)] = 0  # zero out the low neighbors
            corr2[top_max_high_ind, np.arange(col_num_current)] = 0  #


            # compute confidence by averaging over 95% of maximum for each column of the correlation matrix
            second_max_values       = np.max(corr2, axis=0)  # shape (M,)
            confidence              = np.zeros_like(top_max_values)
            valid_mask             = top_max_values > 0.1
            confidence[valid_mask] = 1 - (second_max_values[valid_mask] / top_max_values[valid_mask] )  # shape (M,), higher confidence if second max is much lower than top max  
        
            #pixels_above_threshold  = ((corr >= top_max_values*top_percentile) & (corr > 0.4)).sum(axis=0)  # shape (M,)
            # make confidence a matrix of the same shape as corr by repeating the confidence values for each row
            # confidence              = pixels_above_threshold #np.exp(-(pixels_above_threshold-1)/8.0)  # shape (1, M)
            #confidence_mtrx         = np.repeat(confidence[np.newaxis, :], corr.shape[0], axis=0)  # shape (N, M)
            #prob_current            = cv.resize(confidence_mtrx, (prob.shape[1], prob.shape[0]))  # shape (N, M)
            prob_current            = cv.resize(corr_disp, (prob.shape[1], prob.shape[0]))  #
              # normalize to [0, 1]

            #corr                    = corr * confidence_mtrx  # shape (N, M)
            #corr[:,confidence < 0.3] = 0  # zero out low confidence matches



            # show
            img_show = np.concatenate((img_left, img_right ), axis = 0)
            img_show[[row_index,row_index+img_left.shape[0]],:] = 255  # highlight the row in both images
            plt.figure(figsize=(8, 12))
            plt.imshow(img_show, cmap='gray')
            plt.title(f'Image L-R : Level {k}, row {row_index}')              

            # show image rows
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1) 
            plt.plot(rowL, '-.')
            plt.title(f'Gabor row features: Left image, Level {k}, row {row_index}')
            plt.subplot(1, 2, 2)        
            plt.plot(rowR, '-.')
            plt.title(f'Gabor row features: Right image, Level {k}, row {row_index}')
            plt.tight_layout()

            # show confidence
            plt.figure(figsize=(8, 4))
            plt.plot(confidence, '-o')
            plt.title(f'Confidence: Level {k}, row {row_index}')
            plt.xlabel('Right image column index')
            plt.ylabel('Confidence')
            plt.tight_layout()

            # Plot correlation matrix.
            plt.figure(figsize=(8, 7))
            plt.imshow(corr, cmap='viridis', aspect='auto')
            plt.colorbar(label='Inner product over C channels')
            plt.xlabel('Right image column index')
            plt.ylabel('Left image column index')
            plt.title(f'Gabor row correlation matrix at row Level {k}, row {row_index}')
            plt.tight_layout()
            plt.show(block=False)

            plt.figure(figsize=(8, 4))
            plt.imshow(corr_disp, cmap='viridis', aspect='auto')
            plt.colorbar(label='Inner product over C channels')
            plt.xlabel('Right image column index')
            plt.ylabel('Left image column index')
            plt.title(f'Gabor row correlation matrix (diagonal band) at row Level {k}, row {row_index}')
            plt.tight_layout()
            plt.show(block=False)

            # show image rows
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1) 
            plt.imshow(prob_current, cmap='viridis', aspect='auto')
            plt.colorbar(label='Probability current level')
            plt.title(f'Probability current level , Level {k}, row {row_index}')
            plt.subplot(1, 2, 2)        
            plt.imshow(prob, cmap='viridis', aspect='auto')
            plt.colorbar(label='Probability accumulated')
            plt.title(f'Probability accumulated , Level {k}, row {row_index}')
            plt.tight_layout()            

            # downscale
            img_left  = cv.pyrDown(img_left)
            img_right = cv.pyrDown(img_right)
            row_index = row_index // 2
            max_disparity = max_disparity // 2
            #prob      = prob + prob_current*prob_weights[k]
            #prob       = np.maximum(prob , prob_current)
            #prob      = (prob*k + prob_current*(level_num-k))/level_num  
            #prob       = prob * prob_current
            prob       = softmax_columns(prob + prob_current,dim=0,T=0.1)

            # save confidence for this level - interpolate to original size
            interpolated_values = np.interp(np.arange(col_num), np.linspace(0, col_num, col_num_current), confidence)
            confidence_array[k,:] = interpolated_values


        # Plot correlation matrix.
        prob = prob / prob_weights.sum()  # average over levels
        plt.figure(figsize=(8, 7))
        plt.imshow(prob, cmap='viridis', aspect='auto')
        plt.colorbar(label='Probability of match (sum of levels)')
        plt.xlabel('Right image column index')
        plt.ylabel('Left image column index')
        plt.title(f'probability')
        plt.tight_layout()
        plt.show(block=False)

        # Plot confidence across columns.
        plt.figure(figsize=(8, 4))
        plt.plot(confidence_array.T, '-o')
        plt.title('Confidence across columns for each level')
        plt.xlabel('Pixel column index')
        plt.ylabel('Confidence')
        plt.legend([f'Level {k}' for k in range(level_num)], loc='upper right')
        plt.tight_layout()


        plt.show()
        return True
        
    def test_gabor_line_correlation_updown(self):
        "compute row-wise left/right gabor channel correlation at multiple levels"
        "shows the correlation MxM matrix. integrates information from coars to fine levels"
        ""
        
        d               = DataSource()
        ret             = d.init_image(55) # 4,7,9,11,54,55,56-ok, ,62,66-nok
        self.assertTrue(ret)
        #d.show_images_left_right()

        level_num       = 4
        row_index       = 128 #128 # Select row K (middle row by default).
        p               = ShazamDepthEstimator()
        prob_weights    = 4+2**np.arange(0,level_num)  # e.g. for 4 levels: [8, 4, 2, 1]
        prob_scaling    = 1/prob_weights[::-1]

        col_num        = d.imgL.shape[1]
        confidence_array = np.zeros((level_num, col_num), dtype=np.float32)

        # Gabor bank parameters.
        thetas          = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        lambdas         = [8.0, 16.0, 32.0];  psis            = [0.0, np.pi / 2];  ksize           = 13; sigma           = 4.0;      gamma = 0.5
        bank            = p.gabor_bank_init(ksize=ksize, sigma=sigma, lambdas=lambdas, gamma=gamma, psis=psis, thetas=thetas)

        prob            = np.zeros((d.imgL.shape[1],d.imgL.shape[1]), dtype=np.float32)
        img_left, img_right = d.imgL, d.imgR
        corr_matrix_levels = []
        for k in range(level_num):

            # Build NxMxC responses for both images.
            gaborL              = p.gabor_bank_filter(img_left, bank=bank)
            gaborR              = p.gabor_bank_filter(img_right, bank=bank)

            # Row feature matrices: (M, C).
            rowL                = gaborL[row_index, :, :].astype(np.float32)
            rowR                = gaborR[row_index, :, :].astype(np.float32)

            # compute energy / norm
            normL               = np.linalg.norm(rowL, axis=1) 
            normR               = np.linalg.norm(rowR, axis=1) 

            # Absolute difference across channels for all xL, xR pairs: (M, M).
            # A: (N, C) and B: (M, C)
            expandedL           = rowL[:, np.newaxis, :]  # Shape (N, 1, C)
            expandedR           = rowR[np.newaxis, :, :]  # Shape (1, M, C)

            # Subtracting (N, 1, C) and (1, M, C) results in (N, M, C)
            # Then we sum along the last axis (C)
            corr                    = np.exp(-np.abs(expandedL - expandedR).mean(axis=2)*prob_scaling[k])  # (N, M) similarity matrix with exponential decay
            row_num_current, col_num_current   = corr.shape

            # normalize by rows to get a probability distribution over right columns for each left column
            corr_sum_over_rows       = corr.sum(axis=0, keepdims=True) + 1e-6  # shape (1, M)
            corr                     = corr / corr_sum_over_rows  # shape (N, M), now

            # remove no response columns and rows
            corr[normL < 1e-1, :] = 0
            corr[:, normR < 1e-1] = 0            
            
            # save
            corr_matrix_levels.append(corr)



            # show
            img_show = np.concatenate((img_left, img_right ), axis = 0)
            img_show[[row_index,row_index+img_left.shape[0]],:] = 255  # highlight the row in both images
            plt.figure(figsize=(4, 8))
            plt.imshow(img_show, cmap='gray')
            plt.title(f'Image L-R : Level {k}, row {row_index}')              

            # show image rows
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1) 
            plt.plot(rowL, '-.')
            plt.title(f'Gabor row features: Left image, Level {k}, row {row_index}')
            plt.subplot(1, 2, 2)        
            plt.plot(rowR, '-.')
            plt.title(f'Gabor row features: Right image, Level {k}, row {row_index}')
            plt.tight_layout()  

            # Plot correlation matrix.
            plt.figure(figsize=(8, 7))
            plt.imshow(corr, cmap='viridis', aspect='auto')
            plt.colorbar(label='Inner product over C channels')
            plt.xlabel('Right image column index')
            plt.ylabel('Left image column index')
            plt.title(f'Gabor row correlation matrix at row Level {k}, row {row_index}')
            plt.tight_layout()
            plt.show(block=False)     

            # downscale
            img_left  = cv.pyrDown(img_left)
            img_right = cv.pyrDown(img_right)
            row_index = row_index // 2                 


        # loop to inegrate information from coarse levels to fine levels
        corr_current = corr_matrix_levels[-1]  # start from the coarsest level
        for k in reversed(range(level_num-1)):

            corr_previous_level          = cv.pyrUp(corr_current)  # upsample the correlation matrix from the previous coarser level
            corr_current                 = corr_matrix_levels[k]
            row_num_current, col_num_current   = corr_current.shape
            corr_previous                = cv.resize(corr_previous_level, (col_num_current, row_num_current))
            

            # show image rows
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 2) 
            plt.imshow(corr_current, cmap='viridis', aspect='auto')
            plt.colorbar(label='Probability current level')
            plt.title(f'Probability current level , Level {k}, row {row_index}')
            plt.subplot(1, 2, 1)        
            plt.imshow(corr_previous, cmap='viridis', aspect='auto')
            plt.colorbar(label='Probability accumulated')
            plt.title(f'Probability accumulated , Level {k}, row {row_index}')
            plt.tight_layout()

            #corr_current    = corr_current * corr_previous  # element-wise multiplication to integrate information from previous level            
            corr_current    = np.maximum(corr_current , corr_previous)
        plt.show()
        return True

    def test_gabor_image_disparity_multiscale(self):
        "compute row-wise left/right gabor channel correlation at multiple levels and extract disparity information to get a disparity map. shows the correlation MxM matrix. integrates information from coars to fine levels"
        ""
        
        d               = DataSource()
        ret             = d.init_image(622) # 4,7,9,11,54-chair,55,56-off-ok, ,62,66-nok, 601-ok, 621,622
        d.show_images_left_right()
        img_left, img_right = d.imgL, d.imgR

        debug_row       = 400   # set to None to disable per-row debug plots

        p               = ShazamDepthEstimator()
        prob            = p.gabor_image_disparity_multiscale(img_left, img_right, debug_row=debug_row)
        return True
    
    def test_gabor_image_disparity_down_up(self):
        "compute row-wise left/right gabor channel correlation at multiple levels and extract disparity information to get a disparity map. shows the correlation MxM matrix. integrates information from coars to fine levels"
        ""
        
        d               = DataSource()
        ret             = d.init_image(624) # 4-ok,7-ok,11-nok,21-sim,26-ok, 54-chair,55-office far,56-office-chess-ok, ,62,66-nok, 71-home, 601-ok, 620,621,622-mbox
        d.show_images_left_right()
        img_left, img_right = d.imgL, d.imgR

        debug_row       = 125 #140   # set to None to disable per-row debug plots

        p               = ShazamDepthEstimator()
        #prob            = p.gabor_image_disparity_down_up(img_left, img_right, debug_row=debug_row)
        #prob            = p.gabor_image_disparity_down_up_on_volume(img_left, img_right, debug_row=debug_row)
        prob            = p.gabor_image_disparity_down_up_full_volume(img_left, img_right, debug_row=debug_row)
        
        return True    
        
    def test_context_upsampling(self):
        "test upsampling of disparity map using context from the image. The idea is to use the image features to guide the upsampling of a coarse disparity map to a finer resolution, which can help preserve edges and details in the disparity map."
        
        d               = DataSource()
        ret             = d.init_image(56) # 4,7,9,11,54,55,56-ok, ,62,66-nok
        #d.show_images_left_right()
        img_left, img_right = d.imgL, d.imgR
        height, width       = img_left.shape[:2]
        p              = ShazamDepthEstimator()

        # 1. Load low-res image
        img_low         = cv.pyrDown(cv.pyrDown(img_left))  # simulate a low-res disparity map by downsampling the left image. In practice, this would be your coarse disparity map.

        # 2. Perform a standard upsample to act as your spatial baseline
        # (We scale by 4x using standard bicubic as a starting point)

        img_large_baseline = cv.resize(img_low, (width , height ), interpolation=cv.INTER_CUBIC)

        # 3. Apply a Guided Filter
        # This uses the baseline image to guide its own smoothing, 
        # filtering out pixel noise while preserving sharp contextual edges.
        # d = filter radius, eps = regularization (higher values = smoother edges)
        context_aware_upsample = cv.ximgproc.guidedFilter(
            guide=img_large_baseline, 
            src=img_large_baseline, 
            radius=4, 
            eps=100
        )

        # 4. Show results
        img_list = [img_left, img_large_baseline, context_aware_upsample]
        ttl_list = ['Original Image', 'Upsampled (Bicubic)', 'Context-Aware Upsampled']
        p.show_subset(img_list, ttl_list)

        return True

    def test_gabor_bank_upsampling(self):
        "test upsampling of a low-resolution image using a Gabor filter bank as structural guidance"
        
        d               = DataSource()
        ret             = d.init_image(56) # 4,7,9,11,54,55,56-ok, ,62,66-nok
        self.assertTrue(ret)
        #d.show_images_left_right()
        img_left, _     = d.imgL, d.imgR
        height, width   = img_left.shape[:2]
        p               = ShazamDepthEstimator()

        # 1. Load low-res image
        img_low         = cv.pyrDown(cv.pyrDown(img_left)).astype(np.float32)
        img_low_nearest = cv.resize(img_low, (width, height), interpolation=cv.INTER_NEAREST).astype(np.float32)  # also create a nearest neighbor upsampled version for comparison

        # 2. Standard upsample to act as the spatial baseline
        img_large_baseline = cv.resize(img_low, (width, height), interpolation=cv.INTER_CUBIC)

        # 3. Build Gabor filter bank and compute response energy on the guidance image
        thetas          = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        lambdas         = [8.0, 16.0, 32.0]
        psis            = [0.0, np.pi / 2]
        bank, _         = p.gabor_bank_init(ksize=13, sigma=4.0, lambdas=lambdas, gamma=0.5, psis=psis, thetas=thetas)

        gabor_full      = p.gabor_bank_filter(img_left.astype(np.float32), bank=bank)
        gabor_low       = p.gabor_bank_filter(img_low, bank=bank)
        gabor_low_up    = cv.resize(gabor_low, (width, height), interpolation=cv.INTER_CUBIC)

        energy_full     = np.mean(np.abs(gabor_full), axis=2)
        energy_low_up   = np.mean(np.abs(gabor_low_up), axis=2)
        energy_delta    = np.maximum(energy_full - energy_low_up, 0)
        gabor_gain      = cv.normalize(energy_delta, None, 0.0, 1.0, cv.NORM_MINMAX)

        # 4. Use Gabor-supported detail injection to refine the bicubic result
        img_float       = img_left.astype(np.float32)
        detail_map      = img_float - cv.GaussianBlur(img_float, (0, 0), 1.2)
        gabor_upsampled = img_large_baseline + 0.35 * gabor_gain * detail_map
        gabor_upsampled = np.clip(gabor_upsampled, 0, 255).astype(np.uint8)
        gabor_gain_u8   = np.clip(gabor_gain * 255.0, 0, 255).astype(np.uint8)

        # 5. Basic sanity checks
        self.assertTrue(img_large_baseline.shape == img_left.shape)
        self.assertTrue(gabor_upsampled.shape == img_left.shape)
        self.assertTrue(np.isfinite(gabor_gain).all())
        self.assertTrue(np.mean(np.abs(gabor_upsampled.astype(np.float32) - img_large_baseline.astype(np.float32))) > 0.01)

        # 6. Show results
        img_list = [img_left, img_low_nearest, img_large_baseline, gabor_gain_u8, gabor_upsampled]
        ttl_list = ['Original Image', 'Low-Resolution Input', 'Upsampled (Bicubic)', 'Gabor Detail Gain', 'Gabor-Guided Upsampled']
        p.show_subset(img_list, ttl_list)

        return True

    def test_context_upsampling_using_ai(self):
        "test upsampling of disparity map using context from the image. The idea is to use the image features to guide the upsampling of a coarse disparity map to a finer resolution, which can help preserve edges and details in the disparity map."
        
        d               = DataSource()
        ret             = d.init_image(56) # 4,7,9,11,54,55,56-ok, ,62,66-nok
        #d.show_images_left_right()
        img_left, img_right = d.imgL, d.imgR
        height, width       = img_left.shape[:2]
        p              = ShazamDepthEstimator()

        # 1. Load low-res image
        img_low         = cv.pyrDown(cv.pyrDown(img_left))  # simulate a low-res disparity map by downsampling the left image. In practice, this would be your coarse disparity map.

        # 2. Perform a standard upsample to act as your spatial baseline
        # (We scale by 4x using standard bicubic as a starting point)

        img_large_baseline = cv.resize(img_low, (width , height ), interpolation=cv.INTER_CUBIC)

        # 3. Apply a Guided Filter
        # This uses the baseline image to guide its own smoothing, 
        # filtering out pixel noise while preserving sharp contextual edges.
        # d = filter radius, eps = regularization (higher values = smoother edges)
        context_aware_upsample = cv.ximgproc.guidedFilter(
            guide=img_large_baseline, 
            src=img_large_baseline, 
            radius=4, 
            eps=100
        )

        # 1. Initialize the Super Resolution object
        sr = cv.dnn_superres.DnnSuperResImpl_create()


        # 3. Load the pre-trained model and set the network
        # (Make sure the path matches where you downloaded the .pb file)
        model_path = "src/EDSR_x4.pb"
        sr.readModel(model_path)

        # 4. Set the model name and upscale factor (EDSR supports 2, 3, or 4)
        sr.setModel("edsr", 4)

        # 5. Upscale the image using deep context awareness
        # create 3 color channels from img_low
        img_low_color = cv.cvtColor(img_low.astype(np.uint8), cv.COLOR_GRAY2BGR)
        high_res_image = sr.upsample(img_low_color)        

        # 4. Show results
        img_list = [img_left, img_large_baseline, context_aware_upsample, high_res_image]
        ttl_list = ['Original Image', 'Upsampled (Bicubic)', 'Context-Aware Upsampled', 'Deep Context-Aware Upsampled']
        p.show_subset(img_list, ttl_list)

        return True

    def test_context_upsampling_validation(self):
        "validation test for context-aware upsampling, similar to test_context_upsampling but with explicit checks"

        d               = DataSource()
        ret             = d.init_image(56)
        self.assertTrue(ret)
        #d.show_images_left_right()
        img_left, _     = d.imgL, d.imgR
        h, w            = img_left.shape[:2]
        p               = ShazamDepthEstimator()

        # 1) Build low-resolution source (proxy for coarse disparity)
        img_low         = cv.pyrDown(cv.pyrDown(img_left)).astype(np.float32)

        # 2) Baseline upsampling (spatial only)
        baseline        = cv.resize(img_low, (w, h), interpolation=cv.INTER_CUBIC)

        # 3) Context-aware refinement
        if baseline.dtype != np.uint8:
            guide_u8 = cv.normalize(img_left, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
            src_u8 = cv.normalize(baseline, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        else:
            guide_u8 = img_left
            src_u8 = baseline.astype(np.uint8)

        context_guided = cv.ximgproc.guidedFilter(
            guide=guide_u8,
            src=src_u8,
            radius=4,
            eps=25.0
            )
        context_join = cv.ximgproc.jointBilateralFilter(
            joint=guide_u8,
            src=src_u8,
            d=7,
            sigmaColor=18.0,
            sigmaSpace=4.0
            )

        # fallback if opencv-contrib is unavailable
        context_bilateral = cv.bilateralFilter(src_u8, d=7, sigmaColor=18.0, sigmaSpace=4.0)


        # 5) Visual check

        img_list = [img_left, baseline, context_guided, context_join, context_bilateral]
        ttl_list = [ 'Guidance Image',  'Upsampled Baseline (Cubic)', 'Context-Aware Upsampled (Guided)',
            'Context-Aware Upsampled (Joint Bilateral)', 'Context-Aware Upsampled (Bilateral)'
        ]
        p.show_subset(img_list, ttl_list)

        return True

    def test_down_upsampling_consistency(self):
    
        """Downsamples a 3D array and then upsamples it back to the original size,

        minimizing and calculating the reconstruction error.
        """

        # Create a dummy 3D array (e.g., 64x64x64 grid)
        # Simulating a smooth volumetric gradient (like a physical field or scan)
        x = np.linspace(-2, 2, 64)
        y = np.linspace(-2, 2, 64)
        z = np.linspace(-2, 2, 64)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Mathematical 3D data: a smooth spherical Gaussian field
        original_3d = np.exp(-(X**2 + Y**2 + Z**2)) * (2 * (Y < 0) - 1)  # add an edge by zeroing out one half of the sphere


        original_array = original_3d
        scale_factor = 0.25  # Downsample to 25% of original size (16x16x16)
        # 1. DOWNSAMPLE
        # order=3 specifies cubic spline interpolation for all 3 dimensions
        downsampled = zoom(original_array, zoom=scale_factor, order=1)

        # 2. UPSAMPLE
        # To return to original size, invert the scale factor (e.g., 1 / 0.5 = 2.0)
        upsample_factor = 1.0 / scale_factor
        upsampled = zoom(downsampled, zoom=upsample_factor, order=1)

        # Note: Because of rounding in matrix dimensions, ensure shapes match perfectly
        if upsampled.shape != original_array.shape:
            # Pad or crop slightly if floating-point zoom caused a 1-pixel mismatch
            slices = tuple(slice(0, min(i, j)) for i, j in zip(upsampled.shape, original_array.shape))
            
            # Create a perfectly sized base array
            fixed_upsampled = np.zeros_like(original_array)
            fixed_upsampled[slices] = upsampled[slices]
            upsampled = fixed_upsampled

        # 3. Calculate Reconstruction Error (Mean Squared Error)
        error = np.mean((original_array - upsampled) ** 2)

        # Downsample by 50% (64^3 -> 32^3) and reconstruct back to 64^3
        down, up, error = downsampled, upsampled, error

        print(f"Original Shape:    {original_3d.shape}")
        print(f"Downsampled Shape: {down.shape}")
        print(f"Upsampled Shape:   {up.shape}")
        print(f"Mean Squared Reconstruction Error: {error:.1e}")

        # show original, downsampled, and upsampled slices for visual inspection
        slice_index = original_3d.shape[2] // 2  # middle slice
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(original_3d[:, :, slice_index], cmap='viridis')
        plt.title('Original Slice')
        #plt.colorbar()
        plt.subplot(1, 3, 2)
        plt.imshow(down[:, :, down.shape[2] // 2], cmap='viridis')
        plt.title('Downsampled Slice')
        #plt.colorbar()
        plt.subplot(1, 3, 3)
        plt.imshow(up[:, :, up.shape[2] // 2], cmap='viridis')
        plt.title('Upsampled Slice')    
        #plt.colorbar()
        plt.tight_layout()
        plt.show()

    def test_edge_preserving_methods(self):
        "validation test for several edge-preserving filtering methods using the same setup as context-aware upsampling"

        d               = DataSource()
        ret             = d.init_image(56)
        self.assertTrue(ret)
        #d.show_images_left_right()
        img_left, _     = d.imgL, d.imgR
        h, w            = img_left.shape[:2]
        p               = ShazamDepthEstimator()

        # 1) Build low-resolution source (proxy for coarse disparity)
        #img_low         = cv.pyrDown(cv.pyrDown(img_left)).astype(np.float32)
        img_low         = cv.pyrDown(img_left).astype(np.float32)
        #img_low         = img_left.astype(np.float32)
        # add some noise to make it more realistic and to test the edge-preserving capabilities of the filters. This simulates the imperfections that might be present in a real coarse disparity map, 
        # making the test more robust and relevant to practical applications.  
        # 1. Generate random Gaussian noise with the same shape as the image
        sigma           = 15.0
        noise           = np.random.normal(0, sigma, img_low.shape).astype(np.float32)
        
        # 2. Add the noise to the original image
        # We convert the image to float32 first to prevent uint8 underflow/overflow
        noisy_image     = img_low + noise

        # # 1. Calculate how many pixels should be altered
        # amount           = 0.01
        # num_salt        = np.ceil(amount * img_low.size * 0.5).astype(int)
        # num_pepper      = np.ceil(amount * img_low.size * 0.5).astype(int)
        
        # # 2. Add Salt (White pixels)
        # # Generate random coordinates across all dimensions
        # coords          = [np.random.randint(0, i - 1, num_salt) for i in img_low.shape]
        # noisy_image[tuple(coords)] = 255
        
        # # 3. Add Pepper (Black pixels)
        # coords          = [np.random.randint(0, i - 1, num_pepper) for i in img_low.shape]
        # noisy_image[tuple(coords)] = 0        
        
        # 3. Clip values to keep them in the valid [0, 255] range, then convert back to uint8
        #noisy_image     = np.clip(noisy_image, 0, 255).astype(np.uint8)        
        img_low         = noisy_image

        # 2) Baseline upsampling (spatial only)
        baseline        = cv.resize(img_low, (w, h), interpolation=cv.INTER_CUBIC)

        # 3) Prepare guide and source for edge-preserving filters
        guide_u8        = cv.normalize(img_left, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        src_u8          = cv.normalize(baseline, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

        img_list        = [img_left, baseline]
        ttl_list        = ['Guidance Image', 'Upsampled Baseline (Cubic)']

        # 4) Guided filter
        context_guided  = cv.ximgproc.guidedFilter(
            guide=guide_u8,
            src=src_u8,
            radius=4,
            eps=25.0
            )
        img_list.append(context_guided)
        ttl_list.append('Edge-Preserving (Guided)')

        # 5) Joint bilateral filter
        context_joint   = cv.ximgproc.jointBilateralFilter(
            joint=guide_u8,
            src=src_u8,
            d=7,
            sigmaColor=18.0,
            sigmaSpace=4.0
            )
        img_list.append(context_joint)
        ttl_list.append('Edge-Preserving (Joint Bilateral)')

        # 6) Classical bilateral filter
        context_bilateral = cv.bilateralFilter(src_u8, d=7, sigmaColor=18.0, sigmaSpace=4.0)
        img_list.append(context_bilateral)
        ttl_list.append('Edge-Preserving (Bilateral)')

        # 7) Additional OpenCV edge-preserving methods when available
        if hasattr(cv.ximgproc, 'rollingGuidanceFilter'):
            context_rolling = cv.ximgproc.rollingGuidanceFilter(src_u8, d=7, sigmaColor=18.0, sigmaSpace=4.0, numOfIter=4)
            img_list.append(context_rolling)
            ttl_list.append('Edge-Preserving (Rolling Guidance)')

        if hasattr(cv.ximgproc, 'fastGlobalSmootherFilter'):
            context_fgs = cv.ximgproc.fastGlobalSmootherFilter(
                guide=guide_u8,
                src=src_u8,
                lambda_=12.0,
                sigma_color=8.0
                )
            img_list.append(context_fgs)
            ttl_list.append('Edge-Preserving (Fast Global Smoother)')

        if hasattr(cv, 'edgePreservingFilter'):
            src_bgr = cv.cvtColor(src_u8, cv.COLOR_GRAY2BGR)
            context_epf = cv.edgePreservingFilter(src_bgr, flags=1, sigma_s=40, sigma_r=0.15)
            context_epf = cv.cvtColor(context_epf, cv.COLOR_BGR2GRAY)
            img_list.append(context_epf)
            ttl_list.append('Edge-Preserving (Photo Filter)')


        #context_diff = anisotropic_diffusion(src_u8, num_iter=10, kappa=5, delta=0.3)
        #context_diff = p.anisotropic_filter_avergaing(baseline, img_left)
        context_diff = p.anisotropic_filter(baseline, img_left)
        img_list.append(context_diff)
        ttl_list.append('Edge-Preserving (Anisotropic Diffusion)')        

        # 8) Sanity checks
        for img in img_list[2:]:
            self.assertTrue(img.shape == img_left.shape)
        self.assertTrue(len(img_list) >= 5)

        # 9) Visual comparison
        p.show_subset(img_list, ttl_list)

        return True

    def test_kalman_filtering(self):
        # --- Test Case Setup ---
        from scipy.ndimage import sobel

        # 1. Create a clean baseline: A white square on a dark gray background
        N = 200
        base_img = np.ones((N, N)) * 0.2
        base_img[50:150, 50:150] = 0.8
        base_img[N//2-5:N//2+5, :] = 0.2  # add a horizontal line
        #base_img  = base_img + np.rotate(base_img, 45)   # add some structure to the image

        # 2. Generate Noisy Observations
        # Image A: Ruined by noise on the left half
        noise_A = np.random.normal(0, 0.1, (N, N))
        img_A = base_img.copy()
        ind_A = np.triu(np.ones((N, N))) == 1  # Identity mask to add noise only to the left half
        img_A[ind_A] += noise_A[ind_A]
        img_A = np.clip(img_A, 0, 1)


        # 3. Derive Probabilities from Edges (Sobel Filter)
        def get_edge_probability(img):
            # Compute gradients along x and y axes
            dx = sobel(img, axis=1)
            # dy = sobel(img, axis=0)
            edge_magnitude = np.hypot(dx, 0)

            #edge_magnitude = np.abs(img - cv.GaussianBlur(img, (5, 5), 1.5))
            
            # # Normalize to [0, 1] to treat as a valid probability/confidence metric
            # if np.max(edge_magnitude) > 0:
            #     edge_magnitude /= np.max(edge_magnitude)
            edge_probability = 1- np.exp(-edge_magnitude/0.03)*0.999
            return edge_probability

        p_A = get_edge_probability(img_A)


        # this is working
        #p_A  = anisotropic_diffusion(img_A, num_iter=20, delta=0.1, kappa=0.1)
        #p_B  = anisotropic_diffusion(img_B, num_iter=20, delta=0.1, kappa=0.1)

        # 4. Run the Neighborhood Kalman Fusion
        s           = ShazamDepthEstimator()
        img_C, p_C  = s.kalman_neighborhood_fusion(img_A, p_A, kernel_size=11)

        s.show_subset([img_A, img_C, p_A, p_C], ['Noisy Input', 'Kalman Filtered Output', 'Probability P_A', 'Probability P_C'])
        
        # 5. One more time
        img_C2, p_C2  = s.kalman_neighborhood_fusion(img_C, p_A, kernel_size=11)

        s.show_subset([img_C, img_C2, p_A, p_C2], ['Kalman Filtered Output', 'Kalman Filtered Output 2', 'Probability P_C', 'Probability P_C2'])       

        plt.show()

    def test_kalman_filtering_two_images(self):
        # --- Test Case Setup ---
        from scipy.ndimage import sobel

        # 1. Create a clean baseline: A white square on a dark gray background
        N = 200
        base_img = np.ones((N, N)) * 0.2
        base_img[50:150, 50:150] = 0.8
        #base_img  = base_img + np.rotate(base_img, 45)   # add some structure to the image

        # 2. Generate Noisy Observations
        # Image A: Ruined by noise on the left half
        noise_A = np.random.normal(0, 0.1, (N, N))
        img_A = base_img.copy()
        ind_A = np.triu(np.ones((N, N))) == 1  # Identity mask to add noise only to the left half
        img_A[ind_A] += noise_A[ind_A]
        img_A = np.clip(img_A, 0, 1)

        # Image B: Ruined by noise on the right half
        noise_B = np.random.normal(0, 0.1, (N, N))
        img_B = base_img.copy()
        #img_B[:, N//2:] += noise_B[:, N//2:]
        ind_B = np.rot90(np.triu(np.ones((N, N)))) == 1
        img_B[ind_B] += noise_B[ind_B]
        img_B = np.clip(img_B, 0, 1)

        # 3. Derive Probabilities from Edges (Sobel Filter)
        def get_edge_probability(img):
            # Compute gradients along x and y axes
            # dx = sobel(img, axis=0)
            # dy = sobel(img, axis=1)
            # edge_magnitude = np.hypot(dx, dy)

            edge_magnitude = np.abs(img - cv.GaussianBlur(img, (5, 5), 1.5))
            
            # # Normalize to [0, 1] to treat as a valid probability/confidence metric
            # if np.max(edge_magnitude) > 0:
            #     edge_magnitude /= np.max(edge_magnitude)
            edge_probability = 1- np.exp(-edge_magnitude/0.03)*0.999
            return edge_probability

        p_A = get_edge_probability(img_A)
        p_B = get_edge_probability(img_B)

        # this is working
        #p_A  = anisotropic_diffusion(img_A, num_iter=20, delta=0.1, kappa=0.1)
        #p_B  = anisotropic_diffusion(img_B, num_iter=20, delta=0.1, kappa=0.1)

        # 4. Run the Neighborhood Kalman Fusion
        s           = ShazamDepthEstimator()
        img_C, p_C  = s.kalman_neighborhood_fusion_two_images(img_A, p_A, img_B, p_B, kernel_size=3, sigma=2.5)

        

        # --- Visualization ---
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))

        # Top Row: Inputs and Fused Result
        axes[0, 0].imshow(img_A, cmap='gray')
        axes[0, 0].set_title("Image A (Noisy Left)")
        axes[0, 0].axis('off')

        axes[0, 1].imshow(img_B, cmap='gray')
        axes[0, 1].set_title("Image B (Noisy Right)")
        axes[0, 1].axis('off')

        axes[0, 2].imshow(img_C, cmap='gray')
        axes[0, 2].set_title("Fused Image C")
        axes[0, 2].axis('off')

        # Bottom Row: Corresponding Probability Matrices
        axes[1, 0].imshow(p_A, cmap='plasma')
        axes[1, 0].set_title("Probability P_A (Edges)")
        axes[1, 0].axis('off')

        axes[1, 1].imshow(p_B, cmap='plasma')
        axes[1, 1].set_title("Probability P_B (Edges)")
        axes[1, 1].axis('off')

        axes[1, 2].imshow(p_C, cmap='plasma')
        axes[1, 2].set_title("Combined Probability P_C")
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.show()        

    def test_fast_guided_filter(self):
        "test fast guided filter for upsampling / edge-preserving smoothing"
        import cv2

        def fast_guided_filter(p, I, r, eps, s):
            """
            Fast Guided Filter for upsampling / edge-preserving smoothing.
            
            Parameters:
            -----------
            p : np.ndarray
                The low-resolution input image to be guided/upsampled. 
                Shape: (H_lr, W_lr, C) or (H_lr, W_lr).
            I : np.ndarray
                The high-resolution guidance image containing sharp structural details.
                Shape: (H_hr, W_hr, C_guide) or (H_hr, W_hr). Must match C or be 1-channel.
            r : int
                Window radius on the original high-resolution scale.
            eps : float
                Regularization parameter (analogue to range variance).
                A small eps preserves subtle edges; large eps causes more smoothing.
            s : int or float
                Subsampling scale factor (e.g., 2, 4, 8). 
                This is the ratio: (HR size) / (LR size).
                
            Returns:
            --------
            q : np.ndarray
                The final upsampled, edge-preserved high-resolution output.
            """
            # Ensure inputs are float32
            p = p.astype(np.float32)
            I = I.astype(np.float32)
            
            H_hr, W_hr = I.shape[:2]
            
            # Calculate radius for downsampled scale
            r_sub = int(round(r / s))
            ksize_sub = (2 * r_sub + 1, 2 * r_sub + 1)
            
            # 1. Subsample the HR Guide and LR Input to the working scale
            # (If p is already low-resolution, we just resize it to match the subsampled I)
            h_sub, w_sub = int(round(H_hr / s)), int(round(W_hr / s))
            
            I_sub = cv2.resize(I, (w_sub, h_sub), interpolation=cv2.INTER_LINEAR)
            p_sub = cv2.resize(p, (w_sub, h_sub), interpolation=cv2.INTER_LINEAR)
            
            # 2. Compute local means using box filter on the subsampled scale
            mean_I = cv2.boxFilter(I_sub, -1, ksize_sub)
            mean_p = cv2.boxFilter(p_sub, -1, ksize_sub)
            
            mean_II = cv2.boxFilter(I_sub * I_sub, -1, ksize_sub)
            mean_Ip = cv2.boxFilter(I_sub * p_sub, -1, ksize_sub)
            
            # 3. Compute covariance and variance on downsampled scale
            # var(I) = E[I^2] - (E[I])^2
            var_I = mean_II - mean_I * mean_I
            # cov(I, p) = E[Ip] - E[I]*E[p]
            cov_Ip = mean_Ip - mean_I * mean_p
            
            # 4. Compute linear coefficients a and b on downsampled scale
            # a = cov(I,p) / (var(I) + eps)
            # b = mean_p - a * mean_I
            a = cov_Ip / (var_I + eps)
            b = mean_p - a * mean_I
            
            # 5. Compute mean of a and b on downsampled scale
            mean_a = cv2.boxFilter(a, -1, ksize_sub)
            mean_b = cv2.boxFilter(b, -1, ksize_sub)
            
            # 6. Upsample the smoothed a and b back to the high-resolution scale
            mean_a_hr = cv2.resize(mean_a, (W_hr, H_hr), interpolation=cv2.INTER_LINEAR)
            mean_b_hr = cv2.resize(mean_b, (W_hr, H_hr), interpolation=cv2.INTER_LINEAR)
            
            # 7. Apply the linear model with the high-resolution guide I
            q = mean_a_hr * I + mean_b_hr
            
            return q


        # ==========================================================
        # 1. Generate Simulated High-Res & Low-Res Image Pair
        # ==========================================================

        # Create an elegant high-resolution Guide (512x512)
        # We will draw a clean geometric circle pattern to act as the sharp HR structure
        hr_guide = np.ones((512, 512), dtype=np.uint8) * 40
        cv2.circle(hr_guide, (256, 256), 140, 220, -1)
        cv2.putText(hr_guide, "SHARP GUIDE", (100, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 40, 4, cv2.LINE_AA)

        # Generate a degraded Low-Resolution Target (128x128)
        # We downsample it, add severe blur, and inject Gaussian noise
        lr_clean = cv2.resize(hr_guide, (128, 128), interpolation=cv2.INTER_AREA)
        lr_blurred = cv2.GaussianBlur(lr_clean, (11, 11), 0)
        noise = np.random.normal(0, 15, lr_blurred.shape).astype(np.float32)
        lr_target = np.clip(lr_blurred.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # ==========================================================
        # 2. Run the Fast Guided Filter
        # ==========================================================
        # radius = 16 pixels relative to the HR image
        # eps = regularization factor. Since image range is [0-255], we scale eps by 255^2
        eps = 0.02 * (255 ** 2)
        s = 4  # Scale factor (512 / 128)

        upsampled_output = fast_guided_filter(p=lr_target, I=hr_guide, r=16, eps=eps, s=s)
        upsampled_output = np.clip(upsampled_output, 0, 255).astype(np.uint8)

        # For baseline comparison, let's also do a standard Bilinear upsampling of the LR image
        bilinear_baseline = cv2.resize(lr_target, (512, 512), interpolation=cv2.INTER_LINEAR)

        # ==========================================================
        # 3. Display Everything Side-by-Side using Matplotlib
        # ==========================================================
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=100)

        # 1. Low-Resolution Target
        axes[0].imshow(lr_target, cmap='gray')
        axes[0].set_title(f"1. Low-Res Target\n({lr_target.shape[1]}x{lr_target.shape[0]} + Noise/Blur)")
        axes[0].axis('off')

        # 2. High-Resolution Guide
        axes[1].imshow(hr_guide, cmap='gray')
        axes[1].set_title(f"2. High-Res Guide\n({hr_guide.shape[1]}x{hr_guide.shape[0]})")
        axes[1].axis('off')

        # 3. Standard Bilinear Baseline (Control Group)
        axes[2].imshow(bilinear_baseline, cmap='gray')
        axes[2].set_title("3. Naive Bilinear Upsample\n(Edge is blurry & noisy)")
        axes[2].axis('off')

        # 4. Joint Guided Upsampled Output
        axes[3].imshow(upsampled_output, cmap='gray')
        axes[3].set_title("4. Guided Filter Output\n(Sharp edges, smooth surface)")
        axes[3].axis('off')

        plt.tight_layout()
        plt.show()
    
    def test_joint_bilateral_upsampling(self):
        "validation test for joint bilateral upsampling"
        import cv2

        p = ShazamDepthEstimator()

        # =====================================================================
        # 1. High-Performance NumPy JBU Implementation
        # =====================================================================
        def joint_bilateral_upsampling(lr_img, hr_guide, spatial_sigma=3.0, range_sigma=0.1, radius=4):
            if lr_img.ndim == 2:
                lr_img = lr_img[..., np.newaxis]
            if hr_guide.ndim == 2:
                hr_guide = hr_guide[..., np.newaxis]
                
            H_lr, W_lr, C_lr = lr_img.shape
            H_hr, W_hr, C_g = hr_guide.shape
            
            scale_y = H_hr / H_lr
            scale_x = W_hr / W_lr
            
            # Precompute spatial Gaussian weights
            y_coords, x_coords = np.mgrid[-radius:radius+1, -radius:radius+1]
            spatial_dist_sq = y_coords**2 + x_coords**2
            spatial_weights = np.exp(-spatial_dist_sq / (2 * spatial_sigma**2))
            
            upsampled = np.zeros_like(hr_guide, dtype=np.float32) if C_lr == C_g else np.zeros((H_hr, W_hr, C_lr), dtype=np.float32)
            norm_factor = np.zeros((H_hr, W_hr, 1), dtype=np.float32)
            
            # Neighborhood search loop
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    s_w = spatial_weights[dy + radius, dx + radius]
                    if s_w < 1e-4:
                        continue
                        
                    hr_y, hr_x = np.meshgrid(np.arange(H_hr), np.arange(W_hr), indexing='ij')
                    
                    lr_y_idx = np.clip(np.round(hr_y / scale_y).astype(np.int32) + dy, 0, H_lr - 1)
                    lr_x_idx = np.clip(np.round(hr_x / scale_x).astype(np.int32) + dx, 0, W_lr - 1)
                    
                    lr_val = lr_img[lr_y_idx, lr_x_idx]
                    
                    hr_y_from_lr = np.clip(np.round(lr_y_idx * scale_y).astype(np.int32), 0, H_hr - 1)
                    hr_x_from_lr = np.clip(np.round(lr_x_idx * scale_x).astype(np.int32), 0, W_hr - 1)
                    
                    guide_diff = hr_guide - hr_guide[hr_y_from_lr, hr_x_from_lr]
                    range_dist_sq = np.sum(guide_diff**2, axis=-1, keepdims=True)
                    range_weights = np.exp(-range_dist_sq / (2 * range_sigma**2))
                    
                    weight = s_w * range_weights
                    upsampled += lr_val * weight
                    norm_factor += weight
                    
            upsampled /= (norm_factor + 1e-8)
            return np.squeeze(upsampled)


        # =====================================================================
        # 2. Setup Synthetic Scenario (Simulating a Depth Map Upsampling task)
        # =====================================================================

        # Step A: High-Resolution Guide (512x512) - A sharp diagonal boundary
        hr_guide = np.zeros((512, 512), dtype=np.float32)
        for i in range(512):
            hr_guide[i, :512-i] = 100.0  # Diagonal boundary transition
        hr_guide[200:300, 200:300] = 200.0  # Add a bright square in the bottom-right quadrant

        # Step B: Low-Resolution Target (64x64) representing downscaled noisy depth
        lr_clean = cv2.resize(hr_guide, (64, 64), interpolation=cv2.INTER_AREA)
        # Add massive Gaussian Blur & Salt-and-Pepper style noise to simulate a cheap depth camera
        lr_blurred = cv2.GaussianBlur(lr_clean, (7, 7), 0)
        noise = np.random.normal(0, 8.15, lr_blurred.shape).astype(np.float32)
        lr_target = np.clip(lr_blurred + noise, 0, 255)

        # Step C: Run JBU 
        hr_guide = hr_guide/100
        # Use a narrow range_sigma to enforce alignment to the high-res guide's sharp edges
        jbu_output = p.joint_bilateral_upsampling(
            lr_img=lr_target, 
            hr_guide=hr_guide, 
            spatial_sigma=4.0, 
            range_sigma=0.1,
            radius=4
        )

        # Step D: Standard Bilinear Baseline (for contrast)
        bilinear_output = cv2.resize(lr_target, (512, 512), interpolation=cv2.INTER_LINEAR)


        # =====================================================================
        # 3. Matplotlib Comparative Layout
        # =====================================================================
        fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))

        # Plot 1: Low-Res Target Inputs
        axes[0].imshow(lr_target, cmap='inferno', extent=[0, 512, 512, 0])
        axes[0].set_title("1. Low-Res Target\n(64x64 + Severe Noise & Blur)", fontsize=12)
        axes[0].axis('off')

        # Plot 2: High-Res Guide
        axes[1].imshow(hr_guide, cmap='gray')
        axes[1].set_title("2. High-Res Guide\n(512x512 Structural Boundaries)", fontsize=12)
        axes[1].axis('off')

        # Plot 3: Naive Bilinear Upsampling
        axes[2].imshow(bilinear_output, cmap='inferno')
        axes[2].set_title("3. Naive Bilinear Upsample\n(Jagged, blurry, and noisy)", fontsize=12)
        axes[2].axis('off')

        # Plot 4: JBU Output
        axes[3].imshow(jbu_output, cmap='inferno')
        axes[3].set_title("4. Joint Bilateral Upsample\n(Crisp edges & denoised flat areas)", fontsize=12)
        axes[3].axis('off')

        plt.tight_layout()
        plt.show()

    def test_pixel_image_disparity(self):
        "compute row-wise left/right gabor channel correlation at multiple levels and extract disparity information to get a disparity map. shows the correlation MxM matrix. integrates information from coars to fine levels"
        ""
        
        d               = DataSource()
        ret             = d.init_image(21) # 4-ok,7-ok,11-nok,26-ok, 54-chair,55-office far,56-office-chess-ok, ,62,66-nok, 71-home, 601-ok, 621,622-mbox
        d.show_images_left_right()
        img_left, img_right = d.imgL, d.imgR

        debug_row       = 120 #140   # set to None to disable per-row debug plots

        p               = ShazamDepthEstimator()
        prob            = p.pixel_image_disparity(img_left, img_right)

        #img_list        = [prob[:,:,debug_row+m] for m in [0, 5, 10]]
        img_list        = [prob[:,debug_row+m,:].squeeze().T for m in [0, 5, 10]]
        ttl_list        = [f'Level {m} Probability Volume (row {debug_row+m})' for m in [0, 5, 10]]
        p.show_subset(img_list, ttl_list, col_num=1)          
        
        return True  

    def test_multiscale_disparity(self):
        "compute row-wise left/right gabor channel correlation at multiple levels and extract disparity information to get a disparity map. shows the correlation MxM matrix. integrates information from coars to fine levels"
        ""
        
        d               = DataSource()
        # 4-ok,7-ok,11-nok,21-sim,26-ok, 54-chair, 55-office far,56-office-chess-ok, ,62,66-nok, 71-home, 601-ok, 621,622,623-mbox
        # 181,182,183,184-pickle
        ret             = d.init_image(184) 
        d.show_images_left_right()
        d.show_images_depth()
        img_left, img_right = d.imgL, d.imgR

        debug_row       = 400 #140   # set to None to disable per-row debug plots

        p               = ShazamDepthEstimator()
        #prob            = p.gabor_image_disparity_down_up(img_left, img_right, debug_row=debug_row)
        #prob            = p.gabor_image_disparity_down_up_on_volume(img_left, img_right, debug_row=debug_row)
        #prob            = p.multiscale_disparity(img_left, img_right, debug_row=debug_row)
        #prob            = p.multiscale_disparity_pixel_features(img_left, img_right, debug_row=debug_row)
        #prob            = p.multiscale_disparity_with_energy(img_left, img_right, debug_row=debug_row)
        prob            = p.multiscale_disparity_edge_aware(img_left, img_right, debug_row=debug_row)
        
        
        return True
# ---------------------------------------------------
#%% App
class RunApp:

    def __init__(self, video_src = 'd16', frame_size = (1280,720)):

        #self.cap = video.create_capture(video_src)
        self.cap            = RealSense(video_src, frame_size = frame_size)
        self.cap.set_display_mode('d16')  # l,r and d
        self.cap.set_exposure(16000)
        #self.cap.switch_disparity()

        _, self.frame       = self.cap.read()
        frame_gray          = self.frame[:,:,0]
        vis                 = cv.cvtColor(frame_gray, cv.COLOR_GRAY2BGR) 
        self.imshow_name    = 'Depth (d-Display, t-Estimator, e-Exposure, p-Projector, w-Range Min, r-Range Max, s-Show, space-Pause, q-Quit)'
        cv.imshow(self.imshow_name, vis)
        self.rect_sel       = RectSelector(self.imshow_name, self.on_rect)
        self.trackers       = []
        self.paused         = False
        self.update_rate    = 0 
        self.estim_type     = 'F'  # 'T' - time, 'P' - plane , 'F' fill rate
        self.frame_count     = 0      # frame counter  
        self.control_mode   = 'no controls'  
        self.camera_bf      = self.cap.get_bf()  # just camera bf
        self.display_mode   = 0
        #self.depth_rs_noise = NoiseEstimator()
        #self.depth_fft_noise = NoiseEstimator()
        self.output_range   = [0,2**12]  # extract range to map 16 bit to 8
        self.show_pixel_values = False

        self.run()
 
    def switch_estimator(self, ch = None):
        "gets the next estimator"      
        if ch is None:  
            self.estim_type = 'T'
        elif ch == 1:
            self.estim_type = 'T' # time
        elif ch == 2:    
            self.estim_type = 'P'   # plane                    
        else:
            self.estim_type = 'T' # nothing

        log.info(f'Estimator type {self.estim_type} is enabled')              

    def switch_show(self, ch):
        "show something"      
        if ch is None:  
            pass
        elif ch == 1:
            for tracker in self.trackers:
                tracker.debug_on = True 
        elif ch == 2:     
            self.show_pixel_values = not self.show_pixel_values    
        else:
            pass

        log.info(f'Show value : {ch}')     

    def on_rect(self, rect):
        estim_ind           = len(self.trackers) + 1
        tracker             = ShazamDepthEstimator(noise_type = self.estim_type) #estimator_type=self.estim_type, estimator_id=estim_ind)
        tracker.rect        = rect
        tracker.debug_show  = False
        tracker.camera_bf   = self.camera_bf
        self.trackers.append(tracker)

    def scale_imges(self, img):
        "scales images to the range 0-255"
        scale_factor    = 255.0 / (self.output_range[1] - self.output_range[0])
        img_scaled      = scale_factor*(img.astype(np.float32) - self.output_range[0])
        #img_scaled      = cv.convertScaleAbs(img, alpha=scale_factor, beta=-self.output_range[0])
        img_scaled      = np.clip(img_scaled, 0, 255)  # ensure values are in the range 0-255
        img_scaled      = img_scaled.astype(np.uint8)  # convert to uint8
        return img_scaled

    def create_output_image(self, depth_image, depth_image_new, irl_image, irr_image, img_disp_rs, img_disp_fft):
        "defines the output image"

        if self.display_mode == 1:
            image_out       = irl_image
        elif self.display_mode == 2:
            image_out       = irr_image       

        elif self.display_mode == 3:

            depth_scaled    = cv.convertScaleAbs(depth_image, alpha=0.06)
            image_out       = cv.applyColorMap(depth_scaled, cv.COLORMAP_JET)            

        elif self.display_mode == 4:
            depth_scaled    = cv.convertScaleAbs(depth_image_new, alpha=0.06)
            image_out       = cv.applyColorMap(depth_scaled, cv.COLORMAP_JET) 

        elif self.display_mode == 5:
            depth_scaled    = self.scale_imges(depth_image)
            image_out       = cv.applyColorMap(depth_scaled, cv.COLORMAP_JET)    

        elif self.display_mode == 6:
            depth_scaled    = self.scale_imges(depth_image_new)
            image_out       = cv.applyColorMap(depth_scaled, cv.COLORMAP_JET)  

        elif self.display_mode == 7 and (img_disp_rs is not None):
            image_out       = cv.applyColorMap(img_disp_rs.astype(np.uint8), cv.COLORMAP_JET)  

        elif self.display_mode == 8 and (img_disp_fft is not None):
            image_out       = cv.applyColorMap(img_disp_fft.astype(np.uint8), cv.COLORMAP_JET)              

        else:
            image_out       = irl_image

        image_out           = image_out.astype(np.uint8)  
        if len(image_out.shape) < 3:
            image_out         = cv.cvtColor(image_out, cv.COLOR_GRAY2BGR)    
        #log.info(f'Display mode : {self.display_mode}')
        return image_out   

    def set_controls(self, value_in = 0):
        "implements differnt controls according to the selected control mode. Input is an integer from 0-9"
        if self.control_mode == 'display':
            self.display_mode = value_in

        elif self.control_mode == 'exposure':
            self.cap.set_exposure(value_in*1000*2)

        elif self.control_mode == 'estimator':
            self.switch_estimator(value_in)    
        
        elif self.control_mode == 'projector':
            self.cap.use_projector = value_in == 1
            self.cap.switch_projector() 

        elif self.control_mode == 'range':
            self.output_range[0]    = 0  if value_in < 1 else self.output_range[0] + (value_in - 1.5)*2*100
            self.output_range[0]    = np.minimum(self.output_range[1] - 10, self.output_range[0])
            log.info(f'Output range : {self.output_range}')    

        elif self.control_mode == 'width':
            self.output_range[1]    = 2**12 if value_in < 1 else self.output_range[1] - (value_in - 1.5)*2*100
            self.output_range[1]    = np.maximum(self.output_range[0] + 10, self.output_range[1])
            log.info(f'Output range : {self.output_range}') 

        elif self.control_mode == 'show':
            self.switch_show(value_in)     
                                           
                          
        else:
            pass         

    def show_controls(self, frame):
        "show image on opencv window"
        if self.control_mode == 'display':
            frame = cv.putText(frame, 'Display (1-L, 2-R, 3-D-RS, 4-D-FFT, 5-D-RS, 6-D-FFT, 7-Disp-RS, 8-Disp-FFT)', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (200,200,12), 2)

        elif self.control_mode == 'exposure':
            frame = cv.putText(frame, 'Exposure (1-9 x 1000) ', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (10,200,200), 2) 

        elif self.control_mode == 'estimator':
            frame = cv.putText(frame, 'Estimator (1-Time, 2-Plane) ', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (10,200,200), 2)                    

        elif self.control_mode == 'projector':
            frame = cv.putText(frame, 'Projector (0,1) ', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (200,200,200), 2)   

        elif self.control_mode == 'range':
            frame = cv.putText(frame, 'Min Range (0-Zero, 1-, 2+, 3-snap) ', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (200,200,200), 2) 

        elif self.control_mode == 'width':
            frame = cv.putText(frame, 'Max Range (0-Full, 1-, 2+, 3-snap) ', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (200,200,200), 2)  

        elif self.control_mode == 'show':
            frame = cv.putText(frame, 'Show (0-None, 1-Debug On, 2-Pixel in Time) ', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (200,200,50), 2)  

        else:
            pass

        return frame   
    
    def compute_noise(self, img_depth, img_depth_new):
        "compute noise on the depth images"
        # computes noise for both images 
        isOk = False
        for tracker in self.trackers:
            isOk  = tracker.compute_noise(img_depth , img_depth_new)        

        return isOk

    def show_trackers(self, vis, img_depth, img_depth_new):
        "show trackers on the frame"

        # draw track window and measure noise on original RS depth
        for tracker in self.trackers:
            x1, y1, x2, y2                            = tracker.rect
            rs_mean, rs_std, fft_mean, fft_std        = tracker.get_noise_rs_fft()

            # use image depth original - what to show
            if self.display_mode in [3,5,7]:
                clr, etype, err_mean, err_std  = (0,0,200), 'RS ', rs_mean, rs_std
            else:
                clr, etype, err_mean, err_std  = (200,0,0), 'FFT', fft_mean, fft_std

            vis = cv.rectangle(vis, (x1, y1), (x2, y2), clr)
            vis = cv.putText(vis, f'{etype}:{err_mean:.1f}:{err_std:.2f}', (x1, y1-10), cv.FONT_HERSHEY_PLAIN, 0.9, (0, 0, 0), lineType=cv.LINE_AA)

            # show additional information
            if self.show_pixel_values:
                vis = tracker.show_noise_data(vis) #

        return vis     

    def run(self):
        while True:
            if not self.paused:
                ret, self.frame = self.cap.read()
                if not ret:
                    break

            # compute
            img_left, img_right, img_depth, img_depth_new = self.frame[:,:,0], self.frame[:,:,1], self.frame[:,:,2], self.frame[:,:,2]
            img_disp_rs, img_disp_fft = None, None

            for tracker in self.trackers:
                img_depth_new               = tracker.convert_depth_from_left_and_right(img_left, img_right, img_depth) 

            # compute noise
            #isOk                        = self.compute_noise(img_depth, img_depth_new)
            
            # draw on the main window
            vis                         = self.create_output_image(img_depth, img_depth_new, img_left, img_right, img_disp_rs, img_disp_fft) 

            # draw trackers
            #vis                         = self.show_trackers(vis, img_depth, img_depth_new)

            self.rect_sel.draw(vis)    # draw rectangle
            vis                         = self.show_controls(vis)  # draw controls

            cv.imshow(self.imshow_name, vis)
            ch = cv.waitKey(3)
            if ch == 27 or ch == ord('q'):
                break  
            elif ch == ord(' '):
                self.paused = not self.paused              
            elif ch in np.arange(48,58) : # numbers only
                self.set_controls(ch - 48)
            elif ch == ord('d'): # depth image
                self.control_mode = 'no controls' if self.control_mode == 'display' else 'display'    
            elif ch == ord('e'): # exposure control
                self.control_mode = 'no controls' if self.control_mode == 'exposure' else 'exposure'      
            elif ch == ord('t'):
                self.control_mode = 'no controls' if self.control_mode == 'estimator' else 'estimator'             
            elif ch == ord('p'):
                self.control_mode = 'no controls' if self.control_mode == 'projector' else 'projector' 
            elif ch == ord('w'): 
                self.control_mode = 'no controls' if self.control_mode == 'range' else 'range'
            elif ch == ord('r'): 
                self.control_mode = 'no controls' if self.control_mode == 'width' else 'width'     
            elif ch == ord('s'): 
                self.control_mode = 'no controls' if self.control_mode == 'show' else 'show'                                                
            elif ch == ord('c'):
                if len(self.trackers) > 0:
                    t = self.trackers.pop()
                    
            self.frame_count += 1

        log.info('Finished')
        self.cap.release()
        cv.destroyAllWindows()                

# ----------------------------------------------------
#%% Run Test
def RunTest():
    "Run all tests in the TestShazamDepthEstimator class"
    tst = TestShazamDepthEstimator()
    #tst.test_convert_disparity_to_depth() # ok
    #tst.test_show_images_left_right() # ok
    #tst.test_show_images_depth() # ok    
    #tst.test_gabor_bank_channels() # ok
    #tst.test_gabor_line_correlation() # ok
    #tst.test_gabor_line_correlation_multiscale() # ojk
    #tst.test_gabor_multiline_correlation_multiscale()
    #tst.test_gabor_bank_rotated() # ok
    #tst.test_gabor_line_correlation_multiscale_floor()
    #tst.test_gabor_line_correlation_updown()

    #tst.test_gabor_image_disparity_multiscale()
    #tst.test_gabor_image_disparity_down_up()
    tst.test_multiscale_disparity()

    #tst.test_context_upsampling()
    #tst.test_context_upsampling_using_ai()
    #tst.test_context_upsampling_validation()
    #tst.test_gabor_bank_upsampling()
    #tst.test_down_upsampling_consistency() # ok
    #tst.test_edge_preserving_methods()
    #tst.test_kalman_filtering()
    #tst.test_kalman_filtering_two_images()
    #tst.test_grid_interpolation()
    #tst.test_fast_guided_filter()
    #tst.test_joint_bilateral_upsampling()

    #tst.test_pixel_image_disparity()


if __name__ == '__main__':
    #print(__doc__)

    RunTest()
    #RunApp(frame_size = (640,360))      

