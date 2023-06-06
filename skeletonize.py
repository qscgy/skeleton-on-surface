import matplotlib.pyplot as plt
import matplotlib as mpl
from skimage.transform import rescale
from typing import Tuple
import matplotlib.image as mpimg
import math
import numpy as np
import scipy.ndimage as sn
import os
import skimage
from skimage import filters, transform, morphology, segmentation
import cv2
import torch
import time
import FastGeodis
import networkx as nx
import torch.nn.functional as F
import natsort

def boundary_loop(im):
    """Find the un-oriented traversal of the largest cycle in the boundary.
    Boundary is a graph defined by the 8-connectvity of a binary image,
    where boundary pixels are 1.

    Parameters
    ----------
        im : ndarray
            The boundary as a binary image. Cycles are computed from 8-connected pixels equal to 1.

    Returns
    -------
        path : ndarray
            The pixel indices of the longest cycle in the boundary, in order. Orientation
            (cw or ccw) is not yet defined.
    """
    adj, nodes = skimage.graph.pixel_graph(im, mask=im, connectivity=2)
    graph = nx.from_numpy_array(adj).to_directed()
    cycles = nx.simple_cycles(graph)
    by_len = sorted(list(cycles), key=lambda c: len(c))
    node_inds = np.vstack(np.unravel_index(nodes, im.shape)).T
    if len(node_inds>1):
        path = node_inds[by_len[-1]]
        return path
    else:
        return None

def voronoi_2d(image, mask_pts, n, l=1.0)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the Voronoi cells of a region in a depth image.
    
    Parameters
    ----------
        image : Tensor
            N x M tensor representing the depth map
        mask_pts : list of Tensor
            Coordinates in the image corresponding to the boundary
        n : int
            number of sample points to use for Voronoi computation, or 0 for all
    
    Returns
    -------
        voronoi : Tensor
            image labeled by Voronoi cells, starting from 1, as int
        dist_map : Tensor
            minimum distance to boundary of each point (i.e. geodesic distance transform)
        pts : Tensor
            tensor of the indices of the closest point to each image index, as longs
    """
    # This will return a labeled Voronoi cell map. It still needs some method of pruning
    # Idea: use alignment to tangent field as a saliency measure
    if n>0:
        idx = torch.randperm(len(mask_pts))[:n]
        pts = mask_pts[idx]
    else:
        pts = mask_pts
        n = len(mask_pts)

    pt_diff = pts-torch.mean(pts.float(), axis=0)
    pts_tan = torch.cat([pts, torch.atan2(pt_diff[:,-2], pt_diff[:,-1]).unsqueeze(1)], axis=1)
    _, inds = torch.sort(pts_tan[:,-1])

    geodesic_dists = torch.zeros((n, image.shape[-2], image.shape[-1]))
    
    for i in range(n):
        mask = torch.ones_like(image)
        mask[...,pts[i,-2],pts[i,-1]] = 0
        dist = FastGeodis.generalised_geodesic2d_toivanen(
            image, mask, 1e10, l, 2
        )
        geodesic_dists[i] = dist[0,0]
    voronoi = torch.argmin(geodesic_dists, 0)+1
    
    return voronoi.int(), torch.min(geodesic_dists, dim=0).values, pts.long()

def gauusian_curvature(image):
    # Following Kurita and Boulanger (1992), https://www.mva-org.jp/Proceedings/CommemorativeDVD/1992/papers/1992389.pdf
    # x is dimension 0, y is dimension 1
    image_ = image-image.min()
    image_ = image_/np.max(image_)
    hx = filters.sobel_v(image_)/8.
    hy = filters.sobel_h(image_)/8.
    from skimage.feature.corner import hessian_matrix
    hyy, hxy, hxx = hessian_matrix(image_, sigma=3)
    gaussian_k = (hxx * hyy - hxy**2)/((1 + hx**2 + hy**2)**2)
    return gaussian_k

def geodesic_skeleton(image, p, g=15, device='cpu', plot=False):
    """Compute the geodesic skeleton.

    Args:
        image (ndarray): labeled connected component pixels of the frame
        p (int): index of the pixels comprising the fold to be skeletonized
        g (int): granularity
        device (str, optional): Tensor device. Defaults to 'cpu'.
        plot (bool, optional): Whether to display plots of the skeleton. Defaults to False.

    Returns:
        void
    """
    I = labels==p   # type: np.ndarray  # change this to change which fold in image is used
    cc = morphology.binary_erosion(np.pad(I, ((1,1),(1,1))))[1:-1,1:-1] ^ I

    image_pt = torch.from_numpy(image).unsqueeze_(0).unsqueeze_(0)
    image_pt = image_pt.to(device)
    cycle = boundary_loop(cc)
    if cycle is None:
        return None
    
    dm = "Geodesic"     # distance metric, Geodesic or Euclidean
    
    cycle = torch.from_numpy(cycle).to(device)
    pt_coords = torch.hstack((torch.zeros_like(cycle), cycle))  # coords of boundary points of cycle
    voronoi, distImage, pts = voronoi_2d(image_pt, pt_coords, 0, l=1.0 if dm=='Geodesic' else 0.0)
    voronoi *= I
    distImage *= I
    
    # v_bounds = segmentation.mark_boundaries(np.zeros(voronoi.shape), voronoi.numpy(), mode='subpixel', background_label=0)

    # skeleton with granularity and pooling instead of AOF
    voronoi_inf = voronoi.clone()
    voronoi_inf[voronoi_inf==0] = -(voronoi_inf.max()*10)   # do this so pooling ignores pts outside region
    voronoi_neg_inf = voronoi.clone()
    voronoi_neg_inf[voronoi_neg_inf==0] = voronoi_neg_inf.max()*10
    
    # Apply convolutional min- and max-pooling in order to prune, as explained in my report.
    skel_max = F.max_pool2d(voronoi_inf[None,...].float(), kernel_size=2, stride=1)
    skel_min = -F.max_pool2d(-voronoi_neg_inf[None,...].float(), kernel_size=2, stride=1)
    skel_granular = torch.minimum(skel_max-skel_min, skel_min+voronoi.max()-skel_max)
    
    skel = (skel_granular >= g)
    
    if plot:
        plot_pipeline(image=image,
            I=I,
            cc=cc,
            distImage=distImage,
            skeleton=skel,
            voronoi=voronoi,
            dm=dm,
            # fname=f'frame0_{p}_g{g}.png'
        )
    return skel.int()

def plot_pipeline(**kwargs):
    """Plot the skeletonization steps.
    
    **kwargs : Items to plot. Keywords are:
        image (ndarray of int): `image` passed to `geodesic_skeleton`, the labeled connected components
        
        I (ndarray of bool): binary image of fold
        
        cc (ndarray of bool): binary array with fold region boundary pixels marked as 1 and all others 0
        
        distImage (ndarray of float): distance transform distances of region
        
        skel_granular (ndarray of bool): binary image of skeleton, with skeleton pixels as 1
        
        voronoi (ndarray of int): binary image of fold with pixels labeled by Voronoi cell
        
        dm (str): distance metric, Geodesic or Euclidean
        
        fname (str, optional): filename to save plots. If None, plots will be displayed instead.
    """
    image = kwargs['image']
    I = kwargs['I']
    cc = kwargs['cc']
    distImage = kwargs['distImage']
    skel_granular = kwargs['skeleton']
    voronoi = kwargs['voronoi']
    dm = kwargs['dm']
    if 'fname' in kwargs:
        fname = kwargs['fname']
    else:
        fname = None
    
    fig, ax = plt.subplots(2, 3, figsize=(17,9))
    plt.tight_layout()
    plt.subplot(231)
    plt.imshow(image)
    plt.colorbar()
    plt.title("Input depth map")
    
    plt.subplot(232)
    plt.imshow(I, interpolation='none')
    # plt.imshow(gaussian_k[10:-10,10:-10], cmap='Blues')
    plt.colorbar()
    plt.title("Fold region")
    
    plt.subplot(233)
    plt.title("Boundary")
    plt.imshow(cc, interpolation='none')
    
    plt.subplot(234)
    plt.imshow(distImage)
    plt.colorbar()
    plt.title(dm + " distance to boundary")
    
    plt.subplot(235)
    plt.imshow(voronoi, cmap='jet', interpolation='none')
    # plt.imshow(v_bounds)
    plt.colorbar()
    plt.title("Voronoi cells")
    
    plt.subplot(236)
    plt.imshow(skel_granular, interpolation='none')
    plt.title("Skeleton")
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)

if __name__=='__main__':
    # Configuration
    base_dir = "/Users/sam/Documents/UNC"
    plot_sequence = True
    name = 'Auto_A_Aug18_09-06-42_006'  # sequence name
    s = 2   # downscale factor
    device = "cuda" if torch.cuda.is_available() else "cpu" # does not yet work with MPS (Apple Silicon)
    frames = [0, 1, 2, 3, 4, 5]    # numbers of frames to process
    
    # Find paths and file names
    seg_path = os.path.join(base_dir, name, "results-mine/preds.npy")
    depth_dir = os.path.join(base_dir, name, "colon_geo_light/")
    photo_dir = os.path.join(base_dir, name, "image/")
    depth_paths = natsort.natsorted([p for p in os.listdir(depth_dir) if "_disp.npy" in p])
    
    skel_numbers = [1, 2, 3]   # label numbers of folds to process in each frame
    
    # Set up plots
    if plot_sequence:
        fig, ax = plt.subplots(2, len(frames), figsize=(17,9))
        fig.tight_layout()
        
    # Process each frame
    for i, frame in enumerate(frames):
        # Load frame depth and segmentation maps
        depth_path = os.path.join(depth_dir, depth_paths[frame])
        seg = np.load(seg_path)[frame]
        
        # Preprocess segmentation and resize both
        seg = transform.resize(seg, (seg.shape[-2]//s, seg.shape[-1]//s))
        image = 1/np.load(depth_path)
        image = transform.resize(image, seg.shape)
        seg = filters.gaussian(seg, 3)      # apply Gaussian blur
        seg = seg>0.3
        seg = seg.astype(np.uint8)
        nlabel, labels, stats, centroids = cv2.connectedComponentsWithStats(seg)
        
        skeletons = np.zeros((seg.shape[0]-1, seg.shape[1]-1))  # destination arry for skeletons
        for p in skel_numbers:     # process each fold in skel_numbers
            skel = geodesic_skeleton(image, p, g=12, device=device, plot=False)    # set plot=True to plot the whole pipeline
            if skel is None:
                continue
            skel = skel.numpy()
            skeletons = skeletons + skel*p
        skeletons = np.squeeze(skeletons)

        # Plot a sequence of frames with their skeletal features
        if plot_sequence:        
            skeletons = np.pad(skeletons, ((0,1),(0,1)))

            skeletons = skeletons.astype(int)
            frame_files = natsort.natsorted(os.listdir(photo_dir))
            photo = skimage.io.imread(os.path.join(photo_dir, frame_files[frame]))
            photo = transform.resize(photo, skeletons.shape)
            photo_folds = photo.copy()
            photo_folds[seg==1,0] = 0   # mark fold segmentations in green
            
            plt.subplot(2,len(frames),i+1)
            plt.imshow(photo_folds)
            plt.axis("off")
            plt.title(f"Frame {frame+1}")
            
            viridis = mpl.colormaps['viridis']
            colors = np.array([viridis(n/np.max(skeletons)) for n in np.unique(skeletons)])[:,:3]
            photo[skeletons>0,:] = colors[skeletons[skeletons>0]]
            plt.subplot(2,len(frames),i+len(frames)+1)
            plt.imshow(photo)
            plt.axis("off")
    if plot_sequence:
        plt.show()