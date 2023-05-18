#!/usr/bin/env python
# coding: utf-8

# # 2D AOF Skeleton
#This is a jupyter notebook for 2D AOF Skeletonization code

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

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

fileName = "horse.png"

# plt.imshow(distImage)
# plt.show()

def sample_sphere_2D(number_of_samples):
    sphere_points = np.zeros((number_of_samples,2))
    alpha = (2*math.pi)/(number_of_samples)
    for i in range(number_of_samples):
        sphere_points[i][0] = math.cos(alpha*(i-1))
        sphere_points[i][1] = math.sin(alpha*(i-1))
    return sphere_points

def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind

def ind2sub(array_shape, ind):
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = (ind.astype('int') / array_shape[1])
    cols = ind % array_shape[1]
    return (rows, cols)


def compute_aof(distImage, IDX, sphere_points, epsilon, number_of_samples=60):
    m = distImage.shape[0]
    n = distImage.shape[1]
    normals = np.zeros(sphere_points.shape)
    fluxImage = np.zeros((m,n))
    for t in range(0,number_of_samples):
        normals[t] = sphere_points[t]
    sphere_points = sphere_points * epsilon
    
    XInds = IDX[0]
    YInds = IDX[1]
    
    for i in range(0,m):
        # print(i)
        for j in range(0,n):       
            flux_value = 0
            if (distImage[i][j] > -1.5):
                if( i > epsilon and j > epsilon and i < m - epsilon and j < n - epsilon ):
#                   sum over dot product of normal and the gradient vector field (q-dot)
                    for ind in range (0,number_of_samples):
                                                
#                       a point on the sphere
                        px = i+sphere_points[ind][0]+0.5
                        py = j+sphere_points[ind][1]+0.5                      
#                       the indices of the grid cell that sphere points fall into 
                        cI = math.floor(i+sphere_points[ind][0]+0.5)
                        cJ = math.floor(j+sphere_points[ind][1]+0.5)
                                               
                        # calculate flux on surface
                        # I think the grid cell is selected based on the exponential map of the sphere pt
                        # Still need a way to estimate flux
                        # Don't know the direction of the geodesic to the closest boundary pt
                        
                        # Shooting algorithm: run forward on manifold in different directions
                        # for distance to boundary phi(x) from point x
                        # direction whose end pt has lowest value of phi is the flux direction
                        # idk how accurate or fast this would be
                        
                        # improved idea: breadth-first traversal of 8-connected graph to distance phi(x)
                        
                        # given the feature points (closest point on boundary), can use path straightening
                        
                        # alternatively, we abandon AOF and find a skeleton via Voronoi + MGF

#                       closest point on the boundary to that sphere point
                        bx = XInds[cI][cJ]
                        by = YInds[cI][cJ]
#                       the vector connect them
                        qq = [bx-px,by-py]
                    
                        d = np.linalg.norm(qq)
                        if(d!=0):
                            qq = qq / d
                        else:
                            qq = [0,0]                        
                        flux_value = flux_value + np.dot(qq,normals[ind])
            fluxImage[i][j] = flux_value  
    return fluxImage

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

def geodesic_skeleton(image, p, device='cpu', plot=False):
    I = labels==p   # type: np.ndarray  # change this to change which fold in image is used
    cc = morphology.binary_erosion(np.pad(I, ((1,1),(1,1))))[1:-1,1:-1] ^ I

    image_pt = torch.from_numpy(image).unsqueeze_(0).unsqueeze_(0)
    image_pt = image_pt.to(device)
    cycle = boundary_loop(cc)
    if cycle is None:
        return None
    
    dm = "Geodesic"
    
    cycle = torch.from_numpy(cycle).to(device)
    pt_coords = torch.hstack((torch.zeros_like(cycle), cycle))
    voronoi, distImage, pts = voronoi_2d(image_pt, pt_coords, 0, l=1.0 if dm=='Geodesic' else 0.0)
    voronoi *= I
    distImage *= I
    IDX = np.zeros((2, voronoi.shape[-2], voronoi.shape[-1]))
    IDX[0] = pts[voronoi.reshape(-1,1)-1, 2].reshape(voronoi.shape[-2], voronoi.shape[-1])
    IDX[1] = pts[voronoi.reshape(-1,1)-1, 3].reshape(voronoi.shape[-2], voronoi.shape[-1])
    
    v_bounds = segmentation.mark_boundaries(np.zeros(voronoi.shape), voronoi.numpy(), mode='subpixel', background_label=0)

    # skeleton with granularity and pooling instead of AOF
    voronoi_inf = voronoi.clone()
    voronoi_inf[voronoi_inf==0] = -(voronoi_inf.max()*10)   # do this so pooling ignores pts outside region
    voronoi_neg_inf = voronoi.clone()
    voronoi_neg_inf[voronoi_neg_inf==0] = voronoi_neg_inf.max()*10
    skel_max = F.max_pool2d(voronoi_inf[None,...].float(), kernel_size=2, stride=1)
    skel_min = -F.max_pool2d(-voronoi_neg_inf[None,...].float(), kernel_size=2, stride=1)
    skel_granular = torch.minimum(skel_max-skel_min, skel_min+voronoi.max()-skel_max)
    
    plt.subplots(1,3,figsize=(12,4))
    plt.tight_layout()
    for i, g in enumerate([10, 15, 30]):
        skel = skel_granular >= g
        skel = skel.squeeze()
        plt.subplot(1,3,i+1)
        plt.title(f"Granularity {g}")
        plt.imshow(skel)
        plt.savefig(f'frame0_{p}_granular.png')
    
    # sphere_points = sample_sphere_2D(60)
    # fluxImage = compute_aof(distImage, IDX, sphere_points, 1.0)
    # flux_thresh = 5
    # skel_aof = fluxImage >= flux_thresh
    
    if plot:
        plot_pipeline(image=image,
            I=I,
            cc=cc,
            distImage=distImage,
            skeleton=skel,
            voronoi=voronoi,
            dm=dm,
            fname=f'frame0_{p}_g{g}.png'
        )
    return skel_granular.int()

def plot_pipeline(**kwargs):
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
    name = 'Auto_A_Aug18_09-06-42_006'
    frame = 2
    seg_path = os.path.join("/Users/sam/Documents/UNC", name, "results-mine/preds.npy")
    depth_dir = os.path.join( "/Users/sam/Documents/UNC", name, "colon_geo_light/")
    photo_dir = os.path.join( "/Users/sam/Documents/UNC", name, "image/")
    depth_paths = natsort.natsorted([p for p in os.listdir(depth_dir) if "_disp.npy" in p])
    s = 2   # downscale factor
    device = "cuda" if torch.cuda.is_available() else "cpu" # does not yet work with MPS (Apple Silicon)
    
    for frame in range(1):
        depth_path = os.path.join(depth_dir, depth_paths[frame])
        seg = np.load(seg_path)[frame]
        seg = transform.resize(seg, (seg.shape[-2]//s, seg.shape[-1]//s))
        image = 1/np.load(depth_path)
        image = transform.resize(image, seg.shape)
        seg = filters.gaussian(seg, 3)
        seg = seg>0.3
        seg = seg.astype(np.uint8)
        nlabel, labels, stats, centroids = cv2.connectedComponentsWithStats(seg)
        
        skeletons = np.zeros((seg.shape[0]-1, seg.shape[1]-1))
        for p in [1,2]:
            skel = geodesic_skeleton(image, p, device, plot=False)
            if skel is None:
                continue
            skel = skel.numpy()
            skeletons = skeletons + skel*p
        # skeletons = np.pad(skeletons, ((0,1),(0,1)))

        # skeletons = skeletons.astype(int)
        # frames = natsort.natsorted(os.listdir(photo_dir))
        # photo = skimage.io.imread(os.path.join(photo_dir, frames[frame]))
        # photo = transform.resize(photo, skeletons.shape)
        # photo_folds = photo.copy()
        # photo_folds[seg==1,0] = 0
        # fig, ax = plt.subplots(2, 3, figsize=(17,9))
        # plt.subplot(2,3,frame+1)
        # plt.imshow(photo_folds)
        # plt.axis("off")
        # plt.title(f"Frame {frame+1}")
        
        # viridis = mpl.colormaps['viridis']
        # colors = np.array([viridis(n/np.max(skeletons)) for n in np.unique(skeletons)])[:,:3]
        # photo[skeletons>0,:] = colors[skeletons[skeletons>0]]
        # plt.subplot(2,3,frame+4)
        # plt.imshow(photo)
        # plt.axis("off")
    
    # plt.show()