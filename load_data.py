import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt
from math import sin, cos, atan2, sqrt
import os
import cv2  # Import OpenCV
import tqdm
MATRIX_MATCH_TOLERANCE = 1e-4
from models.fields import NeRF
from models.mbes_renderer import MBESRenderer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF

# TODO:
# 1. use normalized direction vectors: DONE 
# 2. doubble check the normalization of everthing: DONE
# 3. check the normalization of the intensity: DONE
# add plotting of column data locally Done
# add plotting of the data globally Done
# 4. add noise filtering 
#   - alt 1: assume a gaussian intensity distribution of each ray, it should peak at the bottom of the sea
#   - alt 2: remove half of the ray range bins
# 5. sample the rays from the bins
# 6. use adaptive sampling to make the model learn faster, will need changes to the rendering
        
def load_data_naive(target="post_pings_400khz.npz",depth_threshold=2.0):
    """
    Loads data from a .npz containing MBES data and returns xyz_dxdydz_word_frame and intensity as lists of torch tensors.
    
    It has shape [N_pings X Rays X Samples_Along_Ray X 6 (x,y,z,dx,dy,dz)] and [N_pings X Rays X Samples_Along_Ray X 1 (intensity)] respectively.
    
    Intensity is normalized in the range [0,1] and the spatial data is normalized in the range [-1,1].

    All direction vectors have length 1.

    
    - where `XYZRPY` is an array with shape`(nbr_pings,6)` , in ENU convention. x is UTM easting, y is UTM northing, z is depth — all in meters; r is roll, p is pitch and y is yaw -- all in rads. 
    ISO 8855 earth-fixed system (or ENU: East North Up):
    
    - r_max is the max r_max (in meters), an array with shape `(nbr_pings,)`
        * We assume that the all messurements are at perfect points first at 0 and last at r_max.
            
    - angles is the beam angle (from port to starboard), an array with shape `(nbr_pings,256)`, in rads.
        * We assume that this is the angle in the local frame in roll (around x) and that the ray is infinetly small, naive but simple.

    - we assume that the intensity is associated to each point in the ray.
    """
    data = np.load(target, allow_pickle=True)
    XYZRPY = data['XYZRPY']
    RANGE = data['RANGE']
    ANGELS = data['angles']
    INTENSITY = data['INTENSITY'] # (r,angle_index)
        
    # Transform XYZRPY to SE3
    SE3 = []
    for xyzrpy in XYZRPY:
        SE3.append(build_se3_transform(xyzrpy))
    SE3 = np.array(SE3)

    # data N_pings X N_r_max_bins X N_angle_bins X 4 (x,y,z,intensity)
    # in ENU cartesian coordinates
    data = [] # N_pings X Rays X Samples_Along_Ray X 5 (x,y,z,theta,phi)
    target = [] # N_pings X Rays X Samples_Along_Ray X 1 (intensity)
    
    # We go with that each ray is perfect, no ambiguity in r_max or angle
    # for r_max,angels,intensity,se3 in zip(RANGE,ANGELS,INTENSITY,SE3):
    print(f"Loading data from file: {target}")
    for i in tqdm.tqdm(range(len(RANGE))):
        r_max = RANGE[i]
        angels = ANGELS[i]
        intensity = INTENSITY[i]
        se3 = SE3[i]
        
        
        #
        #   (port)                  (sin)                 (starboard) 
        #   (port)      (y) <--------------------         (starboard)   
        #   (port)                            / |         (starboard)   
        #   (port)                           /  |         (starboard)  
        #   (port)                          /   |         (starboard)     
        #   (port)                         /    | (cos)   (starboard)        
        #   (port)                        /     |         (starboard)        
        #   (port)                       /  30  |         (starboard)       
        #   (port)                     V        |         (starboard)    
        #   (port)                              v         (starboard)          
        #   (port)                             (z)        (starboard)           
        #   (port)                                        (starboard)        

        N = intensity.shape[0]
        r_maxs = np.linspace(0,r_max,N) # Might be wrong, how to achieve a messurment at r_max 0? What is the minimum detectable r_max?

        # # find at what index the depth_threshold is reached
        ## Two alternatives
        # * cut out the values that are below the depth_threshold (computaitonally cheaper)
        # * zero out the values that are below the depth_threshold (more logical) 
        if depth_threshold <= r_max and depth_threshold >= 0:
            depth_index = np.where(r_maxs > depth_threshold)[0][0]
            #  cut of r_maxs, intensity, r_maxs and set new N
            r_maxs = r_maxs[:depth_index]
            intensity = intensity[:depth_index]
            N = len(r_maxs)
            


        xyzI_cartesian_matrix = np.zeros((N,256,4)) 
        
        # x is zero in local frame, the beam goes from port to starboard and heading in x
        xyzI_cartesian_matrix[:,:,1] = r_maxs[:,None]*np.sin(angels)
        xyzI_cartesian_matrix[:,:,2] = -r_maxs[:,None]*np.cos(angels)
        xyzI_cartesian_matrix[:,:,3] = 1.0
        
        # apply se3 to all elements in xyzI_cartesian_matrix
        xyz_word_frame = np.einsum('ij,lmj->lmi',se3,xyzI_cartesian_matrix)

        # cut off the last column
        xyz_word_frame = xyz_word_frame[:,:,:3]        

        # Calculate the direction of the rays
        dirs = xyz_word_frame[1:,:,:] - xyz_word_frame[:-1,:,:]

        # Calculate the mean dirs for each ray
        dirs = np.mean(dirs,axis=0)
        
        # Normalize the mean dirs
        dirs = dirs / np.linalg.norm(dirs,axis=1)[:,None]
        
        
        
        # Move axis in intensity and xyz_word_frame
        intensity = np.moveaxis(intensity,0,1)
        xyz_word_frame = np.moveaxis(xyz_word_frame,0,1)
        
        # dirs: [N_rays,3(=dx,dy,dz)]
        # intensity: [N_rays,N_samples] 
        # xyz_word_frame: [N_rays,N_samples,3(=x,y,z)]
        
        # Concatenate xyz_word_frame and dirs
        dirs_repeated = np.repeat(dirs[:, None, :], xyz_word_frame.shape[1], axis=1)
        xyz_dxdydz_word_frame = np.concatenate([xyz_word_frame,dirs_repeated],axis=2)
        data.append(xyz_dxdydz_word_frame)
        target.append(intensity)

    # Dimensionality of data is [N_pings X Rays X Samples_Along_Ray X 5 (x,y,z,theta,phi)]
    # We fold the data so multiple pings are in the same batch 


    # make target int64 from uint16
    target = [intensity.astype(np.float32) for intensity in target]
    
    # Normalize target in range [0,1]
    max_intensity = np.max([np.max(intensity) for intensity in target])
    target = [intensity / max_intensity for intensity in target]
        
    # Normalize spatial data in range [-1,1]
    x_min = np.min([np.min(xyztp_word_frame[:,:,0]) for xyztp_word_frame in data])
    x_max = np.max([np.max(xyztp_word_frame[:,:,0]) for xyztp_word_frame in data])
    y_min = np.min([np.min(xyztp_word_frame[:,:,1]) for xyztp_word_frame in data])
    y_max = np.max([np.max(xyztp_word_frame[:,:,1]) for xyztp_word_frame in data])
    z_min = np.min([np.min(xyztp_word_frame[:,:,2]) for xyztp_word_frame in data])
    z_max = np.max([np.max(xyztp_word_frame[:,:,2]) for xyztp_word_frame in data])

    max_spatial = np.max([x_max,y_max,z_max])
    min_spatial = np.min([x_min,y_min,z_min])
    
    # Normalize spatial data in range [-1,1] using the max_spatial and min_spatial
    for i in range(len(data)):
        data[i][:,:,0] = (data[i][:,:,0] - min_spatial) / (max_spatial - min_spatial) * 2 - 1
        data[i][:,:,1] = (data[i][:,:,1] - min_spatial) / (max_spatial - min_spatial) * 2 - 1
        data[i][:,:,2] = (data[i][:,:,2] - min_spatial) / (max_spatial - min_spatial) * 2 - 1


    # center the data
    x_min = np.min([np.min(xyztp_word_frame[:,:,0]) for xyztp_word_frame in data])
    x_max = np.max([np.max(xyztp_word_frame[:,:,0]) for xyztp_word_frame in data])
    y_min = np.min([np.min(xyztp_word_frame[:,:,1]) for xyztp_word_frame in data])
    y_max = np.max([np.max(xyztp_word_frame[:,:,1]) for xyztp_word_frame in data])
    z_min = np.min([np.min(xyztp_word_frame[:,:,2]) for xyztp_word_frame in data])
    z_max = np.max([np.max(xyztp_word_frame[:,:,2]) for xyztp_word_frame in data])
    center = np.array([(x_max+x_min)/2,(y_max+y_min)/2,(z_max+z_min)/2])
    for i in range(len(data)):
        data[i][:,:,0] = data[i][:,:,0] - center[0]
        data[i][:,:,1] = data[i][:,:,1] - center[1]
        data[i][:,:,2] = data[i][:,:,2] - center[2]
    
    
    
    # Calculate the dist 
    dists = [xyztp_word_frame[:,1:,:3] - xyztp_word_frame[:,:-1,:3] for xyztp_word_frame in data]
    # Calculate the norm of the dists
    dists = [np.linalg.norm(dist,axis=2) for dist in dists]
    dists = [np.mean(dist) for dist in dists]

    data = [torch.tensor(xyztp_word_frame, dtype=torch.float32) for xyztp_word_frame in data]
    target = [torch.tensor(intensity, dtype=torch.float32) for intensity in target]



    return data, target, dists

    
def ploy_xyz_dxdydz_data(ping_data,render_out,folder_name):
    """takes in data in format xyz dxdydz and plots it"""
    
    # input data has format rays X samples X 6 (x,y,z,dx,dy,dz)
    
    # find the plane spanned by the rays
    origo = ping_data[0,0,:3]
    u = ping_data[0,-1,:3]
    v = ping_data[-1,-1,:3]

    uo = u - origo
    # normalize uo
    uo = uo / np.linalg.norm(uo)

    vo = v - origo
    # normalize vo
    vo = vo / np.linalg.norm(vo)
    
    # find the normal of the plane
    w = np.cross(uo,vo)
    
    # find a normal vector to uo,w
    vo = np.cross(w,uo)
    
    # subtract the origo from all points
    ping_data[:,:,:3] = ping_data[:,:,:3] - origo
    
    # normalize the vectors
    uo = uo / np.linalg.norm(uo)
    vo = vo / np.linalg.norm(vo)
    w = w / np.linalg.norm(w)
    
    # project the points to u0 and v0
    u_proj = np.einsum('abj,j->ab',ping_data[:,:,:3],uo)
    v_proj = np.einsum('abj,j->ab',ping_data[:,:,:3],vo)
    
    # create a data 
    data = np.zeros((ping_data.shape[0],ping_data.shape[1],2))
    data[:,:,0] = u_proj
    data[:,:,1] = v_proj
    
    # find the smallest and largest x or y 
    xy_min = np.min(data[:,:,:2])
    xy_max = np.max(data[:,:,:2])
    
    # normalize the data in the range [0,1]
    data[:,:,:2] = (data[:,:,:] - xy_min) / (xy_max - xy_min)
    
    # scale the xy in data to the range [0,255]
    data[:,:,:] = data[:,:,:] * 255
    
    # make it int, 
    data = data.astype(int)

    # flatten the data shape[0]*shape[1] X 2
    data = data.reshape(-1,2)

    # create a img for every key in render_out
    for key in render_out.keys():
        img = np.zeros((256,256))
        # color value
        color = render_out[key]
        # flatten color
        color = color.reshape(-1)
        


        img[data[:,0],data[:,1]] = color[:]
        plt.imshow(img)
        # add colorbar
        plt.colorbar()
        # add title
        plt.title(key)
        plt.savefig(f"{folder_name}/{key}.png")
        plt.clf()
        



        
                
        
        


def load_data_aucustic_ray(target,visualize=False):
    """
    Loads data from a .npz containing MBES data and returns it as a dictionary.
    
    - where `XYZRPY` is an array with shape`(nbr_pings,6)` , in ENU convention. x is UTM easting, y is UTM northing, z is depth — all in meters; r is roll, p is pitch and y is yaw -- all in rads. 
    ISO 8855 earth-fixed system (or ENU: East North Up):
    
    - r_max is the max r_max (in meters), an array with shape `(nbr_pings,)`
    
    - angles is the beam angle (from port to starboard), an array with shape `(nbr_pings,256)`, in rads.
        * Might be the mean angle of the each beam: 
            - < middle angle of the beam_1 angle> , <middle angle of the beam_2 angle> , ... , <middle angle of the beam_256 angle> 
        * Might be the angle where each beam is starting or ending: (Assume this one for now)
            - |_angle_1_| <beam 1> |_angle_2_| <beam 2> |_anglet_3_| ...  <beam 3> |_angle_256_|

    - INTENSITY is the returned water column intensity (16bit quantization), with shape `(nbr_pings,)`, but every one of the item has shape `(N,256)` — N can vary; the r_max resolution (for i_th ping) can be calculated as `data[”r_max”][i]/data[”INTENSITY”][i]`


    """
    raise NotImplementedError("Not implemented yet")
    
    

def build_se3_transform(xyzrpy):
    """Creates an SE3 transform from translation and Euler angles.

    Args:
        xyzrpy (list[float]): translation and Euler angles for transform. Must have six components.

    Returns:
        numpy.matrixlib.defmatrix.matrix: SE3 homogeneous transformation matrix

    Raises:
        ValueError: if `len(xyzrpy) != 6`

    """
    if len(xyzrpy) != 6:
        raise ValueError("Must supply 6 values to build transform")

    se3 = matlib.identity(4)
    se3[0:3, 0:3] = euler_to_so3(xyzrpy[3:6])
    se3[0:3, 3] = np.matrix(xyzrpy[0:3]).transpose()
    return se3


def euler_to_so3(rpy):
    """Converts Euler angles to an SO3 rotation matrix.

    Args:
        rpy (list[float]): Euler angles (in radians). Must have three components.

    Returns:
        numpy.matrixlib.defmatrix.matrix: 3x3 SO3 rotation matrix

    Raises:
        ValueError: if `len(rpy) != 3`.

    """
    if len(rpy) != 3:
        raise ValueError("Euler angles must have three components")

    R_x = np.matrix([[1, 0, 0],
                     [0, cos(rpy[0]), -sin(rpy[0])],
                     [0, sin(rpy[0]), cos(rpy[0])]])
    R_y = np.matrix([[cos(rpy[1]), 0, sin(rpy[1])],
                     [0, 1, 0],
                     [-sin(rpy[1]), 0, cos(rpy[1])]])
    R_z = np.matrix([[cos(rpy[2]), -sin(rpy[2]), 0],
                     [sin(rpy[2]), cos(rpy[2]), 0],
                     [0, 0, 1]])
    R_zyx = R_z * R_y * R_x
    return R_zyx


def so3_to_euler(so3):
    """Converts an SO3 rotation matrix to Euler angles

    Args:
        so3: 3x3 rotation matrix

    Returns:
        numpy.matrixlib.defmatrix.matrix: list of Euler angles (size 3)

    Raises:
        ValueError: if so3 is not 3x3
        ValueError: if a valid Euler parametrisation cannot be found

    """
    if so3.shape != (3, 3):
        raise ValueError("SO3 matrix must be 3x3")
    roll = atan2(so3[2, 1], so3[2, 2])
    yaw = atan2(so3[1, 0], so3[0, 0])
    denom = sqrt(so3[0, 0] ** 2 + so3[1, 0] ** 2)
    pitch_poss = [atan2(-so3[2, 0], denom), atan2(-so3[2, 0], -denom)]

    R = euler_to_so3((roll, pitch_poss[0], yaw))

    if (so3 - R).sum() < MATRIX_MATCH_TOLERANCE:
        return np.matrix([roll, pitch_poss[0], yaw])
    else:
        R = euler_to_so3((roll, pitch_poss[1], yaw))
        if (so3 - R).sum() > MATRIX_MATCH_TOLERANCE:
            raise ValueError("Could not find valid pitch angle")
        return np.matrix([roll, pitch_poss[1], yaw])


def so3_to_quaternion(so3):
    """Converts an SO3 rotation matrix to a quaternion

    Args:
        so3: 3x3 rotation matrix

    Returns:
        numpy.ndarray: quaternion [w, x, y, z]

    Raises:
        ValueError: if so3 is not 3x3
    """
    if so3.shape != (3, 3):
        raise ValueError("SO3 matrix must be 3x3")

    R_xx = so3[0, 0]
    R_xy = so3[0, 1]
    R_xz = so3[0, 2]
    R_yx = so3[1, 0]
    R_yy = so3[1, 1]
    R_yz = so3[1, 2]
    R_zx = so3[2, 0]
    R_zy = so3[2, 1]
    R_zz = so3[2, 2]

    try:
        w = sqrt(so3.trace() + 1) / 2
    except(ValueError):
        # w is non-real
        w = 0

    x = sqrt(1 + R_xx - R_yy - R_zz) / 2
    y = sqrt(1 + R_yy - R_xx - R_zz) / 2
    z = sqrt(1 + R_zz - R_yy - R_xx) / 2

    max_index = max(r_max(4), key=[w, x, y, z].__getitem__)

    if max_index == 0:
        x = (R_zy - R_yz) / (4 * w)
        y = (R_xz - R_zx) / (4 * w)
        z = (R_yx - R_xy) / (4 * w)
    elif max_index == 1:
        w = (R_zy - R_yz) / (4 * x)
        y = (R_xy + R_yx) / (4 * x)
        z = (R_zx + R_xz) / (4 * x)
    elif max_index == 2:
        w = (R_xz - R_zx) / (4 * y)
        x = (R_xy + R_yx) / (4 * y)
        z = (R_yz + R_zy) / (4 * y)
    elif max_index == 3:
        w = (R_yx - R_xy) / (4 * z)
        x = (R_zx + R_xz) / (4 * z)
        y = (R_yz + R_zy) / (4 * z)

    return np.array([w, x, y, z])


def se3_to_components(se3):
    """Converts an SE3 rotation matrix to linear translation and Euler angles

    Args:
        se3: 4x4 transformation matrix

    Returns:
        numpy.matrixlib.defmatrix.matrix: list of [x, y, z, roll, pitch, yaw]

    Raises:
        ValueError: if se3 is not 4x4
        ValueError: if a valid Euler parametrisation cannot be found

    """
    if se3.shape != (4, 4):
        raise ValueError("SE3 transform must be a 4x4 matrix")
    xyzrpy = np.empty(6)
    xyzrpy[0:3] = se3[0:3, 3].transpose()
    xyzrpy[3:6] = so3_to_euler(se3[0:3, 0:3])
    return xyzrpy
    
    
    
    
    
    
def test_nerf(checkpoint_path = "experiments/14deg_submarine/checkpoints/ckpt_031279.pth"):
    # crate a nerf network
    nerf_network = NeRF(output_ch=2,d_in_view=3,d_in=3,use_viewdirs=True,multires=3,multires_view=3).to("cuda")
    # load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    nerf_network.load_state_dict(checkpoint['nerf_network'])
    # make sure nerf is on cude
    nerf_network = nerf_network.to("cuda")
    return nerf_network
    pass


def create_plots_from_checkpoint():
    nerf_network = test_nerf()
    x,y,dists = load_data_naive("post_pings_400khz.npz")
    
    # check that both x and y are lists full of torch tensors
    assert isinstance(x,list)
    assert isinstance(y,list)
    assert all([isinstance(xyztp_word_frame,torch.Tensor) for xyztp_word_frame in x])
    assert all([isinstance(intensity,torch.Tensor) for intensity in y])
    
    # Make x numpy array
    x = [xyztp_word_frame.numpy() for xyztp_word_frame in x]
    y = [intensity.numpy() for intensity in y]

    # Find min and max values of x 
    x_min = np.min([np.min(xyztp_word_frame[:,:,0]) for xyztp_word_frame in x])
    x_max = np.max([np.max(xyztp_word_frame[:,:,0]) for xyztp_word_frame in x])
    y_min = np.min([np.min(xyztp_word_frame[:,:,1]) for xyztp_word_frame in x])
    y_max = np.max([np.max(xyztp_word_frame[:,:,1]) for xyztp_word_frame in x])
    z_min = np.min([np.min(xyztp_word_frame[:,:,2]) for xyztp_word_frame in x])
    z_max = np.max([np.max(xyztp_word_frame[:,:,2]) for xyztp_word_frame in x])
    print(f"x_min: {x_min}, x_max: {x_max} y_min: {y_min}, y_max: {y_max} z_min: {z_min}, z_max: {z_max}")
    
    # check the normalization of the intensity
    min_I = np.min([np.min(intensity) for intensity in y])
    max_I = np.max([np.max(intensity) for intensity in y])
    print(f"min_I: {min_I}, max_I: {max_I}")
    
    # check that all directions are normalized
    for dirs in x:
        dirs = dirs[:,:,3:]
        norms = np.linalg.norm(dirs,axis=2)
        if np.min(norms) < 0.99 or np.max(norms) > 1.01:
            print(f"min_norm: {np.min(norms)}, max_norm: {np.max(norms)}")

    # 

    renderer = MBESRenderer(nerf_network)
    # move MBESRenderer to cuda
    for i,(ping,intens,dist) in enumerate(zip(x,y,dists)):
        # make ping a torch tensor
        render_out = renderer.render_core(torch.tensor(ping, dtype=torch.float32).to("cuda"),torch.tensor(dist, dtype=torch.float32).to("cuda"),nerf_network,calculate_ray_orgin_color_light=True)
        #     return_out = {
        #     "density": density,
        #     "sampled_color": sampled_color,
        #     "transmittance": transmittance,
        #     "opacity": opacity,
        #     "absorption": absorption,
        #     "ray_orgin_color_sound": ray_orgin_color_sound.squeeze()
        #     "ray_orgin_color_light": ray_orgin_color_light.squeeze() (optional with arg: calculate_ray_orgin_color_light=True)
        # }

        # make all elements in render_out to numpy
        render_out = {key: value.squeeze().detach().cpu().numpy() for key,value in render_out.items()}
        
        # add true intensities to render_out
        render_out["true_intensities"] = intens


        # make dir if it does not exist
        if not os.path.exists(f'mbes_nerf_plots/ping_{i}/'):
            os.makedirs(f'mbes_nerf_plots/ping_{i}/')
        ploy_xyz_dxdydz_data(ping,render_out,folder_name=f'mbes_nerf_plots/ping_{i}/')
        pass

    # print the shapes of the data
    print(f"Data shape: {x[0].shape}")
    print(f"Target shape: {y[0].shape}")

if __name__ == "__main__":
    create_plots_from_checkpoint()
    