import torch
import torch.nn as nn
import torch.nn.functional as F

class MBESRenderer:
    def __init__(self, nerf):
        self.nerf = nerf

    def render_core(self, ray_sample_xyzdxdydz,dist, nerf, calculate_ray_orgin_color_light=False):
        """
        Render background
        """
        batch_size = ray_sample_xyzdxdydz.shape[0]  # rays (water columns/beams)
        # move batch_zize to cuda
        n_samples = ray_sample_xyzdxdydz.shape[1]  # samples (range bins)
        
        # Calculate the direction of the rays using PyTorch operations
        pts = ray_sample_xyzdxdydz[:, :, :3]
        dirs = ray_sample_xyzdxdydz[:, :, 3:]
        dist = dist
        
        # Flatten the points and directions
        pts_flat = pts.view(-1, 3)
        dirs_flat = dirs.view(-1, 3)
        
        
        
        density, sampled_color = nerf(pts_flat, dirs_flat)

        # Nerf outputs are in the shape of (batch_size * n_samples, 1)
        
        # Density (σ): The volume density of a point, which indicates how much light is absorbed by that point. Higher density means more absorption (more opacity).
        # Delta Distance (Δs): The distance between consecutive samples along the ray.
        # Opacity (α): A measure of opaqueness; 00 indicates full transparency, and 11 indicates full opacity.
        # Transmittance (T): The fraction of light that is not absorbed when passing through a medium; 00 indicates full absorption (no light passes through), and 11 indicates full transmission (no absorption).


        # Math TODO: This model is a little bit simplified, we might to model the wave travelling back and forth further down the development
        # absoprtion: a = e^(-σΔs)
        # transmittance: T_1 = 1, T_i = T_(i-1) * (e^(-σΔs))
        # opacity: α_i = 1 - e^(-σ_iΔs)
        # ray_orgin_color_light = sum(T_i*α_i*C_i)  # This is true for light, light moves instantly, but the sonar sends a pulse and waits dt seconds for the return,
        # the strength is not the cumulative sum of the stengths, but the strength of the last pulse. The weaking of the pulse is modelled by the cumprod in the transmittance.
        # ray_orgin_color_sound = T_i*α_i*C_i  
        
        # Nerf engineering decisions
        # apply sigmoid to the sampled color to make sure it's in the range of [0, 1] and make it more stable
        # apply softplus to the density to make sure it's positive and make it more stable 
        
        # TODO: determine what we are interested in, we are not interested in the final color at the pose, rather the individual colors at each sample point
        # but these are reflected sonar rays, so they depend on the the materials properties of the objects below, if we used the ray sample color, the network
        # will not learn to predict the density. 
        # The color depends on the density of the objects before it, we therefore need to use a comelative final color using the previous colors, for this we use the 
        # density and color. 
        
        # Why are we using colors and densities? Shouldn't we just use the density? 
        # The density is the pass through of the ray, the color is modelling the absorption of the ray, together they give the final color of the ray.
        # Without color, we would assume that all ray that is not pass through is reflected, which is not true.


        # reconstruct the rays 
        density = density.view(batch_size, n_samples, 1)
        sampled_color = sampled_color.view(batch_size, n_samples, 1)
        
        # apply engineering decisions to the sampled color and density
        density = F.softplus(density)
        sampled_color = torch.sigmoid(sampled_color)

        # calculate absorption
        absorption = torch.exp(-density * dist)
        # For each ray, calculate the transmittance
        transmittance = torch.cumprod(torch.cat([torch.ones([batch_size, 1, 1]).to(absorption.device), absorption + 1e-7], dim=1), dim=1)[:, :-1]
        # calculate the opacity
        opacity = 1.0 - absorption
        # calculate the ray_origin_color
        # ray_origin_color = torch.cumsum(transmittance * opacity * sampled_color, dim=1) # For light
        ray_orgin_color_sound = transmittance * opacity * sampled_color        

        return_dict = {
            "density": density,
            "sampled_color": sampled_color,
            "transmittance": transmittance,
            "opacity": opacity,
            "absorption": absorption,
            "ray_orgin_color_sound": ray_orgin_color_sound.squeeze()
        }
        
        if calculate_ray_orgin_color_light:
            ray_orgin_color_light = torch.cumsum(transmittance * opacity * sampled_color, dim=1)
            return_dict["ray_orgin_color_light"] = ray_orgin_color_light.squeeze()
        
        return return_dict



