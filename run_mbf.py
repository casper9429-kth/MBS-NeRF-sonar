import os, sys
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from helpers import *
from MLP import *
#from PIL import Image
import cv2 as cv
import time
import random
import string 
from models.fields import NeRF
from models.mbes_renderer import MBESRenderer
from itertools import groupby
from operator import itemgetter
from load_data import load_data_naive
import logging
import argparse 
import json
import yaml

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()



class Runner:
    def __init__(self, conf, is_continue=False, write_config=True):
        conf_path = conf
        f = open(conf_path)
        conf_text = f.read()
        self.is_continue = is_continue
        self.conf = ConfigFactory.parse_string(conf_text)
        self.write_config = write_config

    def set_params(self):
        self.expID = self.conf.get_string('conf.expID') 
        self.image_setkeyname =  self.conf.get_string('conf.image_setkeyname') 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.variation_reg_weight = self.conf.get_float('train.variation_reg_weight')
        self.px_sample_min_weight = self.conf.get_float('train.px_sample_min_weight')
        self.base_exp_dir = './experiments/{}'.format(self.expID)


        self.data,self.target,self.dists = load_data_naive()
        
        
        self.timef = self.conf.get_bool('conf.timef')
        self.end_iter = self.conf.get_int('train.end_iter')
        self.start_iter = self.conf.get_int('train.start_iter')
         


        extrapath = './experiments/{}'.format(self.expID)
        if not os.path.exists(extrapath):
            os.makedirs(extrapath)

        extrapath = './experiments/{}/checkpoints'.format(self.expID)
        if not os.path.exists(extrapath):
            os.makedirs(extrapath)

        extrapath = './experiments/{}/model'.format(self.expID)
        if not os.path.exists(extrapath):
            os.makedirs(extrapath)

        if self.write_config:
            with open('./experiments/{}/config.json'.format(self.expID), 'w') as f:
                json.dump(self.conf.__dict__, f, indent = 2)


        self.criterion = torch.nn.L1Loss(reduction='mean')
        
        self.model_list = []

        # Networks
        params_to_train = []
        # nerf
        self.nerf_network = NeRF(output_ch=2,d_in_view=3,d_in=3,use_viewdirs=True,multires=3,multires_view=3).to(self.device)
        params_to_train += list(self.nerf_network.parameters())
        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)


        self.iter_step = 0
        self.renderer = MBESRenderer(self.nerf_network)

        latest_model_name = None
        if self.is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth': #and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)
    
    
    def train(self):
        for epoch in tqdm(range(self.start_iter, self.end_iter)):            

            # Shuffle the training data list, self.target and self.data
            ## random permutation of 0 to len(self.data) axis 0
            permutation_pings = np.random.permutation(len(self.data))
            ## shuffle the data and target according to the permutation, it is a python list with numy arrays inside
            self.data = [self.data[i] for i in permutation_pings]
            self.target = [self.target[i] for i in permutation_pings]
            
            


            sum_intensity_loss = 0
            
            for i in tqdm(range(len(self.data))):           
                x = self.data[i]
                y = self.target[i]
                dist = self.dists[i]


                render_out = self.renderer.render_core(x, dist,self.nerf_network)
                #     return_out = {
                #     "density": density,
                #     "sampled_color": sampled_color,
                #     "transmittance": transmittance,
                #     "opacity": opacity,
                #     "absorption": absorption,
                #     "ray_orgin_color_sound": ray_orgin_color_sound.squeeze()
                #     "ray_orgin_color_light": ray_orgin_color_light.squeeze() (optional with arg: calculate_ray_orgin_color_light=True)
                # }
                y_pred = render_out['ray_orgin_color_sound']

                # normalize the y_pred by elements in axis 0 and 1
                intensity_error = self.criterion(y_pred, y)#*(1/(y.shape[0]*y.shape[1]))
                    
            
                loss = intensity_error # TODO: variation_regularization*self.variation_reg_weight

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


                with torch.no_grad():
                    sum_intensity_loss += intensity_error.cpu().numpy().item()
                    
                    
                self.iter_step += 1
                self.update_learning_rate()

                del(y_pred)
                del(loss)
                del(intensity_error)
                del(x)
                del(y)
                torch.cuda.empty_cache()
                
                

            if epoch ==0 or epoch % 10 == 0:
                logging.info('iter:{} ********************* SAVING CHECKPOINT ****************'.format(self.optimizer.param_groups[0]['lr']))
                self.save_checkpoint()

            print('iter: {} loss: {}'.format(i, sum_intensity_loss))
            


    def save_checkpoint(self):
        checkpoint = {
            'nerf_network': self.nerf_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.nerf_network.load_state_dict(checkpoint['nerf_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default="./confs/conf.conf")
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.is_continue)
    runner.set_params()
    runner.train()