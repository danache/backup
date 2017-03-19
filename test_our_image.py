from fit_3d import  run_single_fit
from os.path import join, exists, abspath, dirname
from os import makedirs
import logging
import cPickle as pickle
from time import time
from glob import glob
import argparse

import cv2
import numpy as np
import chumpy as ch
from smpl_webuser.serialization import load_model
from opendr.camera import ProjectPoints
from lib.robustifiers import GMOf
from lib.sphere_collisions import SphereCollisions
from lib.max_mixture_prior import MaxMixtureCompletePrior
from render_model import render_model
_LOGGER = logging.getLogger(__name__)
MODEL_DIR = join(abspath(dirname(__file__)), 'models')
    # Model paths:
MODEL_NEUTRAL_PATH = join(
        MODEL_DIR, 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
MODEL_FEMALE_PATH = join(
        MODEL_DIR, 'basicModel_f_lbs_10_207_0_v1.0.0.pkl')
MODEL_MALE_PATH = join(MODEL_DIR,
                           'basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
def test_our(img_path,
             joint_path,
         out_path="test.pkl",
         use_interpenetration=True,
         n_betas=10,
         flength=5000.,
         pix_thsh=25.,
         use_neutral=False,
         viz=True):
    with open('lsp_gender.csv') as f:
        genders = f.readlines()
    model_female = load_model(MODEL_FEMALE_PATH)
    model_male = load_model(MODEL_MALE_PATH)

    out_path = out_path + ".pkl"
    print img_path
    img = cv2.imread(img_path)

    gender = 'male'

    data = np.load(joint_path)['pose']
    joint = []
    for i in range(len(data[0])):
        joint.append([data[0][i],data[1][i]])
    conf = data[2]
    do_degrees = [0.]
    joint = np.array(joint)
    sph_regs = None
    params, vis = run_single_fit(
        img,
        joint,
        conf,
        model_male,
        regs=sph_regs,
        n_betas=n_betas,
        flength=flength,
        pix_thsh=pix_thsh,
        scale_factor=1,
        viz=viz,
        do_degrees=do_degrees)
    if viz:
        import matplotlib.pyplot as plt
        plt.ion()
        plt.show()
        plt.subplot(121)
        plt.imshow(img[:, :, ::-1])
        if do_degrees is not None:
            for di, deg in enumerate(do_degrees):
                plt.subplot(122)
                plt.cla()
                plt.imshow(vis[di])
                plt.draw()
                plt.title('%d deg' % deg)
                plt.pause(1)
    with open(out_path, 'w') as outf:
        pickle.dump(params, outf)

    # This only saves the first rendering.
    if do_degrees is not None:
        cv2.imwrite(out_path.replace('.pkl', '.png'), vis[0])




import os
path = './1'
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith('png'):
            img_path =  path + '/' + file
            joint_path = path + '/' + file[:-9] + ".npz"
            output_path = path + '/' + file[:-9] + "_out"
            test_our(img_path=img_path, joint_path=joint_path,out_path=output_path)
