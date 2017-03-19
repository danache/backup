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

from opendr.camera import ProjectPoints
from lib.robustifiers import GMOf
from lib.sphere_collisions import SphereCollisions
from lib.max_mixture_prior import MaxMixtureCompletePrior
from render_model import render_model
import os
path = './1'
for root, dirs, files in os.walk(path):
    for file in files:
        if  file.endswith('png'):
            print path+'/'+file[:-9]+".npz"