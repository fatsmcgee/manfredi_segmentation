# -*- coding: utf-8 -*-
"""
Created on Tue May 05 21:39:03 2015

@author: gumpy
"""

import os
import shutil
files = os.listdir('.')

otherfiles = os.listdir(r'C:\Users\gumpy\Desktop\Class Notes\Advanced Machine Learning\Project\pennfudan_segments2')

for f,otherf in zip(files,otherfiles):
    basepath = r'C:\Users\gumpy\Desktop\Class Notes\Advanced Machine Learning\Project\pennfudan_segments2'
    shutil.move(os.path.join(basepath,otherf),os.path.join(basepath,f))