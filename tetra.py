#!/bin/env python3

import math
#import OpenImageIO as oiio
import OpenEXR
import os
import sys
import re
import numpy as np
from tqdm import tqdm
import random
import cProfile

def tetra_vol_2(a, b, c, d):
    # dont need to divide by 6 since it's all relative anyway
    return np.abs(np.dot(a-d, np.cross(b-d, c-d)))

def tetra_vol_t(tetra):
    return tetra_vol_2(tetra[0], tetra[1], tetra[2], tetra[3])

## converts a cartesian coordinate to tetrahedral barycentric coordinates
## tetra should be a 4x3 numpy array
## point should be a 3 long numpy array
#def to_tetra_bary(tetra, point):
#    result = np.zeros(4)
#    vol_bcdp = tetra_vol(tetra[1], tetra[2], tetra[3], point)
#    vol_cdap = tetra_vol(tetra[2], tetra[3], tetra[0], point)
#    vol_dabp = tetra_vol(tetra[3], tetra[0], tetra[1], point)
#    vol_abcp = tetra_vol(tetra[0], tetra[1], tetra[2], point)
#    vol_total = tetra_vol(tetra[0], tetra[1], tetra[2], tetra[3])
#    result[0] = vol_bcdp/vol_total
#    result[1] = vol_cdap/vol_total
#    result[2] = vol_dabp/vol_total
#    result[3] = vol_abcp/vol_total
#    return result

def tetra_vol(a,b,c):
    return np.abs(np.dot(a,np.cross(b,c)))

def to_tetra_bary(tetra, p):
    a = tetra[0]
    b = tetra[1]
    c = tetra[2]
    d = tetra[3]
    va = tetra_vol(p-b, p-d, p-c)
    vb = tetra_vol(p-a, p-c, p-d)
    vc = tetra_vol(p-a, d-a, b-a)
    vd = tetra_vol(p-a, b-a, a-c)
    v = tetra_vol(b-a, c-a, d-a)
    return np.array([va/v, vb/v, vc/v, vd/v])

# tetrahedronal barycentric coordinates to cartesian coordinates
# probably could use matrices for this but i'm too stupid
# bary should be a 4 long numpy array
# tetra should be a 4x3 numpy array
def tetra_bary_to_cart(tetra, bary):
    result = np.zeros(3)
    result[0] = bary[0]*tetra[0][0] + bary[1]*tetra[1][0] + bary[2]*tetra[2][0] + bary[3]*tetra[3][0] 
    result[1] = bary[0]*tetra[0][1] + bary[1]*tetra[1][1] + bary[2]*tetra[2][1] + bary[3]*tetra[3][1] 
    result[2] = bary[0]*tetra[0][2] + bary[1]*tetra[1][2] + bary[2]*tetra[2][2] + bary[3]*tetra[3][2] 
    return result

def tetra_interp(point, src, tgt):
    return tetra_bary_to_cart(tgt, to_tetra_bary(src, point))

# returns true if any of the values are negative
def has_negatives(tetra):
    for i in tetra:
        if i < 0: return True
        if i == float('inf'): return True
        if np.isnan(i): return True
    return False

def unique(a):
    return len(set(a)) == len(a)

# find nearest tetrahedron to target 
# cloud should be an Nx3 numpy array
# obviously N should be greater than 4
# target should be a 3 long numpy array
# returns a list of tuples where 0 is the triangle and 1 is the index
def find_tetra(cloud, target):
    result = (np.zeros((4,3)), [])
    # point, index, exclude
    index = [(cloud[i], i, False) for i in range(len(cloud))]
    point_in_tetra = False
    while not point_in_tetra:
        result[1].clear()
        for i in range(4):
            j = random.randrange(len(index))
            result[0][i] = index[j][0] # tetras
            result[1].append(index[j][1]) # indices
#       print(to_tetra_bary(result[0], target))
#       print(result[1])
        if not has_negatives(to_tetra_bary(result[0], target)) and unique(result[1]) and tetra_vol_t(result[0]) > 0:
            print(to_tetra_bary(result[0], target))
            point_in_tetra = True
#       else:
#           print('point outside tetra, trying again')
#   print(f'found good tetra\n{result[0]}')
    return result

def load_table(file):
    file = open(file, 'r').read()
    list = []
    for i in file.split('\n'):
        row = []
        if i == '': continue
        for j in i.split(' '):
            row.append(float(j))
        list.append(row)
    return np.array(list)

cloud_src = load_table('test1')
cloud_tgt = load_table('test2')
lut_res = 3
result = np.zeros((lut_res,lut_res,lut_res))
outfile = open('test.cube', 'w')
outfile.write(f'LUT_3D_SIZE {lut_res}\n\n')



#src_tetra = np.array([[0,0,0],[1,1,0],[1,0,0],[1,0,1]])
#tgt_tetra = np.array([[0,0,0],[1,1,0],[1.2,0,0],[1,0,1]])
#n=5
#for x, y, z in np.ndindex((n,n,n)):
#    point = np.array([x/(n-1),y/(n-1),z/(n-1)])
#    #point = np.array([random.random(), random.random(), random.random()])
#    #point = np.array([0.57110176, 0.2171957, 0.1344861])
#    bary = to_tetra_bary(src_tetra, point)
#    after_point = tetra_interp(point, src_tetra, tgt_tetra)
#    print(f'{point[0]:.8f} {point[1]:.8f} {point[2]:.8f}  ->  {bary[0]:.8f} {bary[1]:.8f} {bary[2]:.8f} {bary[3]:.8f}  ->  {after_point[0]:.8f} {after_point[1]:.8f} {after_point[2]:.8f}')
#




for z, y, x in tqdm(np.ndindex(result.shape), total=lut_res**3):
    xf = x/(lut_res-1)
    yf = y/(lut_res-1)
    zf = z/(lut_res-1)
    point = np.array([xf,yf,zf])
#   point = np.array([random.random(), random.random(), random.random()])
    src_tetra, indices = find_tetra(cloud_src, point)
    tgt_tetra = np.array([cloud_tgt[indices[0]], cloud_tgt[indices[1]], cloud_tgt[indices[2]], cloud_tgt[indices[3]]])
    point = tetra_interp(point, src_tetra, tgt_tetra)
    outfile.write(f'{point[0]} {point[1]} {point[2]}\n')
    




