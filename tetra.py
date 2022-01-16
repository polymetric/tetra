#!/bin/env python3

import math
#import OpenImageIO as oiio
import OpenEXR
import os
import sys
import re
import numpy as np
from tqdm import tqdm

def tetra_vol(a, b, c, d):
    # dont need to divide by 6 since it's all relative anyway
    return np.abs(np.dot(a-d, np.cross(b-d, c-d)))

# converts a cartesian coordinate to tetrahedral barycentric coordinates
# tetra should be a 4x3 numpy array
# point should be a 3 long numpy array
def to_tetra_bary_old(tetra, point):
    result = np.zeros(4)
    vol_dabp = tetra_vol(tetra[3], tetra[0], tetra[1], point)
    vol_abcp = tetra_vol(tetra[0], tetra[1], tetra[2], point)
    vol_bcdp = tetra_vol(tetra[1], tetra[2], tetra[3], point)
    vol_cdap = tetra_vol(tetra[2], tetra[3], tetra[0], point)
    vol_total = tetra_vol(tetra[0], tetra[1], tetra[2], tetra[3])
    result[0] = vol_dabp/vol_total
    result[1] = vol_abcp/vol_total
    result[2] = vol_bcdp/vol_total
    result[3] = vol_cdap/vol_total
    return result

def sctp(a, b, c):
    return np.dot(a, np.cross(b,c))

def to_tetra_bary(tetra, point):
    p = point
    a = tetra[0]
    b = tetra[1]
    c = tetra[2]
    d = tetra[3]

    vap = p-a
    vbp = p-b
    vab = b-a
    vac = c-a
    vad = a-d
    vbc = b-c
    vbd = b-d

    va6 = np.abs(sctp(vbp,vbd,vbc))
    vb6 = np.abs(sctp(vap,vac,vad))
    vc6 = np.abs(sctp(vap,vad,vab))
    vd6 = np.abs(sctp(vap,vab,vac))
    v6 = 1/sctp(vab,vac,vad)
    return np.array([va6*v6, vb6*v6, vc6*v6, vd6*v6])

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

# find nearest tetrahedron to target 
# cloud should be an Nx3 numpy array
# obviously N should be greater than 4
# target should be a 3 long numpy array
# returns a list of tuples where 0 is the triangle and 1 is the index
def find_tetra(cloud, target):
    result = (np.zeros((4,3)), [])
    # point, distance, index
    index = [(cloud[i], ((cloud[i] - target)**2).sum(), i) for i in range(len(cloud))]
    index.sort(key=lambda i: i[1])
    for i in range(4):
        result[0][i] = index[i][0] # tetras
        result[1].append(index[i][2]) # indices
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
lut_res = 33
result = np.zeros((lut_res,lut_res,lut_res))
outfile = open('test.cube', 'w')
outfile.write(f'LUT_3D_SIZE {lut_res}\n\n')



src_tetra = np.array([[1,0,0],[0,0,0],[1,1,0],[1,0,1]])
tgt_tetra = np.array([[1,0,0],[0,0,0],[1,1,0],[1,0,1]])
print(to_tetra_bary(src_tetra, np.array([0, 0, 0])))
print(to_tetra_bary_old(src_tetra, np.array([0, 0, 0])))
n=5
for x, y, z in np.ndindex((n,n,n)):
    point = np.array([x/(n-1),y/(n-1),z/(n-1)])
    bary = to_tetra_bary(src_tetra, point)
    after_point = tetra_interp(point, src_tetra, tgt_tetra)
    print(f'{point[0]:.2f} {point[1]:.2f} {point[2]:.2f}  ->  {bary[0]:.2f} {bary[1]:.2f} {bary[2]:.2f} {bary[3]:.2f}  ->  {after_point[0]:.2f} {after_point[1]:.2f} {after_point[2]:.2f}')
print(tetra_vol(src_tetra[0], src_tetra[1], src_tetra[2], src_tetra[3]))




exit()

for z, y, x in tqdm(np.ndindex(result.shape), total=lut_res**3):
    xf = x/(lut_res-1)
    yf = y/(lut_res-1)
    zf = z/(lut_res-1)
    point = np.array([xf,yf,zf])
    src_tetra, indices = find_tetra(cloud_src, point)
    tgt_tetra = np.array([cloud_tgt[indices[0]], cloud_tgt[indices[1]], cloud_tgt[indices[2]], cloud_tgt[indices[3]]])
    point = tetra_interp(point, src_tetra, tgt_tetra)
    outfile.write(f'{point[0]} {point[1]} {point[2]}\n')
    




