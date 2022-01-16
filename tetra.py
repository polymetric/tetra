import math
import OpenImageIO as oiio
import OpenEXR
import os
import sys
import re
import numpy as np

def tetra_vol(a, b, c, d):
    return np.linalg.norm(np.dot(a-d, np.cross(b-d, c-d)))/6

# converts a cartesian coordinate to tetrahedral barycentric coordinates
# tetra should be a 4x3 numpy array
# point should be a 3 long numpy array
def to_tetra_bary(tetra, point)
    result = np.zeros(4)
    vol_abcp = tetra_vol(tetra[0], tetra[1], tetra[2], point)
    vol_abdp = tetra_vol(tetra[0], tetra[1], tetra[3], point)
    vol_acdp = tetra_vol(tetra[0], tetra[2], tetra[3], point)
    vol_bcdp = tetra_vol(tetra[1], tetra[2], tetra[3], point)
    vol_total = tetra_vol(tetra[0], tetra[1], tetra[2], tetra[3])
    result[0] = vol_abcp/vol_total
    result[1] = vol_abdp/vol_total
    result[2] = vol_acdp/vol_total
    result[3] = vol_bccp/vol_total
    return result

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

# find nearest tetrahedron to target 
# cloud should be an Nx3 numpy array
# target should be a 3 long numpy array
def find_tetra(cloud, target):
    result = np.zeros(4, 3)
    index = [(point, dist = ((point - target)**2).sum()) for point in cloud]
    index.sort(key=lambda i: i[1])
    for i in range(len(index)):
        result[i] = index[i][0]
    return result

