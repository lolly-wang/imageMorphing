'''
  File name: morph_tri.py
  Author: Luoli Wang
  Date created: 10/10/2019
'''
from scipy.spatial import Delaunay
from numpy.linalg import inv
import numpy as np
from cpselect import cpselect
from PIL import Image
import imageio


def morph_tri(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac):
    point_mid = (im1_pts + im2_pts) / 2

    Tri = Delaunay(point_mid)
    num_vertices = Tri.simplices

    num_row, num_col = im1.shape[0], im1.shape[1]
    tar = np.zeros((len(warp_frac), num_row, num_col, 3))
    for fr in range(len(warp_frac)):  # search every frame
        t = warp_frac[fr]
        tt = dissolve_frac[fr]

        assigned_tri = np.zeros((num_row, num_col))
        inter_img = np.zeros_like(im1)
        for i in range(num_row):
            for j in range(num_col):
                assigned_tri[i, j] = Tri.find_simplex(np.array([i, j]))

        for num_tri in range(num_vertices.shape[0]):

            a, b, c = num_vertices[num_tri]

            a1_x, a1_y = im1_pts[a, 0], im1_pts[a, 1]
            b1_x, b1_y = im1_pts[b, 0], im1_pts[b, 1]
            c1_x, c1_y = im1_pts[c, 0], im1_pts[c, 1]

            a2_x, a2_y = im2_pts[a, 0], im2_pts[a, 1]
            b2_x, b2_y = im2_pts[b, 0], im2_pts[b, 1]
            c2_x, c2_y = im2_pts[c, 0], im2_pts[c, 1]

            at_x = t * a1_x + (1 - t) * a2_x
            at_y = t * a1_y + (1 - t) * a2_y
            bt_x = t * b1_x + (1 - t) * b2_x
            bt_y = t * b1_y + (1 - t) * b2_y
            ct_x = t * c1_x + (1 - t) * c2_x
            ct_y = t * c1_y + (1 - t) * c2_y

            x, y = np.where(assigned_tri == num_tri)
            for p in range(x.shape[0]):
                current_pt = np.array([x[p], y[p], 1])
                vertice_tar = np.array([[at_x, bt_x, ct_x],
                                        [at_y, bt_y, ct_y],
                                        [1, 1, 1]])

                bary_crd = np.dot(inv(vertice_tar), current_pt)
                vertice_img1 = np.array([[a1_x, b1_x, c1_x],
                                         [a1_y, b1_y, c1_y],
                                         [1, 1, 1]])

                crd1 = np.dot(vertice_img1, bary_crd)
                x1 = round(np.float(crd1[0] / crd1[2]))
                y1 = round(np.float(crd1[1] / crd1[2]))

                vertice_img2 = np.array([[a2_x, b2_x, c2_x],
                                         [a2_y, b2_y, c2_y],
                                         [1, 1, 1]])
                crd2 = np.dot(vertice_img2, bary_crd)
                x2 = round(np.float(crd2[0] / crd2[2]))
                y2 = round(np.float(crd2[1] / crd2[2]))

                x1 = np.clip(x1, 0, 299)
                x2 = np.clip(x2, 0, 299)
                y1 = np.clip(y1, 0, 299)
                y2 = np.clip(y2, 0, 299)

                inter_img[x[p], y[p], :] = tt * im1[x1, y1, :] + (1 - tt) * im2[x2, y2, :]
        tar[fr, :, :, :] = inter_img
    return tar


