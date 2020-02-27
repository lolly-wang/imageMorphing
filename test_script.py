'''
  File name: test_script.py
  Author: Haoyuan(Steve) Zhang
  Date created: 9/26/2017
'''

'''
  File clarification:
    Check the accuracy of your algorithm
'''

import numpy as np

# from est_tps import est_tps
# from obtain_morphed_tps import obtain_morphed_tps
# from morph_tps import morph_tps
from morph_tri import morph_tri
from PIL import Image
import imageio
from cpselect import cpselect


# test triangulation morphing
def test_tri(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac):
    # generate morphed image
    morphed_ims = morph_tri(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac)

    # check output output image number
    if morphed_ims.shape[0] != 2:
        print("The number of output image is wrong. \n")
        return False

    morphed_im1 = morphed_ims[0, :, :, :]
    # check the color channel number
    if morphed_im1.shape[2] != 3:
        print("What happened to color channel? \n")
        return False

    # check the image size
    if morphed_im1.shape[0] != 50 or morphed_im1.shape[1] != 50:
        print("Something wrong about the size of output image. \n")
        return False

    print("Triangulation Morphing Test Passed!")
    return True


# the main test code
def main():
    im1 = np.ones((50, 50, 3))
    im2 = np.zeros((50, 50, 3))

    im1_pts = np.array([[1, 1], [1, 50], [50, 1], [50, 50], [25, 25]])
    im2_pts = np.array([[1, 1], [1, 50], [50, 1], [50, 50], [20, 20]])

    warp_frac, dissolve_frac = np.array([0.2, 0.3]), np.array([0.1, 0.3])

    if not test_tri(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac):
        print("The Triangulation Morphing test failed. \n")
        return

    print("All tests passed! \n")




if __name__ == "__main__":
   # test triangulation morphing
    main()

    im1 = np.array(Image.open('3.jpeg').convert('RGB'))
    im2 = np.array(Image.open('4.jpeg').convert('RGB'))

    resize_img1 = np.array(Image.fromarray(im1).resize([300, 300]))
    resize_img2 = np.array(Image.fromarray(im2).resize([300, 300]))

    im1_pts, im2_pts = cpselect(im1, im2)

    warp_frac = 1 / 60 * np.array(range(61))
    dissolve_frac = 1 / 60 * np.array(range(61))
 
    E = morph_tri(resize_img1, resize_img2, im1_pts, im2_pts, warp_frac, dissolve_frac)

    imageio.mimwrite('face_morph_test.avi', E, fps=15)


