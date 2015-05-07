import matplotlib.pyplot as pl
import cv2
import os.path
import numpy as np


def plot_images(image_dir, image_files, rows, columns):
    """ Plots image files in a figure with the defined
        number of rows and columns.
        
        Parameters
        ----------
        image_files : A list of image file names.
        rows : The number of rows in the figure.
        columns : The number of columns in the figure.
    """
    
    i = 1
    fig = pl.figure()
    
    for image_file in image_files:
        image = cv2.imread(os.path.join(image_dir, image_file))[:, :, ::-1]  # Reverse to RGB

        sub = fig.add_subplot(rows, columns, i)
        sub.imshow(image)
        i += 1
    
    pl.show()


def plot_pca_projections(V, pc1, pc2):
    """ Plots the projections for the specified principal components.
        
        Parameters
        ----------
        V : The principal component projections in rows.
        pc1 : The first principal component to plot against.
        pc2 : The second principal component to plot against.
    """
    
    pl.figure()
    pl.plot(V[:, pc1], V[:, pc2], '*')
    pl.axhline(0)
    pl.axvline(0)
    pl.show()


def plot_pca_images(image_dir, images, V, pc1, pc2):
    """ Plots the images onto the projections for the specified principal components.

        Parameters
        ----------
        image_dir: The image directory.
        images: The image names.
        V : The principal component projections in rows.
        pc1 : The first principal component to plot against.
        pc2 : The second principal component to plot against.
    """

    sz = 3601
    xm = (sz + 1) / 2
    ym = xm
    unit = (sz + 1) / 2
    im_project = 255 * np.ones((sz, sz, 3), np.uint8)
    im_project[xm-2:xm+3, :, :] = np.zeros((5, sz, 3))
    im_project[:, ym-2:ym+3, :] = np.zeros((sz, 5, 3))

    middles = []
    for index, image in enumerate(images):
        im = cv2.imread(os.path.join(image_dir, image))[:, :, ::-1]  # Reverse to RGB
        size = np.array(im.shape[:2])
        thumb_size = 100 * size / np.max(size)
        im = cv2.resize(im, dsize=(thumb_size[1], thumb_size[0]))
        middle = np.mean([[1, 1], 1 + thumb_size], axis=0)
        middle_x = middle[0] - 1
        middle_y = middle[1] - 1

        x = np.round(unit * V[index, pc1]) + xm
        y = np.round(unit * V[index, pc2]) + ym

        im_project[x-middle_x:x+middle_x, y-middle_y:y+middle_y, :] = im

        middles.append(np.array(x, y))

    middles = np.array(middles).astype(np.int)
    coord_max = np.max(middles, axis=0)
    coord_min = np.min(middles, axis=0)
    coord_diff = np.subtract(coord_max, coord_min)
    k = (np.max(coord_diff) - coord_diff) / 2
    min_lim = np.max([coord_min - k - 100, 0])
    max_lim = np.min([coord_max + k + 100, sz])

    im_project = im_project[min_lim:max_lim, min_lim:max_lim, :]

    pl.figure()
    pl.tick_params(axis='both', which='both',
                   bottom='off', top='off', left='off', right='off',
                   labelbottom='off', labelleft='off')

    pl.xlabel('Principal component {0}'.format(pc1))
    pl.ylabel('Principal component {0}'.format(pc2))
    pl.imshow(im_project)
    pl.show()


def plot_points(x,y):
    """ Plots the values in y against the values in x.
        
        Parameters
        ----------
        x : An array of points.
        y : An array of points.
    """
    
    pl.figure()
    pl.plot(x, y, '*')
    pl.show()