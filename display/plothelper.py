import matplotlib.pyplot as pl
import cv2
import os.path
import numpy as np
import math


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


def plot_result(images, index30, index5, image_dir):

    im_dim = 100
    cols = 10
    space = im_dim / 10
    border = 2 * space

    r_a = math.ceil(len(images) / float(cols))
    h_a = r_a * im_dim + (r_a - 1) * space + 2 * border

    r_t = math.ceil(len(index30) / float(cols))
    h_t = r_t * im_dim + (r_t - 1) * space + 2 * border

    r_f = math.ceil(len(index5) / float(cols / 2))
    h_f = r_f * 2 * im_dim + (r_f - 1) * 2.25 * space + 2 * border

    w = cols * im_dim + (cols - 1) * space + 2 * border
    h = h_a + space + h_t + space + h_f

    background = 255 * np.ones((h, w, 3), np.uint8)
    background[h_a+1:h_a+space+1, border:w-border, :] = np.zeros((space, w-2*border, 3), np.uint8)
    background[h_a + space + h_t+1:h_a + space + h_t+space+1, border:w-border, :] = \
        np.zeros((space, w-2*border, 3), np.uint8)

    for index, image in enumerate(images):
        im = load_image(image, image_dir, im_dim)
        row, col = get_row_col(index, 0, im_dim, space, border, cols)
        insert_image(background, im, col, row)

    for index, image in enumerate(np.array(images)[index30]):
        im = load_image(image, image_dir, im_dim)
        row, col = get_row_col(index, h_a + space, im_dim, space, border, cols)
        insert_image(background, im, col, row)

    for index, image in enumerate(np.array(images)[index5]):
        im = load_image(image, image_dir, 2 * im_dim)
        row, col = get_row_col(index, h_a + space + h_t + space, 2 * im_dim, 2.25 * space, border, cols / 2)
        insert_image(background, im, col, row)

    pl.figure()
    pl.tick_params(axis='both', which='both',
                   bottom='off', top='off', left='off', right='off',
                   labelbottom='off', labelleft='off')

    pl.imshow(background)
    pl.show()


def get_row_col(index, translation, im_dim, space, border, columns):
    index_col = index % columns
    index_row = index / columns
    col = border + np.round(index_col * space + index_col * im_dim + im_dim / 2)
    row = translation + border + np.round(index_row * space + index_row * im_dim + im_dim / 2)

    return row, col


def load_image(image, image_dir, max_size):
    im = cv2.imread(os.path.join(image_dir, image))[:, :, ::-1]  # Reverse to RGB
    size = np.array(im.shape[:2])
    thumb_size = max_size * size / np.max(size)
    im = cv2.resize(im, dsize=(thumb_size[1], thumb_size[0]))

    return im


def insert_image(background, im, col, row):
    size = np.array(im.shape[:2])
    middle = np.mean([[1, 1], 1 + size], axis=0)
    middle_row = middle[0] - 1
    middle_col = middle[1] - 1

    background[row-middle_row:row+middle_row, col-middle_col:col+middle_col, :] = im


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