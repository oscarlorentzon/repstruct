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


def plot_pca_images(image_dir, images, V, pc1, pc2, im_dim=100, dim=4000, min_axis=0.):
    """ Plots the images onto the projections for the specified principal components.
        Crops the projection image to the outermost images automatically. This can be
        overridden by setting the min_axis.

        Parameters
        ----------
        image_dir: The image directory.
        images: The image names.
        V : The principal component projections in rows.
        pc1 : The first principal component to plot against.
        pc2 : The second principal component to plot against.
        im_dim : Dimension of longest side of collection images.
        dim : Dimension of projection background.
        min_axis : Minimum axis span in interval [0, 1].
    """

    unit = dim / 2

    center = np.round(unit * np.max([np.max(np.abs(V[:, [pc1, pc2]])), min_axis])) + im_dim
    background_dim = 2 * center
    background = 255 * np.ones((background_dim, background_dim, 3), np.uint8)
    background[center-2:center+3, :, :] = np.zeros((5, background_dim, 3))
    background[:, center-2:center+3, :] = np.zeros((background_dim, 5, 3))

    for index, image in enumerate(images):
        im = load_image(image, image_dir, im_dim)
        row = np.round(unit * V[index, pc1]) + center
        col = np.round(unit * V[index, pc2]) + center
        insert_image(background, im, col, row)

    fig = pl.figure()
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0, hspace=0)
    pl.tick_params(axis='both', which='both',
                   bottom='off', top='off', left='off', right='off',
                   labelbottom='off', labelleft='off')

    pl.xlabel('Principal component {0}'.format(pc1), fontsize=12)
    pl.ylabel('Principal component {0}'.format(pc2), fontsize=12)
    pl.imshow(background)
    pl.show()


def plot_result(images, index_thirty, index_five, image_dir, im_dim=200, cols=10):
    """ Shows the result by plotting all images on top, then the thirty closest images
        and at last the five closest in double size.

        Parameters
        ----------
        images: Image names.
        index_thirty: Indexes for the thirty closest images.
        index_five: Indexes for the five closest images.
        image_dir: Image directory.
        im_dim: Dimension of the longest side of the image.
        cols: Number of image columns. The five closest images will have half the columns.
    """

    space = im_dim / 10
    border = 2 * space
    k_space = (cols - 1.) / (cols / 2 - 1)

    r_a = math.ceil(len(images) / float(cols))
    h_a = r_a * im_dim + (r_a - 1) * space + 2 * border

    r_t = math.ceil(len(index_thirty) / float(cols))
    h_t = r_t * im_dim + (r_t - 1) * space + 2 * border

    r_f = math.ceil(len(index_five) / float(cols / 2))
    h_f = r_f * 2 * im_dim + (r_f - 1) * k_space * space + 2 * border

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

    for index, image in enumerate(np.array(images)[index_thirty]):
        im = load_image(image, image_dir, im_dim)
        row, col = get_row_col(index, h_a + space, im_dim, space, border, cols)
        insert_image(background, im, col, row)

    for index, image in enumerate(np.array(images)[index_five]):
        im = load_image(image, image_dir, 2 * im_dim)
        row, col = get_row_col(index, h_a + space + h_t + space, 2 * im_dim, k_space * space, border, cols / 2)
        insert_image(background, im, col, row)

    fig = pl.figure()
    fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0, hspace=0)
    pl.axis('off')
    pl.imshow(background)
    pl.show()


def get_row_col(index, translation, im_dim, space, border, columns):
    """ Retrieves the middle row and middle column based on an index.

        Parameters
        ----------
        index: The index.
        translation: The row translation.
        im_dim: The dimension of the image.
        space: The space between images.
        border: The image border size.
        columns: The number of columns.

        Returns
        -------
        row: The row.
        col: The column.
    """

    index_col = index % columns
    index_row = index / columns
    col = border + np.round(index_col * space + index_col * im_dim + im_dim / 2)
    row = translation + border + np.round(index_row * space + index_row * im_dim + im_dim / 2)

    return row, col


def load_image(image, image_dir, max_size):
    """ Loads an image and resizes it to the specified max size.

        Parameters
        ----------
        image: Image name.
        image_dir: Image directory..
        max_size: The max dimension of the image.

        Returns
        -------
        row: The image array.
    """

    im = cv2.imread(os.path.join(image_dir, image))[:, :, ::-1]  # Reverse to RGB
    size = np.array(im.shape[:2])
    thumb_size = max_size * size / np.max(size)
    im = cv2.resize(im, dsize=(thumb_size[1], thumb_size[0]))

    return im


def insert_image(background, im, col, row):
    """ Inserts an image into a background image.

        Parameters
        ----------
        background: Background image array.
        im: Image array.
        col: The middle row to insert the image to.
        row: The middle column to insert the image to.
    """

    rows = im.shape[0] / 2.
    cols = im.shape[1] / 2.

    background[row-rows:row+rows, col-cols:col+cols, :] = im


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