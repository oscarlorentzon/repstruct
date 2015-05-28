import matplotlib.pyplot as pl
import cv2
import os.path
import numpy as np
import math


def plot_images(image_dir, image_files, rows, columns):
    """ Plots image files in a figure with the defined number of
        rows and columns.

    :param image_files: A list of image file names.
    :param rows: The number of rows in the figure.
    :param columns: The number of columns in the figure.
    """

    fig = pl.figure()
    
    for index, image_file in enumerate(image_files):
        # Load image and reverse to RGB
        image = cv2.imread(os.path.join(image_dir, image_file))[:, :, ::-1]

        sub = fig.add_subplot(rows, columns, index + 1)
        sub.imshow(image)

    pl.show()


def plot_pca_projections(pc_projections, pc1, pc2):
    """ Plots the projections for the specified principal components.

    :param pc_projections: The principal component projections in rows.
    :param pc1: The first principal component to plot against.
    :param pc2: The second principal component to plot against.
    """
    
    pl.figure()
    pl.plot(pc_projections[:, pc1], pc_projections[:, pc2], '*')
    pl.axhline(0)
    pl.axvline(0)
    pl.show()


def plot_pca_images(image_dir, images, pc_projections, pc1, pc2, im_dim=100, dim=3300, min_axis=0.,
                    ticks=False, save_path=None):
    """ Plots the images onto the projections for the specified
        principal components. Crops the projection image to the
        outermost images automatically. This can be overridden by
        setting the min_axis.

    :param image_dir: The image directory.
    :param images: The image names.
    :param pc_projections: The principal component projections in rows.
    :param pc1: The index of the first principal component to plot against.
    :param pc2: The index of the second principal component to plot against.
    :param im_dim: Dimension of longest side of collection images.
    :param dim: Dimension of projection background.
    :param min_axis: Minimum axis span in interval [0, 1].
    :param ticks: Boolean specifying if the plot should display custom ticks.
    :param save_path: Path for saving as an image. If none the plot is shown in a figure.
    """

    unit = dim / 2

    center = int(np.round(unit * np.max([np.max(np.abs(pc_projections[:, [pc1, pc2]])), min_axis]))) + im_dim
    background_dim = 2 * center
    background = 255 * np.ones((background_dim, background_dim, 3), np.uint8)
    background[center-2:center+3, :, :] = np.zeros((5, background_dim, 3))
    background[:, center-2:center+3, :] = np.zeros((background_dim, 5, 3))

    for index, image in enumerate(images):
        im = load_image(image, image_dir, im_dim)
        row = center - np.round(unit * pc_projections[index, pc1])
        col = center + np.round(unit * pc_projections[index, pc2])
        insert_image(background, im, col, row)

    fig = pl.figure(figsize=(12, 12))
    fig.subplots_adjust(left=0.08, bottom=0.05, right=0.92, top=0.95, wspace=0, hspace=0)

    if ticks:
        neg_positions = np.arange(im_dim, center, (center - im_dim) / 2)[:2]
        pos_positions = np.arange(center, background_dim, (center - im_dim) / 2)[:3]
        positions = np.hstack((neg_positions, pos_positions)).astype(np.float)
        labels = (positions - center) / unit
        pl.xticks(positions, ['%.2g' % label for label in labels])
        pl.yticks(positions, ['%.2g' % label for label in -labels])
    else:
        pl.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off',
                       labelbottom='off', labelleft='off')

    pl.xlabel('Principal component {0}'.format(pc1 + 1), fontsize=12)
    pl.ylabel('Principal component {0}'.format(pc2 + 1), fontsize=12)
    pl.imshow(background)

    if save_path is not None:
        fig.savefig(save_path + 'pcs_{0}_{1}.jpg'.format(pc1 + 1, pc2 + 1), dpi=100)
        pl.close()
    else:
        pl.show()


def plot_structures(image_dir, images, structures, im_dim=100, cols=10, save_path=None):
    """ Shows all structures by plotting each set of structure images with a line between them.

    :param image_dir: Image directory.
    :param images: Image names.
    :param structures: Array of structure indices in rows.
    :param im_dim: Dimension of the longest side of the image.
    :param cols: Number of image columns. Must be greater than one. The five
                 closest images will have half the columns.
    :param save_path: Path for saving as an image. If none the plot is shown in a figure.
    """

    space = im_dim / 10
    border = 2 * space

    h_s = 0
    for structure in structures:
        r_a = math.ceil(len(structure) / float(cols))
        h_a = r_a * im_dim + (r_a - 1) * space + 2 * border
        h_s += h_a

    w = cols * im_dim + (cols - 1) * space + 2 * border
    h = h_s + (structures.shape[0] - 1) * space

    background = 255 * np.ones((h, w, 3), np.uint8)

    translation = 0
    for i, structure in enumerate(structures):
        if i > 0:
            background[translation-space+1:translation+1, border:w-border, :] = np.zeros((space, w-2*border, 3), np.uint8)

        for index, image in enumerate(images[structure]):
            im = load_image(image, image_dir, im_dim)
            row, col = get_row_col(index, translation, im_dim, space, border, cols)
            insert_image(background, im, col, row)

        translation = row + im_dim / 2 + border + space

    if save_path is not None:
        cv2.imwrite(save_path + 'structures.jpg', background[:, :, ::-1])
    else:
        fig = pl.figure()
        fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0, hspace=0)
        pl.axis('off')
        pl.imshow(background)
        pl.show()


def plot_result(image_dir, images, index_closest_group, index_representative, im_dim=100, cols=10, save_path=None):
    """ Shows the result by plotting all images on top, then the thirty
        closest images and at last the five closest in double size.

    :param image_dir: Image directory.
    :param images: Image names.
    :param index_closest_group: Indexes for the closest group of images.
    :param index_representative: Indexes for the most representative images.
    :param im_dim: Dimension of the longest side of the image.
    :param cols: Number of image columns. Must be greater than one. The five
                 closest images will have half the columns.
    :param save_path: Path for saving as an image. If none the plot is shown in a figure.
    """

    space = im_dim / 10
    border = 2 * space
    k_space = (cols - 1) / (cols / 2. - 1) if cols > 2 else 1

    r_a = math.ceil(len(images) / float(cols))
    h_a = r_a * im_dim + (r_a - 1) * space + 2 * border

    r_t = math.ceil(len(index_closest_group) / float(cols))
    h_t = r_t * im_dim + (r_t - 1) * space + 2 * border

    r_f = math.ceil(len(index_representative) / float(cols / 2))
    h_f = r_f * 2 * im_dim + (r_f - 1) * k_space * space + 2 * border

    w = cols * im_dim + (cols - 1) * space + 2 * border
    h = h_a + space + h_t + space + h_f

    background = 255 * np.ones((h, w, 3), np.uint8)
    background[h_a+1:h_a+space+1, border:w-border, :] = np.zeros((space, w-2*border, 3), np.uint8)
    background[h_a+space+h_t+1:h_a+space+h_t+space+1, border:w-border, :] = np.zeros((space, w-2*border, 3), np.uint8)

    for index, image in enumerate(images):
        im = load_image(image, image_dir, im_dim)
        row, col = get_row_col(index, 0, im_dim, space, border, cols)
        insert_image(background, im, col, row)

    for index, image in enumerate(np.array(images)[index_closest_group]):
        im = load_image(image, image_dir, im_dim)
        row, col = get_row_col(index, h_a + space, im_dim, space, border, cols)
        insert_image(background, im, col, row)

    for index, image in enumerate(np.array(images)[index_representative]):
        im = load_image(image, image_dir, 2 * im_dim)
        row, col = get_row_col(index, h_a + space + h_t + space, 2 * im_dim, k_space * space, border, cols / 2)
        insert_image(background, im, col, row)

    if save_path is not None:
        cv2.imwrite(save_path + 'representative.jpg', background[:, :, ::-1])
    else:
        fig = pl.figure()
        fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0, hspace=0)
        pl.axis('off')
        pl.imshow(background)
        pl.show()


def get_row_col(index, translation, im_dim, space, border, columns):
    """ Retrieves the middle row and middle column based on an index.

    :param index: The index.
    :param translation: The row translation.
    :param im_dim: The dimension of the image.
    :param space: The space between images.
    :param border: The image border size.
    :param columns: The number of columns.

    :return row: The row.
    :return col: The column.
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

    :param background: Background image array.
    :param im: Image array.
    :param col: The middle row to insert the image to.
    :param row: The middle column to insert the image to.
    """

    rows = im.shape[0] / 2.
    cols = im.shape[1] / 2.

    background[row-rows:row+rows, col-cols:col+cols, :] = im