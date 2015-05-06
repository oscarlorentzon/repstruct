import matplotlib.pyplot as pyplot
import cv2

def plot_images(image_files, rows, columns):
    """ Plots image files in a figure with the defined
        number of rows and columns.
        
        Parameters
        ----------
        image_files : A list of image file names.
        rows : The number of rows in the figure.
        columns : The number of columns in the figure.
    """
    
    i = 1
    fig = pyplot.figure()
    
    for image_file in image_files:
        image = cv2.imread(image_file)[:,:,::-1] # Turn BGR to RGB

        sub = fig.add_subplot(rows, columns, i)
        sub.imshow(image)
        i += 1
    
    pyplot.show()

def plot_pca_projections(V, pc1, pc2):
    """ Plots the projections for the specified principal components.
        
        Parameters
        ----------
        V : The principal component projections in rows.
        pc1 : The first principal component to plot against.
        pc2 : The second principal component to plot against.
    """
    
    pyplot.figure()
    pyplot.plot(V[:,pc1], V[:,pc2], '*')
    pyplot.axhline(0)
    pyplot.axvline(0)
    pyplot.show()

def plot_points(x,y):
    """ Plots the values in y against the values in x.
        
        Parameters
        ----------
        x : An array of points.
        y : An array of points.
    """
    
    pyplot.figure()
    pyplot.plot(x, y, '*')
    pyplot.show()
    