import matplotlib.pyplot as pyplot
from PIL import Image

def plot_images(image_files, rows, columns):
    
    i = 1
    fig = pyplot.figure()
    
    for image_file in image_files:
        image = Image.open(image_file)   
        
        sub = fig.add_subplot(rows, columns, i)
        sub.imshow(image)
        i += 1
    
    pyplot.show()

def plot_pca_projections(V, pc1, pc2):
    pyplot.figure()
    pyplot.plot(V[:,pc1], V[:,pc2], '*')
    pyplot.axhline(0)
    pyplot.axvline(0)
    pyplot.show()
    