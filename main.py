import urllib
import os.path
from numpy import shape, array
from os import listdir
from os.path import isfile, join
from flickrdownloader import get_image_urls
from sift import process
from PIL import Image
from pylab import arange, cos, sin, plot, figure, gray, imshow, axis, show, pi
import scipy.io
from descriptor import test

def plot_features(image, locations, circle=False):
    """ Show image with features. input: image (image as array), 
        locations (row, col, scale, orientation of each feature). """

    def draw_circle(center, radius):
        t = arange(0,1.01,.01) * 2 * pi
        x = radius * cos(t) + center[0]
        y = radius * sin(t) + center[1]
        plot(x, y, 'b', linewidth=2)

    figure()
    gray()
    imshow(image)
    if circle:
        for location in locations:
            draw_circle(location[:2], location[2]) 
    else:
        plot(locations[:,0], locations[:,1], 'ob')
    axis('off')
    show()

api_key = '5145b16c5f46c546da37da57f7dd9bd3'
tag = 'goldengatebridge'

# Get image urls from flickr api
image_urls = get_image_urls(api_key, tag)

image_dir = os.path.dirname(os.path.abspath(__file__)) + "/images/" + tag
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# download images from flickr
i = 1
for image_url in image_urls[:2]:
    
    image_path = image_dir + "/" + tag + str(i) + ".jpg"
    urllib.urlretrieve(image_url, image_path)
    i += 1

# get a list of all downloaded images    
image_files = [join(image_dir,f) for f in listdir(image_dir) if isfile(join(image_dir,f))]

descriptor_data = scipy.io.loadmat('data/kmeans_descriptor_results.mat')

# extract descriptors for all images
for image_file in image_files:
    
    image = Image.open(image_file)
    shape = array(image).shape
    
    if len(shape) == 3:
        image = image.convert('L')
        
    locations, descriptors = process.process_image(image)
    
    test(descriptors, descriptor_data.get('cbest'), descriptor_data.get('idxbest'))
    
    plot_features(image, locations, True)
