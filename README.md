# About repstruct

Repstruct is a python library for finding representative structures in large image collections. It is implemented according to the theory of the Master's thesis [Finding reprensentative structures in large image collections](http://www2.maths.lth.se/vision/education/pages/OscarNils09/) by Oscar Lorentzon and Nils Lundahl.

## Running
To be able to download images from flickr.com an API key is required. An API key can be obtained from flickr.com. The API key should be provided by adding a text file called flickr_key.txt with the API key in the root of the project.

## Dependencies
You need to have Python 2.7+ and the following python libraries:

* [PIL](http://www.pythonware.com/products/pil/)
* [NumPy](http://numpy.scipy.org/)
* [SciPy](http://scipy.org/)
* [Matplotlib](http://matplotlib.sourceforge.net/)
* [Python Flicker API](https://pypi.python.org/pypi/flickrapi)

The SIFT extraction is made using [VLFeat](http://www.vlfeat.org/). The VLFeat binaries needs to be downloaded and added to the PATH.


