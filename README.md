# About repstruct

Repstruct is a python library for finding representative structures in large image collections. It is implemented according to the theory of the Master's thesis [Finding reprensentative structures in large image collections](http://www2.maths.lth.se/vision/education/pages/OscarNils09/) by Oscar Lorentzon and Nils Lundahl.

## Running
To be able to download images from flickr.com an API key is required. An API key can be obtained from flickr.com.

The rsbundler file can be run from the command line in the following way:

	$python rsbundler.py -t <tag> -a <flickrapikey>

The API key can also be provided by adding a text file called flickr_key.txt with the API key in the root of the project. Then the bundler can be run as follows:

	$python rsbundler.py -t <tag>

To view additional bundler options run the bundler with the -h flag:
	
	$python rsbundler.py -h

## Dependencies
You need to have Python 2.7+ and the following python libraries:

* [PIL](http://www.pythonware.com/products/pil/)
* [NumPy](http://numpy.scipy.org/)
* [SciPy](http://scipy.org/)
* [Matplotlib](http://matplotlib.sourceforge.net/)
* [Python Flicker API](https://pypi.python.org/pypi/flickrapi)
* [Enum34](https://pypi.python.org/pypi/enum34)

The SIFT extraction is made using [VLFeat](http://www.vlfeat.org/). The VLFeat binaries needs to be downloaded and added to the PATH.


