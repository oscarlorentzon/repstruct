# About repstruct

Repstruct is a python library for finding representative structures in large image collections. It is implemented according to the theory of the Master's thesis [Finding reprensentative structures in large image collections](http://www2.maths.lth.se/vision/education/pages/OscarNils09/) by Oscar Lorentzon and Nils Lundahl.

The results are obtained by an approach using bags of visual words and colors. The implementation extracts SIFT descriptors as well as colors from the images in the collection and creates feature vectors from histograms created by classifying the features against training data. A collection of images is downloaded from Flickr for a specified tag and the results of the algorithm are presented by plotting the group of closest images and then the most representative.

## Running
To be able to download images from [Flickr](https://www.flickr.com/) an API key is required. An API key can be requested from [Flickr's App Garden](https://www.flickr.com/services/apps/create/).

The rsbundler file can be run from the command line in the following way:

	bin/run TAG -a FLICKR_API_KEY

The API key can also be provided by adding a text file called flickr_key.txt with the API key in the root of the project. Then the bundler can be run as follows:

	bin/run TAG

To view additional bundler options run the bundler with the -h flag:
	
	bin/run -h
	
## Continuous integration

[![Build Status](https://travis-ci.org/oscarlorentzon/repstruct.svg?branch=master)](https://travis-ci.org/oscarlorentzon/repstruct)
[![Coverage Status](https://coveralls.io/repos/oscarlorentzon/repstruct/badge.svg?branch=master)](https://coveralls.io/r/oscarlorentzon/repstruct?branch=master)

## Dependencies
You need to have Python 2.7+ and the following libraries to run the algorithm:

* [OpenCV][]
* [SciPy][]
* [NumPy][]
* [Matplotlib][]
* [Enum34][]
* [PyYAML][]

The following libraries are required to run the tests:

* [Setuptools][]
* [Mock][]
* [Nose][]

### Installing dependencies on Ubuntu

1. [OpenCV][] - Install by following the steps in the Ubuntu OpenCV [installation guide](https://help.ubuntu.com/community/OpenCV).

2. [NumPy][], [SciPy][], [PyYAML][], [Enum34][], [Setuptools][], [Mock][], [Nose][] and [Matplotlib][] - Install [pip](https://pypi.python.org/pypi/pip) and run:

        sudo apt-get install gfortran
        sudo pip install -r requirements.txt
        sudo apt-get install python-matplotlib

## Example output

The images below show the result from a run using the tag **steppyramid.** The first output image shows the collection images plotted against their feature vector projection onto the third and fourth principal components. 

![PCA](example/pca.jpg)

The second output image shows the result after running the algorithm. On top all collection images are shown, in the middle the thirty closest images are shown and at the bottom the five most representative images are shown.

![Representative](example/representative.jpg)

The third output image shows the result for finding all structures ordered according to a score based on the representative result.

![Structures](example/structures.jpg)

[OpenCV]: http://opencv.org/ (Computer vision and machine learning software library)
[NumPy]: http://www.numpy.org/ (Scientific computing with Python)
[SciPy]: http://www.scipy.org/ (Fundamental library for scientific computing)
[Matplotlib]: http://matplotlib.sourceforge.net (Plotting in python)
[Enum34]: https://pypi.python.org/pypi/enum34 (Enum support in python 2.*)
[PyYAML]: http://pyyaml.org/ (YAML implementations for Python)
[Setuptools]: http://pythonhosted.org/setuptools/ (Python project packaging)
[Mock]: http://www.voidspace.org.uk/python/mock/ (Mocking and testing library)
[Nose]: https://nose.readthedocs.org/en/latest/ (Unit test extensions)