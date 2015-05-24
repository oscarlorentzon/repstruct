import os.path as op
import sys
import getopt

from retrieval.flickrwrapper import FlickrWrapper
from analysis import process
from display import plothelper
from features import sift, extract
from runmode import RunMode
from dataset.dataset import DataSet


class RsBundler:

    def __init__(self, api_key, tag):
        self.__data = DataSet(tag, op.dirname(op.abspath(__file__)))
        self.__flickr = FlickrWrapper(api_key)

    def run(self):
        self.download()
        self.extract()
        self.process()
        self.plot_result()
        
    def download(self):
        self.__flickr.download(self.__data)

    def extract(self):
        sift.extract(self.__data)
        extract.extract(self.__data)

    def process(self):
        process.process(self.__data)
        process.closest(self.__data)
        
    def plot(self):
        images, pc_projections, pcs = process.load_principal_components(self.__data.result_path)
        closest_group, representative = process.load_closest(self.__data.result_path)

        plothelper.plot_pca_projections(pc_projections, 1, 2)
        plothelper.plot_pca_projections(pc_projections, 3, 4)
        plothelper.plot_images(self.__data.image_path, images[closest_group], 3, 10)
        plothelper.plot_images(self.__data.image_path, images[representative], 1, 5)

    def plot_result(self):
        images, pc_projections, pcs = process.load_principal_components(self.__data.result_path)
        closest_group, representative = process.load_closest(self.__data.result_path)

        save_path = self.__data.plot_path if self.__data.config.save_plot else None

        plothelper.plot_pca_images(self.__data.image_path, images, pc_projections, 1, 2,
                                   save_path=save_path, ticks=self.__data.config.ticks)
        plothelper.plot_pca_images(self.__data.image_path, images, pc_projections, 3, 4,
                                   save_path=save_path, ticks=self.__data.config.ticks)

        plothelper.plot_result(self.__data.image_path, images, closest_group, representative, save_path=save_path)
 
             
def main(argv):
    api_key = None
    tag = None
    run_mode = RunMode.Download
    
    help_text = """To run the bundler from command line enter:
              
               rsbundler.py -t <tag> -a <apikey>
    
               The following options are available:
    
               Required:
               -t : The tag to search for
               -a : The flickr api key
               
               Optional:
               -r : The run mode. Possible values:
                    d - Downloads, extracts, saves and processes images.
                        This is the default.
                    e - Extracts, saves and processes images.
                    o - Loads saved data and processes images.
               """
    
    try:
        opts, args = getopt.getopt(argv, 'ha:t:f:r:', ['apikey=', 'tag=', 'featuremode=', 'runmode='])
    except getopt.GetoptError:
        print help_text
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print help_text
            sys.exit()
        elif opt in ('-a', '--apikey'):
            api_key = arg
        elif opt in ('-t', '--tag'):
            tag = arg
        elif opt in ('-r', '--runmode'):
            if arg == 'e':
                run_mode = RunMode.Extract
            elif arg == 'o':
                run_mode = RunMode.Load
    
    if tag is None:   
        sys.exit('Tag is required. Usage: ' + help_text)
    
    if api_key is None:
        with open('flickr_key.txt', 'r') as f_out:
            api_key = f_out.readline().rstrip()

    bundler = RsBundler(api_key, tag)
    
    if run_mode == RunMode.Download:
        bundler.download()
        bundler.extract()
    elif run_mode == RunMode.Extract:
        bundler.extract()
        
    bundler.process()
    bundler.plot_result()


if __name__ == '__main__':
    main(sys.argv[1:])