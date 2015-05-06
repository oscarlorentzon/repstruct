import os.path
import urllib
import json

from multiprocessing import Pool


class FlickrWrapper:
    
    def __init__(self, api_key):
        """ Initializes a FlickrWrapper.

        Parameters
-       ----------
        :param api_key: The Flickr API key.
        """

        self.api_key = api_key

    def get_urls(self, tag, sort_mode='relevance'):
        """Gets image URLs from Flickr.

        Parameters
        ----------
        tag : The tag for the search.
        sort_mode : One of the values in the Flickr sort mode enumeration.
                    E.g. date-posted-desc, interestingness-desc and relevance.
        """

        request = 'https://api.flickr.com/services/rest/?method=flickr.photos.search' +\
                  '&api_key={0}&tags={1}&sort={2}&format=json&nojsoncallback=1'

        response = urllib.urlopen(request.format(self.api_key, tag, sort_mode))
        data = json.loads(response.read())

        url = "https://farm{0}.staticflickr.com/{1}/{2}_{3}.jpg"
        return [url.format(photo['farm'], photo['server'], photo['id'], photo['secret'])
                for photo in data['photos']['photo']]

    def download(self, image_dir, tag, sort_mode='relevance', processes=6):
        """ Downloads images for a tag from Flickr.
    
        Parameters
        ----------
        image_dir : The directory for sorting the images
        tag : The tag for the search.
        sort_mode : One of the values in the Flickr sort mode enumeration.
                    E.g. date-posted-desc, interestingness-desc and relevance.
        """

        image_urls = self.get_urls(tag, sort_mode)
        
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        url_paths = []
        for index, image_url in enumerate(image_urls):
            url_paths.append((image_url, image_dir + "/" + tag + str(index + 1) + ".jpg"))

        downloader = Downloader()
        if processes == 1:
            for url_path in url_paths:
                downloader(url_path)
        else:
            pool = Pool(processes)
            pool.map(downloader, url_paths)
        
        print "Images downloaded"


class Downloader:

    def __call__(self, url_path):
        """ Downloads data from a url and saves it in the path.

        :param url_path: A tuple containing a url and a path.
        """

        urllib.urlretrieve(url_path[0], url_path[1])
        print 'Downloaded ' + url_path[1]