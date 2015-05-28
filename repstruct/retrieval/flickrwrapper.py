import os.path
import urllib
import json

from multiprocessing import Pool


class FlickrWrapper:
    
    def __init__(self, api_key):
        """ Initializes a FlickrWrapper.

        :param api_key: The Flickr API key.
        """

        self.__api_key = api_key

    def get_urls(self, tag, count=100, sort_mode='relevance'):
        """Gets image URLs from Flickr.

        :param tag: The tag for the search.
        :param count: The number of images to download. Maximum is 500.
        :param sort_mode: One of the values in the Flickr sort mode enumeration.
                          E.g. date-posted-desc, interestingness-desc and relevance.
        """

        request = 'https://api.flickr.com/services/rest/?method=flickr.photos.search' +\
                  '&api_key={0}&tags={1}&sort={2}&per_page={3}&format=json&nojsoncallback=1'

        response = urllib.urlopen(request.format(self.__api_key, tag, sort_mode, count))
        data = json.loads(response.read())

        url = 'https://farm{0}.staticflickr.com/{1}/{2}_{3}.jpg'
        return [url.format(photo['farm'], photo['server'], photo['id'], photo['secret'])
                for photo in data['photos']['photo']]

    def download(self, data, sort_mode='relevance'):
        """ Downloads images for a tag from Flickr.

        :param data: Data set.
        :param sort_mode: One of the values in the Flickr sort mode enumeration.
                          E.g. date-posted-desc, interestingness-desc and relevance.
        """

        image_urls = self.get_urls(data.tag, data.config.collection_count, sort_mode)

        url_paths = []
        for index, image_url in enumerate(image_urls):
            url_paths.append((image_url, data.image_path + data.tag + str(index + 1) + '.jpg'))

        downloader = Downloader()
        if data.config.processes == 1:
            for url_path in url_paths:
                downloader(url_path)
        else:
            pool = Pool(data.config.processes)
            pool.map(downloader, url_paths)
        
        print 'Images downloaded'


class Downloader:

    def __call__(self, url_path):
        """ Downloads data from a url and saves it in the path.

        :param url_path: A tuple containing a url and a path.
        """

        urllib.urlretrieve(url_path[0], url_path[1])
        print 'Downloaded ' + url_path[1]