import os.path
import urllib
import json

class FlickrWrapper:
    
    def __init__(self, api_key):
        self.api_key = api_key

    def get_urls(self, tag, sort_mode='relevance'):

        request = 'https://api.flickr.com/services/rest/?method=flickr.photos.search' +\
                  '&api_key={0}&tags={1}&sort={2}&format=json&nojsoncallback=1'

        response = urllib.urlopen(request.format(self.api_key, tag, sort_mode))
        data = json.loads(response.read())

        url = "https://farm{0}.staticflickr.com/{1}/{2}_{3}.jpg"
        return [url.format(photo.get('farm'), photo.get('server'), photo.get('id'), photo.get('secret'))
                for photo in data['photos']['photo']]

    def download(self, image_dir, tag, sort_mode='relevance'):
        """ Downloads images for a tag from Flickr.
    
        Parameters
        ----------
        image_dir : The directory for sorting the images
        api_key : The Flickr API key.
        tag : The tag for the search.
        sort_mode : One of the values in the Flickr sort mode enumeration.
                    E.g. date-posted-desc, interestingness-desc and relevance.
        """

        image_urls = self.get_urls(tag, sort_mode)
        
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
    
        i = 1
        for image_url in image_urls:
            urllib.urlretrieve(image_url, image_dir + "/" + tag + str(i) + ".jpg")
            i += 1
        
        print "images downloaded"

