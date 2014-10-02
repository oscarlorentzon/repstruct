import flickrapi
import os.path
import urllib

class FlickrWrapper:
    
    def __init__(self, api_key):
        self.api_key = api_key
        
    def get_image_urls(self, tag, sort_mode='date-posted-desc'):
        """Gets image URLs from Flickr.
    
        Keyword arguments:
        api_key -- The Flickr API key.
        tag -- The tag for the search.
        sort_mode -- One of the values in the Flickr sort mode enumeration.
        """
        
        flickrApi = flickrapi.FlickrAPI(self.api_key)
        photos = flickrApi.photos_search(tags=tag,sortmode=sort_mode)
        
        return ["https://farm{0}.staticflickr.com/{1}/{2}_{3}.jpg".format(photo.get('farm'), photo.get('server'), photo.get('id'), photo.get('secret')) for photo in photos[0]]

    def download_images(self, image_dir, tag, sort_mode='date-posted-desc'):
        """ Downloads images for a tag from Flickr.
    
        Keyword arguments:
        image_dir -- The diriectory for sotring the images
        api_key -- The Flickr API key.
        tag -- The tag for the search.
        sort_mode -- One of the values in the Flickr sort mode enumeration.
        """

        image_urls = self.get_image_urls(tag, sort_mode)
        
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
    
        i = 1
        for image_url in image_urls:
            urllib.urlretrieve(image_url, image_dir + "/" + tag + str(i) + ".jpg")
            i += 1
        
        print "images downloaded"

