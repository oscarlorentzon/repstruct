import flickrapi

def get_image_urls(api_key,tags,sort_mode='date-posted-desc'):
    """Gets image URLs from Flickr.

    Keyword arguments:
    api_key -- The Flickr API key.
    tag -- The tag for the search.
    sort_mode -- One of the values in the Flickr sort mode enumeration.
    """
    
    flickrApi = flickrapi.FlickrAPI("5145b16c5f46c546da37da57f7dd9bd3")
    photos = flickrApi.photos_search(tags=tags,sortmode=sort_mode)
    
    return ["https://farm{0}.staticflickr.com/{1}/{2}_{3}.jpg".format(photo.get('farm'), photo.get('server'), photo.get('id'), photo.get('secret')) for photo in photos[0]]

