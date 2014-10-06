from rsbundler import FlickrRsBundler

with open ("flickr_key.txt", "r") as myfile: api_key=myfile.readline().rstrip()

bundler = FlickrRsBundler(api_key, 'statueofliberty')
bundler.extract()
bundler.process()
bundler.plot()

