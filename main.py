from rsbundler import FlickrRsBundler

with open ("flickr_key.txt", "r") as myfile: api_key=myfile.readline().rstrip()

bundler = FlickrRsBundler(api_key, 'turningtorso')
bundler.download()
bundler.extract()
bundler.save()
bundler.process()
bundler.plot()

