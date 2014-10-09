from rsbundler import FlickrRsBundler
from features.featuremode import FeatureMode

with open ("flickr_key.txt", "r") as myfile: api_key=myfile.readline().rstrip()

bundler = FlickrRsBundler(api_key, 'eagle')
# bundler.run()

# bundler.download()

# bundler.extract()
# bundler.save()
# bundler.process()
# bundler.plot()

bundler.load()
bundler.process(FeatureMode.Descriptors)
bundler.plot()

bundler.process(FeatureMode.Colors)
bundler.plot()



