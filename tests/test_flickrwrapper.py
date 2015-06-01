import unittest
from mock import Mock, patch, PropertyMock

from repstruct.retrieval.flickrwrapper import *
from repstruct.dataset import *

class TestFlickrWrapper(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testDownloader(self):
        urllib.urlretrieve = Mock(return_value=0)

        url_path = ('url', 'path')

        downloader = Downloader()
        downloader(url_path)

        urllib.urlretrieve.assert_called_with(url_path[0], url_path[1])

    @patch('urllib.urlopen')
    def testFlickrWrapperGetUrls(self, mock_urlopen):
        response = {'photos': {'photo': [
            {'id': 'id1', 'secret': 'secret1', 'server': 'server1', 'farm': 1},
            {'id': 'id2', 'secret': 'secret2', 'server': 'server2', 'farm': 2}
        ]}}
        response = json.dumps(response)

        m = Mock()
        m.read = Mock(return_value=response)
        mock_urlopen.return_value = m

        api_key = 'key'
        tag = 'tag'
        wrapper = FlickrWrapper(api_key)
        result = wrapper.get_urls(tag)

        self.assertEqual(2, len(result))

        self.assertTrue('farm1' in result[0])
        self.assertTrue('id1' in result[0])
        self.assertTrue('secret1' in result[0])
        self.assertTrue('server1' in result[0])

        self.assertTrue('farm2' in result[1])
        self.assertTrue('id2' in result[1])
        self.assertTrue('secret2' in result[1])
        self.assertTrue('server2' in result[1])

    def testFlickrWrapperDownload(self):

        data = DataSet('tag')
        data.collection = PropertyMock()
        data.collection.path = 'path'
        data.collection.config = PropertyMock()
        data.collection.config.count = 2
        data.collection.config.processes = 1
        sort_mode = 'sort_mode'

        api_key = 'key'
        urls = ['url1', 'url2']
        wrapper = FlickrWrapper(api_key)
        wrapper.get_urls = Mock(return_value=urls)

        urllib.urlretrieve = Mock(return_value=0)

        with patch('multiprocessing.Pool') as mock:
            instance = mock.return_value
            instance.map = Mock(return_value=0)
            wrapper.download(data, sort_mode)

            self.assertEqual(0, instance.map.call_count)

        self.assertEqual(len(urls), urllib.urlretrieve.call_count)

        wrapper.get_urls.assert_called_with(data.tag, data.collection.config.count, sort_mode)

    def testFlickrWrapperDownloadPool(self):

        data = DataSet('tag')
        data.collection = PropertyMock()
        data.collection.path = 'path'
        data.collection.config = PropertyMock()
        data.collection.config.count = 2
        data.collection.config.processes = 2
        sort_mode = 'sort_mode'

        api_key = 'key'
        urls = ['url1', 'url2']
        wrapper = FlickrWrapper(api_key)
        wrapper.get_urls = Mock(return_value=urls)

        urllib.urlretrieve = Mock(return_value=0)

        with patch('multiprocessing.Pool') as mock:
            instance = mock.return_value
            instance.map = Mock(return_value=0)
            wrapper.download(data, sort_mode)

            self.assertEqual(1, instance.map.call_count)

        self.assertEqual(0, urllib.urlretrieve.call_count)

        wrapper.get_urls.assert_called_with(data.tag, data.collection.config.count, sort_mode)


if __name__ == '__main__':
    unittest.main()