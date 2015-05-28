from setuptools import setup


packages = ['analysis',
            'dataset',
            'display',
            'features',
            'retrieval']

package_data = {
    'data': ['*.mat']
}

setup(
    name='repstruct',
    version="0.1",
    packages=packages,
    package_data=package_data,
    author='Oscar Lorentzon',
    description='Library for finding representative structures in large image collections.',
    license='BSD 3-Clause',
    keywords='image analysis pca bag of words represent',
    url='https://github.com/oscarlorentzon/repstruct',
    test_suite='features.test'
)
