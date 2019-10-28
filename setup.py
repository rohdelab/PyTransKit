import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
  name = 'pytranskit',
  packages = ['pytranskit'],
  version = '0.01',
  license='GNU GPL',
  description = 'Python transport based signal processing toolkit.',
  long_description=long_description,
  long_description_content_type='text/markdown',
  author = 'Abu Hasnat Mohammad Rubaiyat, Xuwang Yin, Liam Cattell',
  author_email = 'ar3fx@virginia.edu, xuwangyin@gmail.com',
  url = 'https://github.com/rohdelab/PyTransKit',
  download_url = 'https://github.com/rohdelab/PyTransKit/archive/0.01.tar.gz',
  keywords = ['optimal transport', 'cumulative distribution transform', 'signal processing', 'pattern recognition'],
  install_requires=[
          'six',
          'scikit-image',
          'scipy',
          'numpy',
          'scikit-learn',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',

    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
