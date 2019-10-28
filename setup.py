from distutils.core import setup
setup(
  name = 'pytranskit',
  packages = ['pytranskit'],
  version = '0.1',
  license='GNU GPL',
  description = 'Python transport based signal processing toolkit',
  author = 'Abu Hasnat Mohammad Rubaiyat, Xuwang Yin, Liam Cattell',
  author_email = 'ar3fx@virginia.edu, xuwangyin@gmail.com',
  url = 'https://github.com/rohdelab/PyTransKit',
  download_url = 'https://github.com/rohdelab/PyTransKit/archive/0.1.tar.gz',
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
    'Topic :: Software Development :: Build Tools',

    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
