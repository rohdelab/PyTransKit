import setuptools

with open("README.md", "r") as fh:
    long_description=fh.read()

setuptools.setup(
  name="pytranskit",
  version="0.2.2",
  author="Abu Hasnat Mohammad Rubaiyat, Xuwang Yin, Liam Cattell, Soheil Kolouri, Mohammad Shifat-E-Rabbi, Yan Zhuang, Gustavo Rohde",
  author_email="ar3fx@virginia.edu, xuwangyin@gmail.com, gustavo.rohde@gmail.com",
  description="Python transport based signal processing toolkit.",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/rohdelab/PyTransKit",
  packages=setuptools.find_packages(),
  classifiers=[
    "Development Status :: 3 - Alpha",

    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",

    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.4",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
  ],
  python_requires='>=3.6',
)
