# PyTransKit
Python Transport Based Signal Processing Toolkit

## Installation
The library could be installed through pip
```
pip install pytranskit
```
Alternately, you could clone/download the repository and add the `pytranskit` directory to your Python path
```
import sys
sys.path.append('path/to/pytranskit')

from pytranskit.optrans.continuous.cdt import CDT
```

## Tutorials
- Cumulative Distribution Transform (CDT) tutorial [[notebook](https://github.com/rohdelab/PyTransKit/blob/master/tutorials/01_tutorial_cdt.ipynb)] [[nbviwer](https://nbviewer.jupyter.org/github/rohdelab/PyTransKit/blob/master/tutorials/01_tutorial_cdt.ipynb)]
- Radon-CDT tutorial [[notebook](https://github.com/rohdelab/PyTransKit/blob/master/tutorials/02_tutorial_rcdt.ipynb)] [[nbviewer](https://nbviewer.jupyter.org/github/rohdelab/PyTransKit/blob/master/tutorials/02_tutorial_rcdt.ipynb)]
- Signal compression with DCT-CDT transform [[notebook](https://github.com/rohdelab/PyTransKit/blob/master/tutorials/Example_01_CDT-DCT-Reconstruction.ipynb)] [[nbviewer](https://nbviewer.jupyter.org/github/rohdelab/PyTransKit/blob/master/tutorials/Example_01_CDT-DCT-Reconstruction.ipynb)]

## Resources
- Overview of ideas/theory/applications https://www.dropbox.com/s/zrhimspwbtpgh12/Lagrangian_transform_19_final.pptx?dl=0
- External website http://imagedatascience.com/transport/
