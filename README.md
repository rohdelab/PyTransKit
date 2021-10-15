# PyTransKit
Python Transport Based Signal Processing Toolkit

Website and documentation: https://pytranskit.readthedocs.io/


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

## Low Level Functions
### CDT, SCDT
- Cumulative Distribution Transform (CDT) [1] tutorial [[notebook](https://github.com/rohdelab/PyTransKit/blob/master/tutorials/01_tutorial_cdt.ipynb)] [[nbviewer](https://nbviewer.jupyter.org/github/rohdelab/PyTransKit/blob/master/tutorials/01_tutorial_cdt.ipynb)]
- Signed Cumulative Distribution Transform (SCDT) [6] tutorial [[notebook](https://github.com/rohdelab/PyTransKit/blob/master/tutorials/09_tutorial_SCDT_classification.ipynb)] [[nbviewer](https://nbviewer.jupyter.org/github/rohdelab/PyTransKit/blob/master/tutorials/09_tutorial_SCDT_classification.ipynb)]
- SCDT tutorial with domain adaptation [[notebook](https://github.com/rohdelab/PyTransKit/blob/master/tutorials/10_tutorial_SCDT.ipynb)] [[nbviewer](https://nbviewer.jupyter.org/github/rohdelab/PyTransKit/blob/master/tutorials/10_tutorial_SCDT.ipynb)]
### R-CDT
- Radon-CDT tutorial [[notebook](https://github.com/rohdelab/PyTransKit/blob/master/tutorials/02_tutorial_rcdt.ipynb)] [[nbviewer](https://nbviewer.jupyter.org/github/rohdelab/PyTransKit/blob/master/tutorials/02_tutorial_rcdt.ipynb)]
- 3D Radon-CDT tutorial [[notebook](https://github.com/rohdelab/PyTransKit/blob/master/tutorials/05_tutorial_rcdt3D.ipynb)][[nbviewer](https://nbviewer.jupyter.org/github/rohdelab/PyTransKit/blob/master/tutorials/05_tutorial_rcdt3D.ipynb)]
### CLOT
- Continuous Linear Optimal Transport Transform (CLOT) tutorial [[notebook](https://github.com/rohdelab/PyTransKit/blob/master/tutorials/07_tutorial_clot.ipynb)] [[nbviewer](https://nbviewer.jupyter.org/github/rohdelab/PyTransKit/blob/master/tutorials/07_tutorial_clot.ipynb)]

## Classification Examples
- CDT Nearest Subspace (CDT-NS) classifier for 1D data [[notebook](https://github.com/rohdelab/PyTransKit/blob/master/tutorials/04_tutorial_CDT-NS_classifier.ipynb)] [[nbviewer](https://nbviewer.jupyter.org/github/rohdelab/PyTransKit/blob/master/tutorials/04_tutorial_CDT-NS_classifier.ipynb)]
- SCDT Nearest Subspace (SCDT-NS) classifier for 1D data [8] [[notebook](https://github.com/rohdelab/PyTransKit/blob/master/tutorials/11_tutorial_SCDT-NS_classifier.ipynb)] [[nbviewer](https://nbviewer.org/github/rohdelab/PyTransKit/blob/master/tutorials/11_tutorial_SCDT-NS_classifier.ipynb)]
- Radon-CDT Nearest Subspace (RCDT-NS) classifier for 2D data [4] [[notebook](https://github.com/rohdelab/PyTransKit/blob/master/tutorials/03_tutorial_RCDT-NS_classifier.ipynb)] [[nbviewer](https://nbviewer.jupyter.org/github/rohdelab/PyTransKit/blob/master/tutorials/03_tutorial_RCDT-NS_classifier.ipynb)]
- 3D Radon-CDT Nearest Subspace (3D-RCDT-NS) classifier for 3D data [[notebook](https://github.com/rohdelab/PyTransKit/blob/master/tutorials/06_tutorial_3DRCDT-NS_classifier.ipynb)] [[nbviewer](https://nbviewer.jupyter.org/github/rohdelab/PyTransKit/blob/master/tutorials/06_tutorial_3DRCDT-NS_classifier.ipynb)]

## Estimation Examples
- Time delay estimation using CDT [5] [[notebook](https://github.com/rohdelab/PyTransKit/blob/master/Examples/Example01_estimation_delay.ipynb)] [[nbviewer](https://nbviewer.jupyter.org/github/rohdelab/PyTransKit/blob/master/Examples/Example01_estimation_delay.ipynb)]
- Time delay and linear dispersion estimation using CDT [5] [[notebook](https://github.com/rohdelab/PyTransKit/blob/master/Examples/Example02_estimation_delay_linear_dispersion.ipynb)] [[nbviewer](https://nbviewer.jupyter.org/github/rohdelab/PyTransKit/blob/master/Examples/Example02_estimation_delay_linear_dispersion.ipynb)]

## Transport-based Morphometry
- Transport-based Morphometry to detect and visualize cell phenotype differences [7] [[notebook](https://github.com/rohdelab/PyTransKit/blob/master/tutorials/08_tutorial_TBM.ipynb)] [[nbviewer](https://nbviewer.jupyter.org/github/rohdelab/PyTransKit/blob/master/tutorials/08_tutorial_TBM.ipynb)]


## References
1. [The cumulative distribution transform and linear pattern classification, Applied and Computational Harmonic Analysis, November 2018](http://www.sciencedirect.com/science/article/pii/S1063520317300076)
2. [The Radon Cumulative Distribution Transform and Its Application to Image Classification, IEEE Transactions on Image Processing, December 2015](https://ieeexplore.ieee.org/document/7358128)
3. [A continuous linear optimal transport approach for pattern analysis in image datasets, Pattern Recognition, March 2016](https://www.sciencedirect.com/science/article/abs/pii/S0031320315003507)
4. [Radon cumulative distribution transform subspace modeling for image classification, Journal of Mathematical Imaging and Vision, 2021](https://link.springer.com/article/10.1007/s10851-021-01052-0)
5. [Parametric Signal Estimation Using the Cumulative Distribution Transform, IEEE Transactions on Signal Processing, May 2020](https://ieeexplore.ieee.org/abstract/document/9099391)
6. [The Signed Cumulative Distribution Transform for 1-D Signal Analysis and Classification, ArXiv 2021](https://arxiv.org/abs/2106.02146)
7. [Detecting and visualizing cell phenotype differences from microscopy images using transport-based morphometry, PNAS 2014](https://www.pnas.org/content/111/9/3448.short)
8. [Nearest Subspace Search in the Signed Cumulative Distribution Transform Space for 1D Signal Classification, ArXiv 2021](https://arxiv.org/abs/2110.05606)

## Resources
External website http://imagedatascience.com/transport/
