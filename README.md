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

## Transport-based Transforms
### Cumulative Distribution Transform (CDT)
Let <img src="http://www.sciweavers.org/tex2img.php?eq=s%28%5Cmathbf%7Bx%7D%29%2C%20%5Cmathbf%7Bx%7D%5Cin%5COmega_s%5Csubseteq%5Cmathbb%7BR%7D&bc=White&fc=Black&im=ps&fs=14&ff=anttor&edit=0" align="center" border="0" alt="s(\mathbf{x}), \mathbf{x}\in\Omega_s\subseteq\mathbb{R}" width="" height="" /> be a positive density function (PDF), and <img src="http://www.sciweavers.org/tex2img.php?eq=s_0%28%5Cmathbf%7Bx%7D%29%2C%20%5Cmathbf%7Bx%7D%5Cin%5COmega_%7Bs_0%7D%5Csubseteq%5Cmathbb%7BR%7D&bc=White&fc=Black&im=ps&fs=14&ff=anttor&edit=0" align="center" border="0" alt="s_0(\mathbf{x}), \mathbf{x}\in\Omega_{s_0}\subseteq\mathbb{R}" width="" height="" /> be a reference PDF.

### Radon-Cumulative Distribution Transform (R-CDT)

### Continuous Linear Optimal Transport Transform

## Tutorials
- Cumulative Distribution Transform (CDT) tutorial [[notebook](https://github.com/rohdelab/PyTransKit/blob/master/tutorials/01_tutorial_cdt.ipynb)] [[nbviwer](https://nbviewer.jupyter.org/github/rohdelab/PyTransKit/blob/master/tutorials/01_tutorial_cdt.ipynb)]
- Radon-CDT tutorial [[notebook](https://github.com/rohdelab/PyTransKit/blob/master/tutorials/02_tutorial_rcdt.ipynb)] [[nbviewer](https://nbviewer.jupyter.org/github/rohdelab/PyTransKit/blob/master/tutorials/02_tutorial_rcdt.ipynb)]
- 3D Radon-CDT tutorial [[notebook](https://github.com/rohdelab/PyTransKit/blob/master/tutorials/05_tutorial_rcdt3D.ipynb)][[nbviewer](https://nbviewer.jupyter.org/github/rohdelab/PyTransKit/blob/master/tutorials/05_tutorial_rcdt3D.ipynb)]
- Radon-CDT Nearest Subspace (RCDT-NS) classifier [[notebook](https://github.com/rohdelab/PyTransKit/blob/master/tutorials/03_tutorial_RCDT-NS_classifier.ipynb)] [[nbviewer](https://nbviewer.jupyter.org/github/rohdelab/PyTransKit/blob/master/tutorials/03_tutorial_RCDT-NS_classifier.ipynb)]
- CDT Nearest Subspace (CDT-NS) classifier for 1D data [[notebook](https://github.com/rohdelab/PyTransKit/blob/master/tutorials/04_tutorial_CDT-NS_classifier.ipynb)] [[nbviewer](https://nbviewer.jupyter.org/github/rohdelab/PyTransKit/blob/master/tutorials/04_tutorial_CDT-NS_classifier.ipynb)]
- 3D Radon-CDT Nearest Subspace (3D-RCDT-NS) classifier for 3D data [[notebook](https://github.com/rohdelab/PyTransKit/blob/master/tutorials/06_tutorial_3DRCDT-NS_classifier.ipynb)] [[nbviewer](https://nbviewer.jupyter.org/github/rohdelab/PyTransKit/blob/master/tutorials/06_tutorial_3DRCDT-NS_classifier.ipynb)]
- Continuous Linear Optimal Transport Transform (CLOT) tutorial [[notebook](https://github.com/rohdelab/PyTransKit/blob/master/tutorials/07_tutorial_clot.ipynb)] [[nbviewer](https://nbviewer.jupyter.org/github/rohdelab/PyTransKit/blob/master/tutorials/07_tutorial_clot.ipynb)]

## Examples
- Time delay estimation using CDT [[notebook](https://github.com/rohdelab/PyTransKit/blob/master/Examples/Example01_estimation_delay.ipynb)] [[nbviewer](https://nbviewer.jupyter.org/github/rohdelab/PyTransKit/blob/master/Examples/Example01_estimation_delay.ipynb)]
- Time delay and linear dispersion estimation using CDT [[notebook](https://github.com/rohdelab/PyTransKit/blob/master/Examples/Example02_estimation_delay_linear_dispersion.ipynb)] [[nbviewer](https://nbviewer.jupyter.org/github/rohdelab/PyTransKit/blob/master/Examples/Example02_estimation_delay_linear_dispersion.ipynb)]

## Resources
- External website http://imagedatascience.com/transport/

