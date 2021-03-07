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
#### Definition
Let ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20s%28%5Cmathbf%7Bx%7D%29%2C%20%5Cmathbf%7Bx%7D%5Cin%5COmega_%7Bs%7D%5Csubseteq%5Cmathbb%7BR%7D) be a positive density function (PDF). The CDT of the PDF 
![equation](https://latex.codecogs.com/svg.latex?s%28%5Cmathbf%7Bx%7D%29) 
with respect to a reference PDF ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20s_0%28%5Cmathbf%7Bx%7D%29%2C%20%5Cmathbf%7Bx%7D%5Cin%5COmega_%7Bs_0%7D%5Csubseteq%5Cmathbb%7BR%7D) is given by the mass preserving function ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cwidehat%7Bs%7D%28%5Cmathbf%7Bx%7D%29) that satisfies - 

![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cint_%7B%5Cinf%28%5COmega_s%29%7D%5E%7B%5Chat%7Bs%7D%28x%29%7D%20s%28u%29du%20%3D%20%5Cint_%7B%5Cinf%28%5COmega_%7Bs_0%7D%29%7D%5E%7Bx%7D%20s_0%28u%29du)

which yields 

![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cwidehat%7Bs%7D%28x%29%20%3D%20S%5E%7B-1%7D%28S_0%28x%29%29) ,

where, ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20S%28x%29%20%3D%20%5Cint_%7B-%5Cinfty%7D%5E%7Bx%7D%20s%28x%29dx)    and    

![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20S_0%28x%29%20%3D%20%5Cint_%7B-%5Cinfty%7D%5E%7Bx%7D%20s_0%28x%29dx)

#### Tutorial
- Cumulative Distribution Transform (CDT) tutorial [[notebook](https://github.com/rohdelab/PyTransKit/blob/master/tutorials/01_tutorial_cdt.ipynb)] [[nbviwer](https://nbviewer.jupyter.org/github/rohdelab/PyTransKit/blob/master/tutorials/01_tutorial_cdt.ipynb)]

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

