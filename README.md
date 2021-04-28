# mikit
`mikit` is a package for python that supports chemical equation-based machine learning. 

This module has the following features.
 - Extraction of mole ratios containing polyatomic molecules (such as NH3) from chemical formulas
 - Output of feature values considering chemical properties and mole ratios of constituent elements
 - Feature reduction considering R2 between features (Filter method)
 - Feature reduction by forward feature selection (Wrapper method)
 - Batch Bayesian Optimization
 - Output of plots and contours in pseudo three-component system diagrams

# Requirement
## Requirements
- python  v3.7+
- [pandas v1.1+](https://pandas.pydata.org/)
- [matplotlib v3.3+](http://matplotlib.org/)
- [numpy v1.19+](http://www.numpy.org/)
- [scipy v1.5+](https://www.scipy.org/)
- [scikit-learn v0.23+](http://scikit-learn.org/stable)
- [tqdm](https://github.com/noamraph/tqdm)
- and some basic packages.


## Install
```
pip install git+https://github.com/Ottomossei/mikit
```

# mikit.compname module
## Usage

Import and initialize

```
from mikit.compname import ChemFormula
cn = ChemFormula()
```

Output (automatically calculate molar ratio from chemical formula)

```
all_molratio = cn.get_molratio(comp)
f_molratio = cn.get_molratio(comp, tar_atoms=["F"])
cation_molratio = cn.get_molratio(comp, exc_atoms=["F"])
```

`comp` is chemical formulas (Ex:["H2O", "CO2"])
`tar_atoms` is list of atoms to be used in the calculation.
`exc_atoms` is list of atoms to be excluded from the calculation

# More detailed usage
Detailed usages can be found at the following link.
 - [Learning process](https://github.com/Ottomossei/mikit/blob/main/example/machine_learning/learning.ipynb)
 - [Predicting process](https://github.com/Ottomossei/mikit/blob/main/example/machine_learning/predict.ipynb)
 - [Bayesian Optimization](https://github.com/Ottomossei/mikit/blob/main/example/bayes/bayes.ipynb)

## Author
If you have any troubles or questions, please contact [Ottomossei](https://github.com/Ottomossei).

January, 2021