[![Build Status](https://travis-ci.com/arteagac/pymlogit.svg?branch=master)](https://travis-ci.com/arteagac/pymlogit)

# mixedlogit
GPU accelerated estimation of mixed, multinomial, and conditional logit models in python.

### Example:
The following example analyzes choices of fishing modes. See the data [here](examples/data/fishing_long.csv) and more information about the data [here](https://doi.org/10.1162/003465399767923827). The parameters are:
- `X`: Data matrix in long format (numpy array, shape [n_samples, n_fvariables])
- `y`: Binary vector of choices (numpy array, shape [n_samples, ])
- `varnames`: List of variable names. Its length must match number of columns in `X`
- `alternatives`:  List of alternatives names or codes.
- `asvars`: List of alternative specific variables
- `randvars`: Variables with random distribution. `"n"` for normal and `"ln"` for log normal.

The current version of `mixedlogit` only supports data in long format.

#### Usage
```python
# Read data from CSV file
import pandas as pd
df = pd.read_csv("examples/data/fishing_long.csv")

varnames = ['price','catch']
X = df[varnames].values
y = df['choice'].values

# Fit the model with mixedlogit
from mixedlogit import ChoiceModel
import numpy as np
np.random.seed(0)

model = ChoiceModel()
model.fit(X,y, varnames= varnames,
          alternatives=['beach','boat','charter','pier'],
          asvars=['price','catch'],
          randvars={'price': 'n', 'catch': 'n'})
model.summary()
```

#### Output
```
Optimization succesfully completed after 28 iterations. 
----------------------------------------------------------------------------------------
Coefficient          	Estimate 	Std. Error 	z-value 	P(>|z|)     
----------------------------------------------------------------------------------------
price            	-0.0274260820 	0.0023047000 	-11.9000657042 	0.0000000000 ***  
catch            	1.3298624416 	0.1725241388 	7.7082688286 	0.0000000000 **   
sd.price         	-0.0104976473 	0.0020694683 	-5.0726300621 	0.0000033594 **   
sd.catch         	1.5929740711 	0.3686123245 	4.3215431637 	0.0000903761 **   
----------------------------------------------------------------------------------------
Significance:  *** 0    ** 0.001    * 0.01    . 0.05

Log-Likelihood= -1300.780
```

## Installation
Install using pip:  
`pip install mixedlogit`  
Alternatively, you can download source code and import `mixedlogit.ChoiceModel`

To enable GPU processing you must install the library CuPy ([see installation instructions](https://docs.cupy.dev/en/stable/install.html)).  When mixedlogit detects that CuPy is installed, it switches to GPU processing.

## Notes:
The current version allows estimation of:
- Mixed logit models with normal and log-normal distributions.
- Mixed logit models with panel data
- Multinomial Logit Models: Models with individual specific variables
- Conditional Logit Models: Models with alternative specific variables
- Models with both, individual and alternative specific variables

## Citing `mixedlogit`

To cite this package:

**BibTeX**

```
@software{mixedlogit,
author = {Arteaga, Cristian and Bhat, Prithvi and Park, JeeWoong and Paz, Alexander},
title = {mixedlogit: A Python package for GPU accelerated estimation of mixed, multinomial and conditional logit models},
url = {https://github.com/arteagac/mixedlogit},
version = {0.0.1},
year = {2020},
}
```

