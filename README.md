# CTESN_surrogates

This repository contains code to create surrogate models using Continuous Time Echo State Networks (CTESNs), which was used to produce results in the paper "Investigating the Surrogate Modeling Capabilities of Continuous Time Echo State Networks" published in Mathematical and Computational Applications, 2024. The attached data generation script will produce data to create surrogates for Robertson's equation, parametrizing the reaction rate of the problem. The work explores using linear (LPCTESN) and non-linear (NLPCTESN) projection methods to create the surrogates, exploring the differences that arise. 

To install the required libraries the user may run: ``` pip install -r requirements.txt ```

To run the linear projection code, run the file: ``` linear_projection_LPCTESN.py ```

To run the non-linear projection code, run the file: ``` nonlinear_method_NLPCTESN.py ```

If you use the code from this repository, please cite the paper:

```
Bhatnagar, S. Investigating the Surrogate Modeling Capabilities of Continuous Time Echo State Networks. Math. Comput. Appl. 2024, 29, 9. https://doi.org/10.3390/mca29010009
```
