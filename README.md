# The problem in a few sentences

Stiff ODE systems (ODE systems that have solution components changing at multiple time scales) are notoriously expensive to solve when coupled with larger mesh-based simulation; for applications like combustion simulation, where the system of ODEs is coupled with a transport equation solution on a discretized mesh, the ODE solve can represent > 50% of the total compute cost. Surrogate models that can rapidly solve stiff systems accurately are hence in increasing demand in the field of modeling and simulation. Further, most traditional ML methods require uniform sampling in time, but the solution of stiff ODEs is non-uniformly sampled in time, requiring them to be handled differently.

# Who does this work benefit?

Anyone looking to explore using Continuous Time Echo State Networks (CTESNs) to create surrogate models of time series problems sampled irregularly in time, not limited to stiff ODEs.

# Requirements

To install the required libraries the user may run: ``` pip install -r requirements.txt ```

# Running the code

To run the linear projection code, run the file: ``` linear_projection_LPCTESN.py ```

To run the non-linear projection code, run the file: ``` nonlinear_method_NLPCTESN.py ```

# Citation

If you found this useful, please consider citing the paper:

```
Bhatnagar, S. Investigating the Surrogate Modeling Capabilities of Continuous Time Echo State Networks. Math. Comput. Appl. 2024, 29, 9. https://doi.org/10.3390/mca29010009
```

# So why is this interesting?
Continuous Time Echo State Networks (CTESNs) are a surrogate modeling tool with a particular application for modeling and parametrizing a family of stiff Ordinary Differential Equations (ODEs). Their unique methodology allows the fitting of models to irregularly sampled data with sharp transient changes, characteristic of solutions of stiff ODE problems. 

Traditional machine learning methods usually struggle with fitting solutions of stiff ODEs due to several reasons:
1. The solutions of stiff ODEs often exhibit very sharp transients, leading to non-smooth solution trajectories. The theory around most ML architectures revolves around approximating smooth functions, limiting the applicability of the methods.
2. Solution data from stiff ODEs are often sampled irregularly in time, but most ML architectures are designed to deal with regularly sampled data.

CTESNs are able to deal effectively with both these issues. They are a variant of echo-state networks, known for their ability to model chaotic problems. This property also allows them to capture sharp transients well. In addition, being continuous-time, they are able to deal with irregularly sampled data without having to resort to methods like interpolation.

This repository contains code that can be used to create surrogate models using Continuous Time Echo State Networks (CTESNs). The scripts attached reproduce some results in the paper "Investigating the Surrogate Modeling Capabilities of Continuous Time Echo State Networks" published in Mathematical and Computational Applications, 2024. The attached data generation script will produce data to create surrogates for Robertson's equation (a benchmark for stiff ODE solvers), parametrizing the reaction rate of the problem. The work explores using linear (LPCTESN) and non-linear (NLPCTESN) projection methods to create the surrogates, exploring the differences that arise. 






