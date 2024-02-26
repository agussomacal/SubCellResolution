

This repository has the python implementation of papers:
1. **Nonlinear approximation spaces for inverse problems**: [arXiv-preprint](https://arxiv.org/abs/2209.09314) | [Analysis and ApplicationsVol. 21, No. 01, pp. 217-253 (2023)](https://www.worldscientific.com/doi/10.1142/S0219530522400140)
<br>
<sub>
This paper is concerned with the ubiquitous inverse problem of recovering an
unknown function $u$ from finitely many measurements 
possibly affected by noise. In recent years, inversion methods based
on _linear_ approximation spaces were introduced with certified recovery bounds. 
It is however known that linear spaces become ineffective 
for approximating simple and relevant families of functions, such as piecewise smooth 
functions that typically occur in hyperbolic PDEs (shocks) or images (edges).
For such families, _nonlinear_ spaces are known to significantly improve
the approximation performance. The first contribution of this paper is to provide with certified recovery 
bounds for inversion procedures based on nonlinear approximation spaces. The second contribution 
is the application of this framework
to the recovery of general bidimensional shapes from cell-average data. 
We also discuss how the application of our results to $n$-term approximation relates to classical results in compressed sensing.
</sub> <br><br>

2. **High order recovery of geometric interfaces from cell-average data**: [arXiv-preprint](http://arxiv.org/abs/2402.00946)
<br><sub> 
We consider the problem of recovering multivariate characteristic functions u := χΩ from cell-average
data on a coarse grid, motivated in particular by the accurate treatment of interfaces in finite volume
schemes. While linear recovery methods are known to perform poorly, nonlinear strategies based on local
reconstructions of the jump interface Γ := ∂Ω by geometrically simpler interfaces may offer significant
improvements. We study two main families of local reconstruction schemes the first one based on nonlinear
least-squares fitting, the second one based on the explicit computation of a polynomial-shaped curve fitting
the data, which yields simpler numerical computations and high order geometric fitting. For each of them,
we derive a general theoretical framework which allows us to control the recovery error by the error of
best approximation up to a fixed multiplicative constant. Numerical tests in 2d illustrate the expected
approximation order of these strategies. Several extensions are discussed, in particular the treatment of
piecewise smooth interfaces with corners. </sub>


## Running experiments

The scripts that produce the different figures of the papers are the following $5$ ordered by order of 
appearance in paper 2. The plots found in paper 1 are covered and enriched by the second and third script.

```
/src/experiments/PaperPlots/paper_orientation_plot.py
/src/experiments/PaperPlots/paper_convergence_plot.py
/src/experiments/PaperPlots/paper_smooth_domains_plot.py
/src/experiments/PaperPlots/paper_corners_plot.py
/src/experiments/PaperPlots/paper_scheme_plot.py
```

## Setup for developers
We recommend first to work in a virtual environment which can be created using 
previously installed python packages venv or virtualenv through
```
python3.8 -m venv venv
```
or
```
virtualenv -p python3.8 test
```

Then activate virtual environment
```
. .venv/bin/activate
```
Install required packages usually through:
```
pip install -r requirements.txt 
```
However, if this doesn't work for you, try to install them one by one in the order specified by the requirements.txt file.



