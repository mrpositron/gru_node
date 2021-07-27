# GRU - NeuralODE 

Modeling the language dynamics using Neural Ordinary Differential Equations.  Main objective is to obtain meaningful results from the interpolation over the hidden dynamics learned by the Neural ODE.

## Introduction

Hidden states in neural networks, specifically in residual networks, recurrent neural networks, and normalizing flows, can be seen as an Euler discretization of a continuous transformation, 
                          
ht+1= ht+fht, t  t∈0…T, htRD

When we increase the number of layers and take smaller step, the equation above can be seen as an approximation to the solution of the following ordinary differential equation:
dh(t)dt=f(ht,t,θ)
 
Chen et al in their seminal NeurIPS paper [1] showed that we can use current techniques in ODE solvers and apply them in the Neural Networks. Specifically, we can use Neural Network to find f(ht, t, θ), and numerical methods to find h(t). Chen et al showed that their proposed ODE Network achieves 0.41% test error on classification dataset MNIST [2]. Furthermore, they showed that ODE Network outperforms RNN as time-series model.[1][3]

Being inspired by the work of Chen et al [1][3] we propose to use Neural ODE as the way to make Seq2Seq models interpretable. Specifically, we propose to model the decoder dynamics with a Neural ODE. 

There was no prior work to use Neural ODEs as a technique to make sequence to sequence architecture interpretable. Thus, it is a promising work in that direction.


## Experiments



## Conclusion



## References
