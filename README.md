# GRU - NeuralODE 

Modeling the language dynamics using Neural Ordinary Differential Equations.  Main objective is to obtain meaningful results from the interpolation over the hidden dynamics learned by the Neural ODE.

## Introduction

Hidden states in neural networks, specifically in residual networks, recurrent neural networks can be seen as an Euler discretization of a continuous transformation [1] , 

<p align="center">                    
 <img width="246" alt="image" src="https://user-images.githubusercontent.com/42044624/127249551-80320c7b-3e5f-454b-a71d-7f0e87286f5f.png">
</p>

When we increase the number of layers and take smaller step, the equation above can be seen as an approximation to the solution of the following ordinary differential equation:

<p align="center">  
<img align="center" width="246" alt="image" src="https://user-images.githubusercontent.com/42044624/127249592-3a959a91-5378-44c7-aaa1-866e771fd6ec.png">
</p>
 
Chen et al in their seminal NeurIPS paper [1] showed that we can use current techniques in ODE solvers and apply them in the Neural Networks. Specifically, we can use Neural Network to find "${ f(ht, t, θ) }$", and numerical methods to find h(t). Chen et al showed that their proposed ODE Network achieves 0.41% test error on classification dataset MNIST [2]. Furthermore, they showed that ODE Network outperforms RNN as time-series model.[1][3]

Being inspired by the work of Chen et al [1][3] we propose to use Neural ODE as the way to make Seq2Seq models interpretable. Specifically, we propose to model the decoder dynamics with a Neural ODE. 

There was no prior work to use Neural ODEs as a technique to make sequence to sequence architecture interpretable. Thus, it is a promising work in that direction.


## Experiments



## Conclusion

Although, the idea was promising, it did not work well as it was intended. 

## References

[1] - Neural Ordinary Differential Equations. Ricky T. Q. Chen*, Yulia Rubanova*, Jesse Bettencourt*, David Duvenaud.   
[2] - MNIST handwritten digit database. LeCun, Yann and Cortes, Corinna and Burges, CJ.  
[3] - Latent ODEs for Irregularly-Sampled Time Series.  Yulia Rubanova, Ricky T. Q. Chen, David Duvenaud. 
