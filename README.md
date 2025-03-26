# Numerical Methods & Optimzation

<p align="center">
  <img src="./Optimzation/assets/batched_sgd_noise5.0.gif" alt="Batched SGD" width="45%">
  <img src="./Optimzation/assets/gds2.gif" alt="Various SGDs" width="45%">
</p>

<p align="center">
(a) The impact of batch on the optimization process; (b) The performance of different gradient descent variants on the same task.
</p>

## # Numberical Methods 
- Solution of Equations   
    - Solution of Linear Equations System
        - Principle Component Method
        - Jacobi Iteration ALgorithm
        - Gauss-Seidel Iteration Algorithm  
    - Solution of Nolinear Equations
        - Bisection Method
        - Newton's Method
        - Secant Method   
- Interpolation
    - 1D Interpolation Methods. 
        - Lagrange interpolation
        - Newtonian interpolation
        - cubic spline interpolation     
        - Aitken interpolation  
        - Hermit interpolation  
    - 2D Interpolation Methods
        - Nearest Neighbor Interpolation
        - Bilinear Interpolation
        - Bicubic Interpolation
- Approximation
    - approximation of functions 
- Integration & Difference Equations
    - Numerical Intergration
        - Rectangular Method
        - Trapezoidal Rule
        - Simpson's Rule
        - Gaussian Quadrature
        - Monte Carlo
    - Numerical Ode
        - Eular Method
        - Improved Euler Method
        - Runge Kutta from 2~10


## # Optimisation Algorithms

- Gradient Descent
    - Batch
        - SGD
        - Batched
        - Mini-Batched
    - Variants
        - SGD
        - Momentum
        - Nesterov Accelerated Gradient (NAG)
        - AdaGrad
        - RMSProp
        - AdaDelta
        - Adam (Adaptive Moment Estimation) 
- Heuristic Methods
    - Traveling Salesman Problem
        - Genetic Algorithm 
        - Particle Swarm Optimization
        - Ant Colony Optimization
    - Constrained optimization problem (TODO)
        - Genetic Algorithm 
        - Particle Swarm Optimization
        - Ant Colony Optimization
        - Simulated Annealing 
    - MNIST
        - Multi-Layer Perceptron (see this [notebook](https://github.com/HugoPhi/jaxdls/blob/main/mlp_mnist.ipynb))
     
## # Some Interesting Animations 

### @ Batched SGDs on Linear Regression

<p align="center">
  <img src="./Optimzation/assets/batched_sgd_noise0.5.gif" alt="TSP" width="30%">
  <img src="./Optimzation/assets/batched_sgd_noise5.0.gif" alt="TSP" width="30%">
  <img src="./Optimzation/assets/batched_sgd_noise10.gif" alt="TSP" width="30%">
</p>

<p align="center">
SGD with <code>batch_size=1, 10, 100</code>, on Linear Regression Task with (a) <code>noise=0.5</code>; (b) <code>noise=5</code>; (c) <code>noise=50</code>.
</p>

### @ GD Varients on Different Regression Functions.


<p align="center">
  <img src="./Optimzation/assets/gds1.gif" alt="TSP" width="45%">
  <img src="./Optimzation/assets/gds2.gif" alt="TSP" width="45%">
</p>

<p align="center">
SGD, Momentum, NAG, AdaGrad, RMSProp, AdaDelta, Adam on Regression Fuction: (a) $f(x) = \sin(wx)+b$; (b) $f(x) = w\sin(wx)+b$. 
</p>


### @ Traveling Salesman Problem by Heuristic Methods

> 💡 Notice:  
> - ACO: Basic.    
> - PSO: 2-opt.   
> - GA : with 2-opt, elite, tournament, OX crossover.    

<p align="center">
  <img src="./Optimzation/assets/tsp_10.gif" alt="TSP" width="90%">
</p>

<p align="center">
TSP with 10 cities. (a) ACO; (b) PSO; (c) GA.
</p>

<p align="center">
  <img src="./Optimzation/assets/tsp_20.gif" alt="TSP" width="90%">
</p>

<p align="center">
TSP with 20 cities. (a) ACO; (b) PSO; (c) GA.
</p>


<p align="center">
  <img src="./Optimzation/assets/tsp_50.gif" alt="TSP" width="90%">
</p>

<p align="center">
TSP with 50 cities. (a) ACO; (b) PSO; (c) GA.
</p>

<p align="center">
  <img src="./Optimzation/assets/tsp_100.gif" alt="TSP" width="90%">
</p>

<p align="center">
TSP with 100 cities. (a) ACO; (b) PSO; (c) GA.
</p>

<p align="center">
  <img src="./Optimzation/assets/tsp_200.gif" alt="TSP" width="90%">
</p>

<p align="center">
TSP with 200 cities. (a) ACO; (b) PSO; (c) GA.
</p>