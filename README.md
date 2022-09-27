# NeuralNetwork
## Personal Project


  ### This project is a **C++** implementation of **Stochastic Gradient Descent** for a **Neural Network**
  
  As a general idea, this application takes a vector and tries to recognize what it represents,
generating another vector as output (as an answer) using a network of neurons and weights.

  There must be a connection between the input vectors the output vectors
( a function   **f : A -> B**   where **A** is the set of input vectors and **B** is the set of output vectors.

  Then the algorithm calculates a function **g : A -> B**, which approximates the function **f : A -> B**)
	
   This is done in 2 steps:
	 
- step 1.  - Training step:

	- We give the algorithm a finite number of pairs of vectors (input vector and output vector)

	- Based on them, the algorithm adjusts its parameters( weights and biases) to approximate the function **f**

	 - After training the algotithm "becomes" the  function **g : A -> B** that  approximate  function **f : A -> B**

	 - We can say that the algorithm "learns" what the function f does based on samples.
			
- step 2.  - Recognition step:

	 -We give the algorithm only input vectors and the algoritm calculate what the output vector should be
 
In this application, I use the **MINIST** database as function **f : A -> B**. This database is a collection
 of 70,000 handwritten digits.
 
   More details about the **MNIST** database can be found here:https://en.wikipedia.org/wiki/MNIST_database
   The **MNIST** database can be downloaded from here: http://yann.lecun.com/exdb/mnist/
 
   To import MNIST database I use the **library mnist_reader_less.hpp**. This import is done by the function
 **read_dataset()**, belonging to this library. This library can be downloaded from here: https://github.com/wichtounet/mnist/blob/master/README.rst . It was created by **Baptiste Wicht**: https://github.com/wichtounet
 
 As a source of documentation I used **Neural Networks and Deep Learning** - Michael Nielsen

 
 Algorithm structure:
 - mnist_reader_less.hpp: Provides the tools for importing the MNIST database
 - NeuralNetwork.h: NeuralNetwork class declaration 
 - NeuralNetwork.cpp: NeuralNetwork class implementation
           (NeuralNetwork class is the algorithm itself)
 - Source.cpp -> It contains the functions:
	- getTrainingMinibatch              
	- getPictureFromTrainingMNISTset     
	- getPictureFromTestMNISTset   
	These first three functions provide the connection between the Neural Network and the MNIST database
	- VisualizationOfNMIST: for a visual test of the algorithm
	- main() function
	
- As an activation function, I used the **Sigmoid Function**
- As a cost function I used the **Quadratic Function**

This is a **basic implementation** of the **Stochastic Gradient Descent** algorithm. Other features can be added to improve network performance.
