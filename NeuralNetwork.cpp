#include<cstdlib>
#include <time.h>
#include <iostream>
#include <vector>
#include "NeuralNetwork.h"
using namespace std;
using namespace neural_network;

	NeuralNetwork::NeuralNetwork(std::vector<int> layersLenghts, float learningIndex, int minibatchLenght)
	{
		// ---------------------------------------------------
		// Set the number of layers of network
		// ---------------------------------------------------
		m_totalNumberOfLayers = layersLenghts.size();
		// ---------------------------------------------------
		// Set the number of neurons for each layer
		// ---------------------------------------------------
		int* temporaryLayersLenght = new int[m_totalNumberOfLayers];
		m_layersLenghts = temporaryLayersLenght;
		for (int i = 0; i < m_totalNumberOfLayers;i++)
		{
			if (layersLenghts[i] < 1)
			{
				cout << endl << endl << "All values for layer lengths must be greater than 0 " << (layersLenghts.size()) << endl << endl << endl;
				exit(0);
			}
			m_layersLenghts[i] = layersLenghts[i];
		}
		// ---------------------------------------------------
		// Calculate the total number of neurons and weights
		// ---------------------------------------------------
		m_totalNumberOfWeights = 0;
		m_totalNumberOfNeurons = 0;
		for (int i = 0;i < m_totalNumberOfLayers - 1;i++)
		{
			m_totalNumberOfWeights += m_layersLenghts[i] * m_layersLenghts[i + 1];
			m_totalNumberOfNeurons += m_layersLenghts[i];
		}
		m_totalNumberOfNeurons += m_layersLenghts[m_totalNumberOfLayers - 1];
		// ---------------------------------------------------
		// Set the vector for weights
		// ---------------------------------------------------
		Weight* temporaryWeightsMatrices = new Weight[m_totalNumberOfWeights];
		m_weightsMatrices = temporaryWeightsMatrices;
		// ---------------------------------------------------
		// Set the vector for neurons
		// ---------------------------------------------------
		Neuron* temporaryNeuronsVectors = new Neuron[m_totalNumberOfNeurons];
		m_neuronsVectors = temporaryNeuronsVectors;
		// ---------------------------------------------------
		// Set the vector for "start points" in neuron vector
		// Its elements are pointers to first neuron on each layer
		// The last elemet is an "end of vector" for neuron vector
		// ---------------------------------------------------
		NeuronStartPoint* temporaryNeuronsVectorStartPoints = new NeuronStartPoint[m_totalNumberOfLayers + 1];
		m_neuronsVectorStartPoints = temporaryNeuronsVectorStartPoints;
		// ---------------------------------------------------
		// Set the number of neurons on input layer
		// ---------------------------------------------------
		m_lenghtOfInputLayer = m_layersLenghts[0];
		// ---------------------------------------------------
		// Set the number of neurons on output layer
		// ---------------------------------------------------
		m_lenghtOfOutputLayer = m_layersLenghts[m_totalNumberOfLayers - 1];
		// ---------------------------------------------------
		// Set the vector for expected output
		// ---------------------------------------------------
		float* temporaryDesiredOutput = new float[m_lenghtOfOutputLayer];
		m_desiredOutput = temporaryDesiredOutput;
		// ---------------------------------------------------
		// Set the learning index
		// ---------------------------------------------------
		m_learningIndex = learningIndex;
		// ---------------------------------------------------
		// Set the minibatch lenght
		// ---------------------------------------------------
		m_miniBatchLenght = minibatchLenght;
		// ---------------------------------------------------
		// Initializes all network components
		// ---------------------------------------------------
		InitializeTheNeuralNetwork();
	}

	NeuralNetwork::~NeuralNetwork()
	{
		delete m_layersLenghts;
		delete m_weightsMatrices;
		delete m_neuronsVectors;
		delete m_neuronsVectorStartPoints;
		delete m_desiredOutput;
	}

	void NeuralNetwork::LearningMinibatchLoop(vector<vector<float>>& minibatchPictures, vector<int>& minibatchLabels)
	{
		int j;
		Neuron* neuronsVectors_EndPoint = m_neuronsVectors + m_layersLenghts[0];
		float* desireOutput_endPoint = m_desiredOutput + m_layersLenghts[m_totalNumberOfLayers - 1];
		InitializeTheDeviations();
		for (int i = 0;i < m_miniBatchLenght;i++)
		{
			// ---------------------------------------------------
			// Initialize the input neurons
			// ---------------------------------------------------
			j = 0;
			for (Neuron* iterator = m_neuronsVectors;iterator < neuronsVectors_EndPoint;iterator++)
			{
				iterator->activation = minibatchPictures[i][j++];
			}
			// ---------------------------------------------------
			// Initiatialize the vector for desired output
			// ---------------------------------------------------
			for (float* iterator = m_desiredOutput;iterator < desireOutput_endPoint; iterator++)
			{
				*iterator = 0;
			}
			m_desiredOutput[minibatchLabels[i]] = 1;
			// ---------------------------------------------------
			// Forward propagation => get an output from the network based on a picture
			// ---------------------------------------------------
			FeedForward();
			// ---------------------------------------------------
			// Back Propagation:
			// It calculates the difference between the output of the network and the desired output
			// Based on this, it calculates the adjustments for each weight and for each bias (deviations)
			// ---------------------------------------------------
			BackPropagation();
		}
		// ---------------------------------------------------
		//Adjust each weightand each bias based on the deviations calculated above
		// ---------------------------------------------------
		GradientDescent();
	}

	int NeuralNetwork::getTestResult(vector<float>& picture)
	{
		// ---------------------------------------------------
		// Initialize the activations for receptor neurons (The first 784 positions of "m_neuronsVectors")
		// ---------------------------------------------------
		int i = 0;
		Neuron* iterator;
		Neuron* neuronsVectors_EndPoint = m_neuronsVectors + m_layersLenghts[0];
		for (iterator = m_neuronsVectors;iterator < neuronsVectors_EndPoint;iterator++)
		{
			iterator->activation = picture[i];
			i++;
		}
		// ---------------------------------------------------
		// Calculates an "answer" that will be found on the activations of the LAST 10 neurons in "m_neuronsVectors"
		// ---------------------------------------------------
		FeedForward();
		// ---------------------------------------------------
		//From the activations of the last 10 neurons of "m_neuronsVectors" calculate the final "answer"
		//    as the position with the max value
		// ---------------------------------------------------
		iterator = m_neuronsVectorStartPoints[m_totalNumberOfLayers - 1].neuronStartPoint;
		float maxValue = iterator->activation;
		int result = 0;
		for (i = 1;i < m_layersLenghts[m_totalNumberOfLayers - 1];i++)
		{
			iterator++;
			if (iterator->activation >= maxValue)
			{
				maxValue = iterator->activation;
				result = i;
			}
		}
		return result;
	}

	void NeuralNetwork::InitializeTheNeuralNetwork()
	{
		// ---------------------------------------------------
		// Initialize the weights
		// ---------------------------------------------------
		srand((int)time(0));
		Weight* endOfWeightsVector = m_weightsMatrices + m_totalNumberOfWeights;
		for (Weight* weight_iterator = m_weightsMatrices; weight_iterator != endOfWeightsVector;weight_iterator++)
		{
			weight_iterator->weight = GetRandomWeight();
			weight_iterator->gradient = 0;
		}
		// ---------------------------------------------------
		// Initializa the neurons
		// ---------------------------------------------------
		Neuron* endOfNeuronsVector = m_neuronsVectors + m_totalNumberOfNeurons;
		for (Neuron* neuron_iterator = m_neuronsVectors + m_layersLenghts[0]; neuron_iterator != endOfNeuronsVector;neuron_iterator++)
		{
			neuron_iterator->bias = 1;
			neuron_iterator->activation = 0;
			neuron_iterator->deviation = 0;
			neuron_iterator->gradient = 0;
			neuron_iterator->zet = 0;
		}
		// ---------------------------------------------------
		// Initializa the starting points for each layer
		// ---------------------------------------------------
		m_neuronsVectorStartPoints[0].neuronStartPoint = m_neuronsVectors;
		for (int i = 1; i < m_totalNumberOfLayers + 1;i++)
		{
			m_neuronsVectorStartPoints[i].neuronStartPoint = m_neuronsVectorStartPoints[i - 1].neuronStartPoint + m_layersLenghts[i - 1];
		}
	}
	void NeuralNetwork::InitializeTheDeviations()
	{
		// ---------------------------------------------------
		// Inititialize the deviations for weights
		// ---------------------------------------------------
		Weight* endOfWeightsVector = m_weightsMatrices + m_totalNumberOfWeights;
		for (Weight* weight_iterator = m_weightsMatrices; weight_iterator != endOfWeightsVector;weight_iterator++)
		{
			weight_iterator->gradient = 0;
		}
		// ---------------------------------------------------
		// Inititialize the deviations for biases
		// ---------------------------------------------------
		Neuron* endOfNeuronsVector = m_neuronsVectors + m_totalNumberOfNeurons;
		for (Neuron* neuron_iterator = m_neuronsVectors + m_layersLenghts[0]; neuron_iterator != endOfNeuronsVector;neuron_iterator++)
		{
			neuron_iterator->gradient = 0;
		}
	}

	// ---------------------------------------------------
	// Forward propagation => get an output from the network based on a picture
	// ---------------------------------------------------
	void NeuralNetwork::FeedForward()
	{
		Neuron* neuronFromCurentLayer_iterator;
		Neuron* neuronFromPreviousLayer_iterator;
		Weight* weightFromCurentLayer_iterator = m_weightsMatrices;
		for (int i = 1;i < m_totalNumberOfLayers;i++)
		{
			for (neuronFromCurentLayer_iterator = m_neuronsVectorStartPoints[i].neuronStartPoint;neuronFromCurentLayer_iterator < m_neuronsVectorStartPoints[i + 1].neuronStartPoint;neuronFromCurentLayer_iterator++)
			{
				neuronFromCurentLayer_iterator->zet = 0;
				for (neuronFromPreviousLayer_iterator = m_neuronsVectorStartPoints[i - 1].neuronStartPoint;neuronFromPreviousLayer_iterator < m_neuronsVectorStartPoints[i].neuronStartPoint;neuronFromPreviousLayer_iterator++)
				{
					neuronFromCurentLayer_iterator->zet += weightFromCurentLayer_iterator->weight * neuronFromPreviousLayer_iterator->activation;
					weightFromCurentLayer_iterator++;
				}
				neuronFromCurentLayer_iterator->zet += neuronFromCurentLayer_iterator->bias;
				neuronFromCurentLayer_iterator->activation = ActivationFunction(neuronFromCurentLayer_iterator->zet);
			}
		}
	}

	// ---------------------------------------------------
	// Back Propagation:
	// It calculates the difference between the output of the network and the desired output
	// Based on this, it calculates the adjustments for each weight and for each bias (deviations)
	// ---------------------------------------------------
	void NeuralNetwork::BackPropagation()
	{
		Neuron* neuronFromCURENTLayer_iterator = m_neuronsVectors + m_totalNumberOfNeurons;
		Neuron* neuronFromPREVIOUSLayer_iterator;
		Neuron* neuronFromNEXTLayer_iterator;
		float  deviationValue;
		Weight* weightFromCurentLayer_iterator = m_weightsMatrices + m_totalNumberOfWeights;
		Weight* weightFromNextLayer_iterator = m_weightsMatrices + m_totalNumberOfWeights;
		int i = 0;
		int j = 0;
		int k = 0;
		for (j = m_layersLenghts[m_totalNumberOfLayers - 1] - 1;j > -1;j--)
		{
			neuronFromCURENTLayer_iterator--;
			neuronFromCURENTLayer_iterator->deviation = CostFunctionDerived(neuronFromCURENTLayer_iterator->activation, m_desiredOutput[j]) * ActivationFunctionDerived(neuronFromCURENTLayer_iterator->zet);
			neuronFromCURENTLayer_iterator->gradient += neuronFromCURENTLayer_iterator->deviation;
			for (neuronFromPREVIOUSLayer_iterator = m_neuronsVectorStartPoints[m_totalNumberOfLayers - 1].neuronStartPoint - 1;neuronFromPREVIOUSLayer_iterator >= m_neuronsVectorStartPoints[m_totalNumberOfLayers - 2].neuronStartPoint;neuronFromPREVIOUSLayer_iterator--)
			{
				weightFromCurentLayer_iterator--;
				weightFromCurentLayer_iterator->gradient += neuronFromCURENTLayer_iterator->deviation * neuronFromPREVIOUSLayer_iterator->activation;
			}
		}
		for (i = m_totalNumberOfLayers - 2;i > 0;i--)
		{
			
			for (j = 0;j < m_layersLenghts[i];j++)
			{
				neuronFromCURENTLayer_iterator--;
				neuronFromNEXTLayer_iterator = m_neuronsVectorStartPoints[i + 2].neuronStartPoint - 1;
				deviationValue = 0;
				weightFromNextLayer_iterator--;
				for (k = 0;k < m_layersLenghts[i + 1];k++)
				{
					deviationValue += (weightFromNextLayer_iterator - k * m_layersLenghts[i])->weight * neuronFromNEXTLayer_iterator->deviation;
					neuronFromNEXTLayer_iterator--;
				}
				deviationValue *= ActivationFunctionDerived(neuronFromCURENTLayer_iterator->zet);
				neuronFromCURENTLayer_iterator->deviation = deviationValue;
				neuronFromCURENTLayer_iterator->gradient += deviationValue;
				for (neuronFromPREVIOUSLayer_iterator = m_neuronsVectorStartPoints[i].neuronStartPoint - 1;neuronFromPREVIOUSLayer_iterator >= m_neuronsVectorStartPoints[i - 1].neuronStartPoint;neuronFromPREVIOUSLayer_iterator--)
				{
					weightFromCurentLayer_iterator--;
					weightFromCurentLayer_iterator->gradient += deviationValue * neuronFromPREVIOUSLayer_iterator->activation;
				}
			}
			weightFromNextLayer_iterator -= m_layersLenghts[i + 1] * (m_layersLenghts[i] - 1);
		}
	}

	// ---------------------------------------------------
	// Applying the gradient
	// Adjust each weightand each bias based on the deviations calculated above
	// ---------------------------------------------------
	void NeuralNetwork::GradientDescent()
	{
		for (Weight* weight_iterator = m_weightsMatrices; weight_iterator < m_weightsMatrices + m_totalNumberOfWeights;weight_iterator++)
		{
			weight_iterator->weight -= (m_learningIndex / m_miniBatchLenght) * weight_iterator->gradient;
		}
		for (Neuron* bias_iterator = m_neuronsVectors;bias_iterator < m_neuronsVectors + m_totalNumberOfNeurons;bias_iterator++)
		{
			bias_iterator->bias -= (m_learningIndex / m_miniBatchLenght) * bias_iterator->gradient;
		}
	}

	// ---------------------------------------------------
	// I used signoid funtion as activation function
	// ---------------------------------------------------
	float NeuralNetwork::ActivationFunction(float& input)
	{
		return 1 / (1 + exp(-input));
	}

	// ---------------------------------------------------
	// Signoid function derived
	// ---------------------------------------------------
	float NeuralNetwork::ActivationFunctionDerived(float& input)
	{
		float signoid = 1 / (1 + exp(-input));
		return signoid * (1 - signoid);
	}

	// ---------------------------------------------------
	// I used quadratic cost function
	// ---------------------------------------------------
	float NeuralNetwork::CostFunction(float& activationInput, float& desiredValue)
	{
		float value = activationInput - desiredValue;
		return (value * value) / 2;
	}

	// ---------------------------------------------------
	// Quadratic cost function derived
	// ---------------------------------------------------
	float NeuralNetwork::CostFunctionDerived(float& activationInput, float& desiredValue)
	{
		return activationInput - desiredValue;
	}

	// ---------------------------------------------------
	// Generate a random value for weights initial values
	// ---------------------------------------------------
	float NeuralNetwork::GetRandomWeight()
	{
		int randomNumber;
		do
		{
			randomNumber = rand() % 100 - 50;
		} while (randomNumber == 0);
		return (float)1 / (2 * randomNumber);
	}




