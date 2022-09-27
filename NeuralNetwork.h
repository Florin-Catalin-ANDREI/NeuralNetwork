#pragma once
#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H
namespace neural_network {
	struct Neuron
	{
		float activation;
		float bias;
		float zet;
		float deviation;
		float gradient;
	};
	
	struct Weight
	{
		float weight;
		float gradient;
	};
	struct NeuronStartPoint
	{
		Neuron* neuronStartPoint;
	};
	class NeuralNetwork
	{
	public:
		NeuralNetwork(std::vector<int> layersLenghts, float learningIndex, int minibatchLenght);
		~NeuralNetwork();
		void LearningMinibatchLoop(std::vector<std::vector<float>>& minibatchPictures, std::vector<int>& minibatchLabels);
		int getTestResult(std::vector<float>& picture);
	private:
		void InitializeTheNeuralNetwork();
		void InitializeTheDeviations();
		void FeedForward();
		void BackPropagation();
		void GradientDescent();
		float ActivationFunction(float& input);  //Signoid function
		float ActivationFunctionDerived(float& input);
		float CostFunction(float& activationInput, float& desiredValue);   // Quadratic cost function
		float CostFunctionDerived(float& activationInput, float& desiredValue);
		float GetRandomWeight();
		int m_totalNumberOfLayers = 0;
		int* m_layersLenghts;
		int m_totalNumberOfNeurons = 0;
		int m_totalNumberOfWeights = 0;
		Weight* m_weightsMatrices;
		Neuron* m_neuronsVectors;
		NeuronStartPoint* m_neuronsVectorStartPoints;
		float* m_desiredOutput;
		int m_lenghtOfInputLayer;
		int m_lenghtOfOutputLayer;
		float m_learningIndex = 2;
		int m_miniBatchLenght = 10;
	};
}//end of namespace neutral_network
#endif NEURAL_NETWORK_H