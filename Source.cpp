
#include <iostream>
#include "mnist/mnist_reader_less.hpp"
#include "NeuralNetwork.h"
using namespace std;
using namespace mnist;
using namespace neural_network;


//  As a general idea, this application takes a vector and tries to recognize what it represents,
//generating another vector as output (as an answer).
//  There must be a connection between the input vectors the output vectors
//( a function   f : A -> B   where A is the set of input vectors and B is the set of output vectors.
//  Tthen the algorithm calculates a function g : A -> B, which approximates the function f : A -> B)
//   This is done in 2 steps:
//  step 1.  - Training step:
//      We give the algorithm a finite number of pairs of vectors (input vector and output vector)
//      Based on them, the algorithm adjusts its parameters( weights and biases) to approximate the function f
//      After training the algotithm "becomes" the function g : A -> B that approximate function f : A -> B
//      We can say that the algorithm "learns" what the function f does based on samples.
//  step 2.  - Recognition step:
//      We give the algorithm only input vectors and the algoritm calculate what the output vector should be
// 
//   In this application, I use the MINIST database as function f : A -> B. This database is a collection
// of 70,000 handwritten digits.
//   More details about the MNIST database can be found here:https://en.wikipedia.org/wiki/MNIST_database
//   The MNIST database can be downloaded from here: http://yann.lecun.com/exdb/mnist/
// 
//   To import MNIST database I use the library mnist_reader_less.hpp. This import is done by the function
// read_dataset(), belonging to this library. This library can be downloaded from here:
// 	https://github.com/wichtounet/mnist/blob/master/README.rst . It was created by Baptiste Wicht:https://github.com/wichtounet
// 
// 
// Algorithm structure:
//     - mnist_reader_less.hpp -> Provides the tools for importing the MNIST database
//     - NeuralNetwork.h   -> NeuralNetwork class declaration 
//     - NeuralNetwork.cpp -> NeuralNetwork class implementation
//           (NeuralNetwork class is the algorithm itself)
//     - Source.cpp -> It contains the functions:
//          - getTrainingMinibatch              |
//          - getPictureFromTrainingMNISTset    | provide the connection between the Neural Network and the MNIST database
//          - getPictureFromTestMNISTset        |
//          - VisualizationOfNMIST  -> for a visual test of the algorithm
//          - main() function







// *******************************************************************************
// *      Next three functions are the connection with MNIST database            *
// *******************************************************************************

//--------------------------------------------------------------------------------------
//   **      Building a set of 10 pictures and their labels from training MNIST set
//--------------------------------------------------------------------------------------
void getTrainingMinibatch(MNIST_dataset<uint8_t, uint8_t>* dataFromMNIST, int index, int minibatchLenght, vector<vector<float>>& minibatchPictures, vector<int>& minibatchLabels)
{
	int j;
	minibatchPictures.resize(minibatchLenght);
	minibatchLabels.resize(minibatchLenght);
	for (int i = 0; i < minibatchLenght; i++)
	{
		minibatchPictures[i].resize(784);
		for (j = 0;j < 784;j++)
		{
			minibatchPictures[i][j] = (float)dataFromMNIST->training_images[index + i][j] / 255;
		}
		minibatchLabels[i] = (int)dataFromMNIST->training_labels[index + i];
	}
}

//--------------------------------------------------------------------------------------
//    **     Bringing a picture and its label from training MNIST set
//--------------------------------------------------------------------------------------
void getPictureFromTrainingMNISTset(MNIST_dataset<uint8_t, uint8_t>* dataFromMNIST, int index, vector<float>& picture, int* label)
{
	picture.resize(784);
	for (int i = 0;i < 784;i++)
	{
		picture[i] = (float)dataFromMNIST->training_images[index][i] / 255;
	}
	*label = (int)dataFromMNIST->training_labels[index];
}

//--------------------------------------------------------------------------------------
//    **     Bringing a picture and its label from test MNIST set
//--------------------------------------------------------------------------------------
void getPictureFromTestMNISTset(MNIST_dataset<uint8_t, uint8_t>* dataFromMNIST, int index, vector<float>& picture, int* label)
{
	picture.resize(784);
	for (int i = 0;i < 784;i++)
	{
		picture[i] = (float)dataFromMNIST->test_images[index][i] / 255;
	}
	*label = (int)dataFromMNIST->test_labels[index];
}


// *******************************************************************************
// *     This function generates a "graphical representation" of  a picture      *
// *                          from MNIST database                                *
// *         and the output of the algorithm after "reading" the picture         *
// *******************************************************************************

void VisualizationOfNMIST(vector<float>& picture, int label, int answer)
{
	std::cout << string(100, '\n');
	std::cout << "  T H E   M.N.I.S.T.   L A B E L   F O R   T H I S   P I C T U R E   I S   " << endl;
	std::cout << "                        T H E   N U M B E R   " << endl << endl << "                            <<  " << label << "  >>" << endl;
	for (int i = 0;i < 784;i++)
	{

		if (picture[i] == 0)
		{
			std::cout << "  ";
		}
		else
		{
			std::cout << "* ";
		}
		if (i % 28 == 0) std::cout << endl;
	}
	std::cout << endl << "The Neural Network recognizes this picture as a number:    " << answer << endl;
	std::cout << endl << "        ------------------------------------             -----    " << endl << endl;
}





// *******************************************************************************
// *                                                                             *
// *                   T H E    M A I N ()    F U N C T I O N                    *
// *                                                                             *
// *******************************************************************************
int main()
{
// **********************************************************
// *                Import MNIST data set                   *
// **********************************************************
	MNIST_dataset<uint8_t, uint8_t> MNISTdataset;
	MNISTdataset = read_dataset();

// **********************************************************
// *             Setup de the neural network                *
// **********************************************************
	float learningIndex = 2;    //    <-  it can be adjusted
	int minibatchLength = 10;   //    <-  it can be adjusted
	int numberOfEpochs = 10;    //    <-  it can be adjusted
	vector<int> neuralNerworkStructure = { 784, 50, 50, 10 };
	//vector<int> neuralNerworkStructure = { 784, 80, 10 };    // <-  alternativ structure
	//  ->  784 = the number of input neurons (input layer)
	//                               >>Do not change the number of input neurons. <<
	//                               >>It must be the same as the number of pixels in a MNIST pictures<<
	//  ->   50 = the number of neurons on first hiden layer
	//  ->   50 = the number of neurons on second hiden layer
	//  ->   10 = the number of output neurons (Do not change this value)
	//  you can change the number of neurons on hiden layers, or you can add or take out hiden layers
	//   !!!!!More neurons in the hidden layers will increase the training time exponentially!!!

	NeuralNetwork network(neuralNerworkStructure, learningIndex, minibatchLength);

// **********************************************************
// *                  Training variables                    *
// **********************************************************
	int loops = (int)60000 / minibatchLength;
	vector<vector<float>> minibatchPictures;
	vector<int> minibatchLabels;
	vector<float> picture;
	int label;
// **********************************************************
// *                   Other variables                      *
// **********************************************************
	int i = 0;
	int j = 0;
	int k = 0;
	int randomIndex = 0;
	int generalCounter = 0;
	int successCounter = 0;
	int successCounterTOTAL = 0;	

	std::cout << "The training of the network will take few minutes:" << endl << endl;
	for (i = 0;i < numberOfEpochs;i++)
	{
		std::cout << "Epoch number: " << i + 1 << ":" << endl;
		successCounter = 0;
		generalCounter = 0;
		for (j = 0;j < loops;j++)
		{	
			// **********************************************************
			// *                   Training block                       *
			// **********************************************************
			getTrainingMinibatch(&MNISTdataset, j * minibatchLength, minibatchLength, minibatchPictures, minibatchLabels);
			network.LearningMinibatchLoop(minibatchPictures, minibatchLabels);
			// **********************************************************
			// * The following "for" loop performs an intermediate test *
			// *         used to track the "learning" progress          *
			// *       (does not affect the "learning" process)         *
			// *         (it just shows the learning progress)          *
			// **********************************************************
			for (k = 0; k < minibatchLength;k++)
			{
				randomIndex = rand() % 60000;
				getPictureFromTrainingMNISTset(&MNISTdataset, randomIndex, picture, &label);
				if (network.getTestResult(picture) == label)
				{
					successCounter++;
				}
				generalCounter++;
			}		
		}
		std::cout << "I read  " << generalCounter << "  digits.  --  I recognized correctly  " << successCounter << "   -->   Success rate is: " << fixed << 100 * ((float)(successCounter) / (generalCounter)) << " %" << endl;
		successCounterTOTAL += successCounter;
	}

// **********************************************************
// *                 Testing Neural Network                 *
// *         It is different from the previous test         *
// *        because it uses the MNIST set for TESTING       *
// **********************************************************
	std::cout << string(8, '\n') << "Testing the Neural Network using TEST pictures from MNIST: " << endl;
	std::cout <<                    "---------------------------------------------------------- " << string(2, '\n');
	successCounter = 0;
	for (i = 0;i < 10000;i++)
	{
		getPictureFromTestMNISTset(&MNISTdataset, i, picture, &label);
		if (network.getTestResult(picture) == label)
		{
			successCounter++;
		}
	}
	std::cout << "I recognized correctly " << successCounter << "    from a total of:" << 10000 << " test images  -->  ";
	std::cout << "Success rate is: " << fixed << 100 * ((float)(successCounter) / (10000)) << " %" << endl;
	std::cout << string(3, '\n');


// **********************************************************
// *              Testing the Neural Network                *
// *                 It is a visual test                    *
// * Displays the image from MNIST and the network response *
// **********************************************************
	char a;
	std::cout << endl << "Enter a letter from the keyboard ... and press ENTER  for the last step" << endl;
	std::cin >> a;
	std::cout << string(100, '\n');
	std::cout << endl << "   - Now let's test the trained Neural Network! -" << endl << endl;
	std::cout << endl << "For a proper view, enlarge this window or make it full screen" << endl;
	std::cout << endl << "Enter a letter from the keyboard ... and press ENTER";
	std::cout << string(10, '\n');
	std::cin >> a;
	do
	{
		randomIndex = rand() % 10000;
		getPictureFromTrainingMNISTset(&MNISTdataset, randomIndex, picture, &label);
		VisualizationOfNMIST(picture, label, network.getTestResult(picture));
		std::cout << endl << "Do you want to make another try?";
		std::cout << "                   Y / N    ... and press ENTER" << endl;
		std::cin >> a;
	} while (a == 'y' || a == 'Y');
	return 0;
};