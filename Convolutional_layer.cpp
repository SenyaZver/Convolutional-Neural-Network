#include "Convolutional_layer.h"

double Convolutional_layer::getRandomNumber(double begin, double end) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(begin, end);

	return dis(gen);
}

std::vector<std::vector<double>> Convolutional_layer::createMatrix(size_t size)
{
	std::vector<std::vector<double>> result(size);
	for (size_t i = 0; i < size; i++) {
		result[i].resize(size);
	}
	return result;
}

double Convolutional_layer::cross(std::vector<std::vector<double>>& const first, std::vector<std::vector<double>>& const second)
{
	double result = 0;
	size_t size = first.size();

	for (size_t i = 0; i < size; i++)
	{
		for (size_t j = 0; j < size; j++)
		{
			result += first[i][j] * second[i][j];
		}
	}
	return result;
}

std::vector<std::vector<double>> Convolutional_layer::getSubMatrix(std::vector<std::vector<double>>& inputLayer, size_t verticalOffset, size_t horizontalOffset)
{
	size_t size = filter1.size();
	auto result = createMatrix(size);


	for (size_t i = 0; i < size; i++) {
		for (size_t j = 0; j < size; j++) {
			result[i][j] = inputLayer[i + verticalOffset][j + horizontalOffset];
		}
	}
	return result;
}

double Convolutional_layer::sigmoid(double const x)
{
	return 1 / (1 + exp(-x));
}

void Convolutional_layer::activate(std::vector<std::vector<double>>& const layer)
{
	size_t size = layer.size();
	for (size_t i = 0; i < size; i++) {
		for (size_t j = 0; j < size; j++) {
			layer[i][j] = sigmoid(layer[i][j]);
		}
	}
}

void Convolutional_layer::randomInit()
{
	for (size_t i = 0; i < filterSize; i++) {
		for (size_t j = 0; j < filterSize; j++) {
			filter1[i][j] = getRandomNumber(-1, 1);
			filter2[i][j] = getRandomNumber(-1, 1);
			filter3[i][j] = getRandomNumber(-1, 1);
			bias[i][j] = getRandomNumber(-1, 1);
		}
	}
}

std::vector<std::vector<double>> Convolutional_layer::cross_correlate(std::vector<std::vector<double>>& inputLayer, std::vector<std::vector<double>>& filter)
{
	size_t size = inputLayer1.size() - filter1.size() + 1;



	auto result = createMatrix(size);
	auto temp = createMatrix(size);

	size_t verticalOffset = 0;
	size_t horizontalOffset = 0;


	for (size_t i = 0; i < size; i++) {
		for (size_t j = 0; j < size; j++) {
			temp = getSubMatrix(inputLayer, i, j);
			result[i][j] = cross(temp, filter);
		}
	}

	return result;
}

Convolutional_layer::Convolutional_layer(size_t newFilterSize, size_t newInputSize)
{
	filterSize = newFilterSize;
	filter1 = createMatrix(newFilterSize);
	filter2 = createMatrix(newFilterSize);
	filter3 = createMatrix(newFilterSize);
	bias = createMatrix(newInputSize);

	randomInit();
}

std::vector<std::vector<double>> Convolutional_layer::propagate(std::vector<std::vector<double>>& input1, std::vector<std::vector<double>>& input2, std::vector<std::vector<double>>& input3)
{
	if (input1.size() < filter1.size()) {
		throw "input size is larger than filter size";
	}

	this->inputLayer1 = input1;
	this->inputLayer2 = input2;
	this->inputLayer3 = input3;

	auto result1 = cross_correlate(inputLayer1, filter1);
	auto result2 = cross_correlate(inputLayer2, filter2);
	auto result3 = cross_correlate(inputLayer3, filter3);


	auto result = createMatrix(input1.size() - filter1.size() + 1);

	for (int i = 0; i < result.size(); i++) {
		for (int j = 0; j < result.size(); j++) {
			result[i][j] = (result1[i][j] + result2[i][j] + result3[i][j]) + bias[i][j];
		}
	}
	activate(result);
	return result;
}

void Convolutional_layer::mutate(double weighChangeLimit, int chance)
{
	for (size_t i = 0; i < filter1.size(); i++) {
		for (size_t j = 0; j < filter1.size(); j++) {
			if (getRandomNumber(0, 100) < chance) {
				double change = getRandomNumber(-weighChangeLimit, weighChangeLimit);
				filter1[i][j] += change;
				change = getRandomNumber(-weighChangeLimit, weighChangeLimit);
				filter2[i][j] += change;
				change = getRandomNumber(-weighChangeLimit, weighChangeLimit);
				filter3[i][j] += change;
				change = getRandomNumber(-weighChangeLimit, weighChangeLimit);
				bias[i][j] += change;
			}
		}
	}
}
