#pragma once
#include <vector>
#include <iostream>
#include <random>


class Dense_layer {
private:
	size_t inputSize;
	size_t hiddenLayerSize;

	std::vector<double> inputLayer;
	std::vector<std::vector<double>> hiddenLayer;
	std::vector<double> bias;


private:
	std::vector<std::vector<double>> createMatrix(size_t verticalSize, size_t horizontalSize) {
		std::vector<std::vector<double>> result(horizontalSize);
		for (size_t i = 0; i < horizontalSize; i++) {
			result[i].resize(verticalSize);
		}
		return result;
	}

	double getRandomNumber(double begin, double end) {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<> dis(begin, end);

		return dis(gen);
	}

	double sigmoid(double const x) {
		return 1 / (1 + exp(-x));
	}
	void randomInit() {
		for (size_t i = 0; i < 2; i++) {
			for (size_t j = 0; j < hiddenLayerSize; j++) {
				hiddenLayer[i][j] = getRandomNumber(-1, 1);
			}
			bias[i] = getRandomNumber(-1, 1);
		}
	}


private:
	//debug function
	void print(std::vector<std::vector<double>>& const matrix) {
		for (size_t i = 0; i < 2; i++) {
			for (size_t j = 0; j < hiddenLayerSize; j++) {
				std::cout << matrix[i][j] << " ";
			}
			std::cout << std::endl;;
		}
	}
	//

public:
	Dense_layer(size_t newInputSize, size_t newHiddenLayerSize) {
		hiddenLayerSize = newHiddenLayerSize;
		inputSize = newInputSize;

		hiddenLayer = createMatrix(newHiddenLayerSize, 2);
		bias = std::vector<double>(newHiddenLayerSize);

		randomInit();
	}

	std::vector<double> propagate(std::vector<double>& input) {
		std::vector<double> result(2);

		for (size_t i = 0; i < 2; i++)
		{
			for (size_t j = 0; j < hiddenLayerSize; j++)
			{
				result[i] = 0;
				for (size_t k = 0; k < 2; k++)
				{
					result[i] += hiddenLayer[i][k] * input[k]+bias[k];
				}
			}
		}
		return result;

	}

	void mutate(double weighChangeLimit, int chance) {
		for (size_t i = 0; i < 2; i++)
		{
			for (size_t j = 0; j < hiddenLayerSize; j++)
			{
				if (getRandomNumber(0, 100) < chance) {
					auto change = getRandomNumber(-weighChangeLimit, weighChangeLimit);
					hiddenLayer[i][j] += change;
				}
			}
			if (getRandomNumber(0, 100) < chance) {
				auto change = getRandomNumber(-weighChangeLimit, weighChangeLimit);
				bias[i] += change;
			}
		}
	}



};
