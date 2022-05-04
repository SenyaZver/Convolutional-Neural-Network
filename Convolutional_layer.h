#pragma once
#include <vector>
#include <iostream>
#include <random>

//rushed code, fix handling different amount of filters


class Convolutional_layer
{

private:
	size_t filterSize;

	std::vector<std::vector<double>> inputLayer1;
	std::vector<std::vector<double>> inputLayer2;
	std::vector<std::vector<double>> inputLayer3;

	std::vector<std::vector<double>> filter1;
	std::vector<std::vector<double>> filter2;
	std::vector<std::vector<double>> filter3;


	std::vector<std::vector<double>> bias;


private:
	double getRandomNumber(double begin, double end);
	std::vector<std::vector<double>> createMatrix(size_t size);
	void randomInit();


	double sigmoid(double const x);
	void activate(std::vector<std::vector<double>>& const layer);


	double cross(std::vector<std::vector<double>>& const first, std::vector<std::vector<double>>& const second);
	std::vector<std::vector<double>> getSubMatrix(std::vector<std::vector<double>>& inputLayer, size_t verticalOffset, size_t horizontalOffset);



	std::vector<std::vector<double>> cross_correlate(std::vector<std::vector<double>>& inputLayer, std::vector<std::vector<double>>& filter);


public:
	//debug function
	void print(std::vector<std::vector<double>>& const matrix) {
		size_t size = matrix.size();
		for (size_t i = 0; i < size; i++) {
			for (size_t j = 0; j < size; j++) {
				std::cout << matrix[i][j] << " ";
			}
			std::cout << std::endl;;
		}
	}

	void printFilter() {
		print(this->filter1);
		std::cout << std::endl;
		print(this->filter2);
		std::cout << std::endl;
		print(this->filter3);
		std::cout << std::endl;
	}
	//



public:
	Convolutional_layer(size_t newFilterSize, size_t newInputSize);

	std::vector<std::vector<double>> propagate(std::vector<std::vector<double>>& input1, std::vector<std::vector<double>>& input2, std::vector<std::vector<double>>& input3);

	void mutate(double weighChangeLimit, int chance);
};