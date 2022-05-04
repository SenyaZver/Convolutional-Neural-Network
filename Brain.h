#pragma once
#include <vector>
#include <iostream>
#include "Convolutional_layer.h"
#include "Reshape_layer.h"
#include "Dense_layer.h"

class Brain {
private:
	Convolutional_layer conv;
	Reshape_layer reshape;
	Dense_layer dense;

private:
	void printMatrix(std::vector<std::vector<double>>& const matrix) {
		size_t size = matrix.size();
		for (size_t i = 0; i < size; i++) {
			for (size_t j = 0; j < size; j++) {
				std::cout << matrix[i][j] << " ";
			}
			std::cout << std::endl;;
		}
	}
	std::vector<std::vector<double>> createMatrix(size_t const size) {
		std::vector<std::vector<double>> result(size);
		for (size_t i = 0; i < size; i++) {
			result[i].resize(size);
		}
		return result;
	}

	void print(std::vector<double> vec) {
		for (int i = 0; i < vec.size(); i++) {
			std::cout << vec[i] << " ";
		}
		std::cout << std::endl;
	}
	void create(size_t filterSize, size_t hiddenLayerSize) {
		conv = Convolutional_layer(filterSize, 30);
		size_t reshapeSize = (31 - filterSize) * (31 - filterSize);
		reshape = Reshape_layer(reshapeSize);
		dense = Dense_layer(reshapeSize, hiddenLayerSize);
	}

public:
	Brain(size_t filterSize, size_t hiddenLayerSize): conv(filterSize, hiddenLayerSize), 
		                                              reshape((31 - filterSize)* (31 - filterSize)),
		                                              dense((31 - filterSize) * (31 - filterSize), hiddenLayerSize) {}


	std::vector<double> feed_forward(std::vector<std::vector<double>>& const input1, 
									 std::vector<std::vector<double>>& const input2, 
									 std::vector<std::vector<double>>& const input3) 
	{
		auto convResult = conv.propagate(input1, input2, input3);
		auto reshapeResult = reshape.propagate(convResult);
		auto denseResult = dense.propagate(reshapeResult);
		return denseResult;
	}

	void mutate(double weighChangeLimit, int chance) {
		this->conv.mutate(weighChangeLimit, chance);
		this->dense.mutate(weighChangeLimit, chance);
	}
};

