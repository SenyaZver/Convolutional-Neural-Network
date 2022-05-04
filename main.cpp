#include <vector>
#include <iostream>
#include "Convolutional_layer.h"
#include "Reshape_layer.h"
#include "Dense_layer.h"
#include "Brain.h"

void printMatrix(std::vector<std::vector<double>> matrix) {
	size_t size = matrix.size();
	for (size_t i = 0; i < size; i++) {
		for (size_t j = 0; j < size; j++) {
			std::cout << matrix[i][j] << " ";
		}
		std::cout << std::endl;;
	}
}
std::vector<std::vector<double>> createMatrix(size_t size) {
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


int main() {

	//init input
	auto input = createMatrix(30);
	for (int i = 0; i < 30; i++) {
		for (int j = 0; j < 30; j++) {
			input[i][j] = 1;
		}
	}


	//Convolutional_layer test(11, 30);
	//auto result = test.propagate(input, input, input);
	//std::cout <<"propagating once" << std::endl << std::endl;
	//printMatrix(result);
	//std::cout << std::endl << std::endl;
	Brain test(11, 30);

	auto result = test.feed_forward(input, input, input);
	print(result);

	test.mutate(0.1, 70);
	result = test.feed_forward(input, input, input);
	print(result);
}
