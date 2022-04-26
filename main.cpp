#include <vector>
#include <iostream>
#include "Convolutional_layer.h"
#include "Reshape_layer.h"

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
	auto input = createMatrix(3);
	input[0][0] = 1;
	input[0][1] = 6;
	input[0][2] = 2;

	input[1][0] = 5;
	input[1][1] = 3;
	input[1][2] = 1;

	input[2][0] = 7;
	input[2][1] = 0;
	input[2][2] = 4;

	//init filter
	auto filter = createMatrix(2);
	filter[0][0] = 1;
	filter[0][1] = 2;

	filter[1][0] = -1;
	filter[1][1] = 0;



	Convoluted_layer test(filter);

	auto result = test.propagate(input);


	printMatrix(result);

	Reshape_layer test1(result);
	
	print(test1.propagate());

}
