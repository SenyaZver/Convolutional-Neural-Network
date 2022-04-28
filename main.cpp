#include <vector>
#include <iostream>
#include "Convolutional_layer.h"
#include "Reshape_layer.h"
#include "Dense_layer.h"

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
	input[0][1] = 1;
	input[0][2] = 1;

	input[1][0] = 1;
	input[1][1] = 2;
	input[1][2] = 1;

	input[2][0] = 1;
	input[2][1] = 2;
	input[2][2] = 2;



	Convoluted_layer test(2, 3);
	auto result = test.propagate(input, input, input);
	std::cout <<"propagating once" << std::endl << std::endl;;
	printMatrix(result);

	std::cout << std::endl << std::endl;


	test.mutate(1, 100);
	result = test.propagate(input, input, input);
	std::cout << "mutating and propagating again" << std::endl << std::endl;
	printMatrix(result);

	std::cout << std::endl << std::endl;

	Reshape_layer rL(4);
	auto result1 = rL.propagate(result);
	std::cout << "reshaping" << std::endl << std::endl;
	print(result1);

	std::cout << std::endl << std::endl;


	std::cout << "full dense layer" << std::endl << std::endl;
	Dense_layer test2(10, 4);
	auto result2 = test2.propagate(result1);
	print(result2);



}
