#pragma once
#include <vector>
#include <iostream>


class Convoluted_layer
{

private:
	size_t filterSize;

	std::vector<std::vector<double>> inputLayer;
	std::vector<std::vector<double>> filter;



private:
	std::vector<std::vector<double>> createMatrix(size_t size) {
		std::vector<std::vector<double>> result(size);
		for (size_t i = 0; i < size; i++) {
			result[i].resize(size);
		}
		return result;
	}

	double operation(std::vector<std::vector<double>>& const first, std::vector<std::vector<double>>& const second)
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

	std::vector<std::vector<double>> getSubMatrix(size_t verticalOffset, size_t horizontalOffset) {
		size_t size = inputLayer.size() - filter.size() + 1;

		auto result = createMatrix(size);


		for (size_t i = 0; i < size; i++) {
			for (size_t j = 0; j < size; j++) {
				result[i][j] = inputLayer[i + verticalOffset][j + horizontalOffset];
			}
		}
		return result;
	}

	double sigmoid(double const x) {
		return 1 / (1 + exp(-x));
	}

	void activate(std::vector<std::vector<double>>& const layer) {
		size_t size = layer.size();
		for (size_t i = 0; i < size; i++) {
			for (size_t j = 0; j < size; j++) {
				layer[i][j] = sigmoid(layer[i][j]);
			}
		}
	}

	void getRandomFilter() {
		for (size_t i = 0; i < filterSize; i++) {
			for (size_t j = 0; j < filterSize; j++) {
				filter[i][j] = rand() % 10;
			}
		}

	}

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
	//

	std::vector<std::vector<double>> cross_correlate() {
		size_t size = inputLayer.size() - filter.size() + 1;

		auto result = createMatrix(size);
		auto temp = createMatrix(size);

		size_t verticalOffset = 0;
		size_t horizontalOffset = 0;

		for (size_t i = 0; i < size; i++) {
			for (size_t j = 0; j < size; j++) {
				temp = getSubMatrix(i, j);
				result[i][j] = operation(temp, filter);
			}
		}

		return result;
	}


public:
	Convoluted_layer(size_t newFilterSize) {
		filterSize = newFilterSize;
		filter = createMatrix(newFilterSize);
		getRandomFilter();
	}

	Convoluted_layer(std::vector<std::vector<double>> newFilter) {
		filterSize = newFilter.size();
		filter = newFilter;
	}


	std::vector<std::vector<double>> propagate(std::vector<std::vector<double>>& input) {
		if (input.size() < filter.size()) {
			throw "input size is larger than filter size";
		}
		this->inputLayer = input;

		auto result = cross_correlate();
		activate(result);
		return result;
	}


};