#pragma once
#include <vector>
#include <iostream>




class Reshape_layer {
private:
	size_t size;
	std::vector<std::vector<double>> inputLayer;



	std::vector<double> reshape() {
		auto result = std::vector<double>(size);

		size_t index = 0;

		for (size_t i = 0; i < inputLayer.size(); i++) {
			for (size_t j = 0; j < inputLayer.size(); j++) {
				result[index] = inputLayer[i][j];
				index++;
			}
		}

		return result;
	}

public:
	Reshape_layer(std::vector<std::vector<double>> const  input) {
		inputLayer = input;
		size = input.size() * input.size();
	}

	Reshape_layer(size_t size) {
		this->size = size;
	}



	std::vector<double> propagate(std::vector<std::vector<double>> const input) {
		this->inputLayer = input;
		return this->reshape();
	}

	std::vector<double> propagate() {
		return this->reshape();
	}







};