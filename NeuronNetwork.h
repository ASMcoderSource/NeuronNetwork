#pragma once
#include <memory>
#include <iomanip>
#include <iostream>
#include "NeuronLayer.h"

class NeuronNetwork{
	std::vector<std::shared_ptr<NeuronLayer> > layers;
public:
	NeuronNetwork();
	void addLayer(int32_t size, bool offset_neuron );
	void loadInputValues(long double* input_values);
	void computeLayers();
	void computeLearn(long double* ideal_array, long double E = 0.7, long double a = 0.3);
	void displayInfo(long double* ideal_array);
	std::vector<long double>& getOutputArray();
};

