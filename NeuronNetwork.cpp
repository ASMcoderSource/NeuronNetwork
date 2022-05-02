#include "NeuronNetwork.h"

NeuronNetwork::NeuronNetwork() {

}

void NeuronNetwork::loadInputValues(long double* input_values) {
	if (!layers.empty()) {
		for (int32_t neuron_i = 0; neuron_i < layers[0]->getNeuronsCount(); neuron_i++) {
			(*layers[0])[neuron_i] = input_values[neuron_i];
		}
	}
}

void NeuronNetwork::addLayer(int32_t size, bool offset_neuron) {
	layers.push_back(std::shared_ptr<NeuronLayer>(new NeuronLayer(size, offset_neuron)));
	if (layers.size() > 1) {
		NeuronLayer::linkLayers(layers[layers.size() - 2], layers.back());
		layers[layers.size() - 2]->randomizeWeights();
	}
}

void NeuronNetwork::computeLayers() {
	for (int16_t layer_i = 1; layer_i < layers.size(); layer_i++) {
		layers[layer_i]->computeLayer();
	}
}

void NeuronNetwork::computeLearn(long double* ideal_array, long double E, long double a ) {
	for (int16_t layer_i = layers.size() - 1; layer_i >= 1; layer_i--) {
		if (layer_i == layers.size() - 1) {
			layers[layer_i]->computeErrorByArray(ideal_array);
		} else {
			layers[layer_i]->getErrorByNextLayer();
			layers[layer_i]->computeLearn(E, a);
		}
	}
}

void NeuronNetwork::displayInfo(long double* ideal_array) {
	std::cout << "Current neuron network error = " << layers.back()->getAbsoluteError(ideal_array) << std::endl;
	for (int32_t neuron_i = 0; neuron_i < layers.back()->getNeuronsCount(); neuron_i++) {
		std::cout << "Neuron [" << neuron_i << "] = " << std::setw(12) << (*layers.back())[neuron_i] << "  - ideal = " << ideal_array[neuron_i] << std::endl;
	}
	std::cout << std::endl;
}

std::vector<long double>& NeuronNetwork::getOutputArray() {
	return layers.back()->getOutputArray();
}