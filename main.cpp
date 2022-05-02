#include <iostream>
#include <memory>
#include <vector>
#include <Windows.h>
#include "NeuronLayer.h"
#include "NeuronNetwork.h"


std::vector<long double> numberToBitVector(unsigned char value){
	std::vector<long double> bit_vector;
	for (int i = 0; i < 8; i++) {
		bit_vector.push_back((value >> i) & 1);
	}
	return bit_vector;
}

unsigned char bitVectorToUnsignedChar(std::vector<long double>& bit_vector) {
	unsigned char value = 0;
	for (int i = 0; i < 8; i++) {
		value += (unsigned char)(std::round(bit_vector.at(i))) << i;
	}
	return value;
}

int main(){
	NeuronNetwork network;
	network.addLayer(8, false);
	network.addLayer(64, true);
	network.addLayer(64, true);
	network.addLayer(8, false);

	std::cout << "Input count of iterations ( 10000+ )\n>";
	int iterations = 0;
	std::cin >> iterations;

	for (int i = 0; i < iterations; i++) {
		unsigned char input_value  = i & 0xFF;
		unsigned char output_value = (input_value * 4);
		auto output_bit_vector = numberToBitVector(output_value);
		auto input_bit_vector = numberToBitVector(input_value);
		network.loadInputValues(input_bit_vector.data());
		network.computeLayers();
		network.computeLearn(output_bit_vector.data(), 0.05, 0.01);
		if( (i % 50000) == 0)
			network.displayInfo(output_bit_vector.data());
	}

	std::cout << "Lets test!" << std::endl;
	std::cout << "Input value from 0 to 127 \n";
	while (true) {
		std::cout << ">";
		int input_value;
		std::cin >> input_value;
		unsigned char output_value = (input_value * 4);
		auto output_bit_vector = numberToBitVector(output_value);
		auto input_bit_vector = numberToBitVector(input_value);
		network.loadInputValues(input_bit_vector.data());
		network.computeLayers();
		network.computeLearn(output_bit_vector.data(), 0.07, 0.03);
		//network.displayInfo(output_bit_vector.data());

		std::cout << (int)input_value << " * " << 4 << " = " << (int)bitVectorToUnsignedChar(network.getOutputArray()) << std::endl;
	}

}
