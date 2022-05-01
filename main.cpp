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
	network.addLayer(16, true);
	network.addLayer(8, false);


	for (int i = 0; i < 1000000; i++) {
		unsigned char input_value  = i & 0xFF;
		unsigned char output_value = (input_value << 1);
		auto output_bit_vector = numberToBitVector(output_value);
		auto input_bit_vector = numberToBitVector(input_value);
		network.loadInputValues(input_bit_vector.data());
		network.computeLayers();
		network.computeLearn(output_bit_vector.data(), 0.07, 0.03);
		if( (i % 10000) == 0)
			network.displayInfo(output_bit_vector.data());
	}

	while (true) {
		std::cout << "Lets test!" << std::endl;
		std::cout << "Input value from 0 to 255 \n";
		std::cout << ">";
		int input_value;
		std::cin >> input_value;
		unsigned char output_value = (input_value << 1);
		auto output_bit_vector = numberToBitVector(output_value);
		auto input_bit_vector = numberToBitVector(input_value);
		network.loadInputValues(input_bit_vector.data());
		network.computeLayers();
		network.computeLearn(output_bit_vector.data(), 0.07, 0.03);
		//network.displayInfo(output_bit_vector.data());

		std::cout << (int)input_value << " * " << 2 << " = " << (int)bitVectorToUnsignedChar(network.getOutputArray()) << std::endl;
	}

}
