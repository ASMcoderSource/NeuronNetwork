#pragma once
#include <iostream>
#include <memory>
#include <string>
#include <vector>

class NeuronLayer {
    int16_t neurons = 0;
    std::weak_ptr<NeuronLayer> prew_layer;
    std::weak_ptr<NeuronLayer> next_layer;
    std::vector<long double> neurons_values;
    std::vector<long double> errors_values;
    std::vector<long double>* neurons_weights;
    std::vector<long double>* grad_moment;
    bool offset_neuron = false;

public:
    NeuronLayer(int16_t neurons, bool add_offset_neuron = true );
    ~NeuronLayer();
    
    int32_t getNeuronsCount();
    bool hasOffsetNeuron();
    void setNextLayer( std::shared_ptr<NeuronLayer>& next_layer_ptr );
    void setPrewLayer( std::shared_ptr<NeuronLayer>& prew_layer_ptr );
    void computeLayer();
    void computeErrorByArray(long double* array_ptr);
    void setWeight(int32_t neuron, int32_t weight, long double value);
    std::vector<long double>& getErrorArray();
    std::vector<long double>& getOutputArray();
    void getErrorByNextLayer();
    void computeLearn(long double E = 0.7, long double a = 0.3);
    void randomizeWeights();

    long double getAbsoluteError();
    static void linkLayers(std::shared_ptr<NeuronLayer>& prew_layer_ptr, std::shared_ptr<NeuronLayer>& next_layer_ptr );
    long double& operator[](int16_t neuron_index);

    static long double sigmoid(long double arg);
    static long double sigmoidDerivative(long double arg);
};