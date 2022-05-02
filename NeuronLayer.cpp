#include "NeuronLayer.h"

NeuronLayer::NeuronLayer(int16_t neurons, bool add_offset_neuron ) {
    neurons_values.resize(neurons);
    if (add_offset_neuron) {
        offset_neuron = true;
        neurons_values[neurons - 1] = 1;
        neurons_values.resize(neurons + 1);
        neurons++;
    }
    NeuronLayer::neurons = neurons;
    errors_values.resize(neurons);
    neurons_weights = new std::vector<long double>[neurons];
    grad_moment = new std::vector<long double>[neurons];
}

NeuronLayer::~NeuronLayer() {
    if( neurons_weights != nullptr )
         delete[] neurons_weights;
}

long double& NeuronLayer::operator[](int16_t neuron_index) {
    if (neuron_index >= neurons)
        throw std::string("neuron index error");
    return neurons_values[neuron_index];
}

void NeuronLayer::setNextLayer(std::shared_ptr<NeuronLayer>& next_layer_ptr) {
    this->next_layer = next_layer_ptr;
    int32_t weights_count = next_layer_ptr->neurons;
    for (int16_t neuron_i = 0; neuron_i < neurons; neuron_i++) {
        neurons_weights[neuron_i].resize(weights_count);
        grad_moment[neuron_i].resize(weights_count);
        std::fill(grad_moment[neuron_i].begin(), grad_moment[neuron_i].end(), 0);
    }
}

void NeuronLayer::setPrewLayer(std::shared_ptr<NeuronLayer>& prew_layer_ptr) {
    this->prew_layer = prew_layer_ptr;
}

void NeuronLayer::linkLayers(std::shared_ptr<NeuronLayer>& prew_layer_ptr, std::shared_ptr<NeuronLayer>& next_layer_ptr) {
    prew_layer_ptr->setNextLayer(next_layer_ptr);
    next_layer_ptr->setPrewLayer(prew_layer_ptr);
}

void NeuronLayer::setWeight(int32_t neuron, int32_t weight, long double value) {
    neurons_weights[neuron][weight] = value;
}

bool NeuronLayer::hasOffsetNeuron() {
    return offset_neuron;
}

void NeuronLayer::computeLayer() {
    int32_t neurons_count = hasOffsetNeuron() ? neurons - 1: neurons;
    for (int32_t neuron_i = 0; neuron_i < neurons_count; neuron_i++) {
        long double input_sum = 0;
        auto prew_layer = this->prew_layer.lock().get();
        int32_t weights = prew_layer->neurons;
        for (int32_t weight_i = 0; weight_i < weights; weight_i++) {
            input_sum += (*prew_layer)[weight_i] * prew_layer->neurons_weights[weight_i][neuron_i];
        }
        neurons_values[neuron_i] = sigmoid(input_sum);
    }
}

void NeuronLayer::computeErrorByArray(long double* array_ptr) {
    int32_t neurons_count = hasOffsetNeuron() ? neurons - 1 : neurons;
    for (int32_t neuron_i = 0; neuron_i < neurons_count; neuron_i++) {
        errors_values[neuron_i] =  (array_ptr[neuron_i] - neurons_values[neuron_i]) * sigmoidDerivative(neurons_values[neuron_i]);
    }
}

void NeuronLayer::computeLearn(long double E, long double a) {
    auto next_layer = this->next_layer.lock().get();
    int32_t neurons_count = neurons;
    int32_t weights_count = next_layer->hasOffsetNeuron() ? next_layer->neurons - 1 : next_layer->neurons;

    for (int32_t neuron_i = 0; neuron_i < neurons_count; neuron_i++) {
        long double neuron_value = neurons_values[neuron_i];
        for (int32_t weight_i = 0; weight_i < weights_count; weight_i++) {
            long double grad = next_layer->errors_values[weight_i] * neuron_value;
            long double weight_delta = E * grad + (a * grad_moment[neuron_i][weight_i]);
            neurons_weights[neuron_i][weight_i] += weight_delta;
            grad_moment[neuron_i][weight_i] = weight_delta;
        }
    }
}

void NeuronLayer::computeMultiThreadLearn(long double E, long double a) {
    auto next_layer = this->next_layer.lock().get();
    int32_t neurons_count = neurons;
    int32_t weights_count = next_layer->hasOffsetNeuron() ? next_layer->neurons - 1 : next_layer->neurons;
    std::atomic<int32_t> neuron_i = 0;
    std::atomic<int32_t> completed_neurons = 0;

    std::mutex neuron_mutex;
    std::vector<std::thread> threads;
    for (int i = 0; i < std::thread::hardware_concurrency(); i++) {
        threads.push_back(std::thread([&, this]() {
            neuron_mutex.lock();
            int32_t thread_neuron_i = neuron_i;
            neuron_i++;
            neuron_mutex.unlock();
            long double neuron_value = neurons_values[neuron_i];
            for (int32_t weight_i = 0; weight_i < weights_count; weight_i++) {
                long double grad = next_layer->errors_values[weight_i] * neuron_value;
                long double weight_delta = E * grad + (a * grad_moment[neuron_i][weight_i]);
                neurons_weights[neuron_i][weight_i] += weight_delta;
                grad_moment[neuron_i][weight_i] = weight_delta;
            }
            completed_neurons++;
            }));
    }
    while (completed_neurons < neurons_count)
        std::this_thread::yield();

}

std::vector<long double>& NeuronLayer::getErrorArray() {
    return errors_values;
}

void NeuronLayer::getErrorByNextLayer() {
    int32_t neurons_count = neurons;
    auto next_layer = this->next_layer.lock().get();
    for (int32_t neuron_i = 0; neuron_i < neurons_count; neuron_i++) {
        long double sum = 0;
        for (int32_t weight_i = 0; weight_i < next_layer->neurons; weight_i++) {
            sum += neurons_weights[neuron_i][weight_i] * next_layer->errors_values[weight_i];
        }
        errors_values[neuron_i] = sum * sigmoidDerivative(neurons_values[neuron_i]);
    }
}

long double NeuronLayer::getAbsoluteError(long double* array_ptr) {
    long double error = 0;
    for (int32_t neuron_i = 0; neuron_i < neurons; neuron_i++) {
        error += pow(array_ptr[neuron_i] - neurons_values[neuron_i], 2);
    }
    error /= neurons;
    return error;
}

long double NeuronLayer::sigmoid(long double arg) {
    return 1.0 / (1 + exp(-arg));
}

long double NeuronLayer::sigmoidDerivative(long double arg) {
    return (1.0 - arg) * arg;
}


void NeuronLayer::randomizeWeights(){
    srand(time(0));
    auto next_layer = this->next_layer.lock().get();
    for (int16_t neuron_i = 0; neuron_i < neurons; neuron_i++) {
        for (int16_t weight_i = 0; weight_i < next_layer->neurons; weight_i++) {
            long double f = (long double)rand() / RAND_MAX;
            f = f * 0.9 + 0.1;
            neurons_weights[neuron_i][weight_i] = f / next_layer->neurons;
        }
    }
}

std::vector<long double>& NeuronLayer::getOutputArray() {
    return neurons_values;
}

int32_t NeuronLayer::getNeuronsCount() {
    return neurons;
}