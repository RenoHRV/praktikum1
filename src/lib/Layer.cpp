#include <Layer.h>

#include <MathUtils.h>

#include <cmath>
#include <iostream>
#include <utility>

namespace dnn {

Layer::Layer(uint64_t id, uint64_t seed, uint64_t layerInputSize, uint64_t layerOutputSize,
             ActivationFunction* activationFunction, std::string  name)
    : id(id), name(std::move(name)), seed(seed), layerInputSize(layerInputSize), layerOutputSize(layerOutputSize),
      activationFunction(activationFunction) {
    accumulatedDWeights.assign(layerOutputSize, std::vector<double>(layerInputSize, 0.0));
    accumulatedDBias.assign(layerOutputSize, 0.0);
}

Layer::~Layer() {
    delete activationFunction;
}

void Layer::init() {
    std::mt19937 randomGenerator(seed);

    // Glorot/Xavier uniform initialization:
    // scales the weight range by fan-in/fan-out so pre-activation variance stays roughly O(1)
    // regardless of layer width, instead of growing with layerInputSize the way a fixed
    // [-1, 1] range does.
    const double limit = std::sqrt(6.0 / static_cast<double>(layerInputSize + layerOutputSize));
    std::uniform_real_distribution<double> distribution(-limit, limit);

    weights.assign(layerOutputSize, std::vector<double>(layerInputSize));
    for (auto& row : weights) {
        for (double& value : row) {
            value = distribution(randomGenerator);
        }
    }

    // Zero-initialized bias
    bias.assign(layerOutputSize, 0.0);
}

void Layer::setParameters(std::vector<std::vector<double>> weights, std::vector<double> bias) {
    this->weights = std::move(weights);
    this->bias = std::move(bias);
}

uint64_t Layer::getId() const {
    return id;
}

const std::string& Layer::getName() const {
    return name;
}

const std::vector<double>& Layer::getActivationOutput() const {
    return activationOutput;
}

const std::vector<double>& Layer::getPreActivationOutput() const {
    return preActivationOutput;
}

ActivationFunction* Layer::getActivationFunction() const {
    return activationFunction;
}

const std::vector<std::vector<double>>& Layer::getWeights() const {
    return weights;
}

const std::vector<double>& Layer::getBias() const {
    return bias;
}

const std::vector<double>& Layer::getDPreviousLayerOutput() const {
    return dPreviousLayerOutput;
}

const std::vector<std::vector<double>>& Layer::getDWeights() const {
    return dWeights;
}

const std::vector<double>& Layer::getDBias() const {
    return dBias;
}
    void Layer::forwardPropagation(const std::vector<double>& previousLayerOutput) {
    preActivationOutput.assign(layerOutputSize, 0.0);
    activationOutput.assign(layerOutputSize, 0.0);
    for (int i = 0; i < layerOutputSize; ++i) {
        double preActivationOutputSum = 0.0;
        for (int j = 0; j < weights[i].size(); ++j) {
            preActivationOutputSum += weights[i][j] * previousLayerOutput[j];
        }
        preActivationOutput[i] = preActivationOutputSum + bias[i];
    }
    for (int i = 0; i < layerOutputSize; ++i) {
        activationOutput[i] = activationFunction->activation(preActivationOutput[i]);
    }
}
    void Layer::backwardPropagation(const std::vector<double>& dActivationOutput, const std::vector<double>& previousLayerOutput) {
    //declare vector dPreActivation
    std::vector<double> dPreActivation;
    dPreActivation.assign(layerOutputSize, 0.0);
    //initialize vector dPreActivation
    for (int i = 0; i < layerOutputSize; ++i) {
        dPreActivation[i] = dActivationOutput[i] * activationFunction->activationBackwards(preActivationOutput[i]);
    }
    //compute dWeights matrix
    dWeights.assign(layerOutputSize, std::vector<double>(layerInputSize, 0.0));
    for (int i = 0; i < layerOutputSize; ++i) {
        for (int j = 0; j < layerInputSize; ++j) {
            dWeights[i][j] = dPreActivation[i] * previousLayerOutput[j];
        }
    }
    //save computed dPreActivation in dBias
    dBias.assign(layerOutputSize, 0.0);
    for (int i = 0; i < dPreActivation.size(); ++i) {
        dBias[i] = dPreActivation[i];
    }
    //update accumulatedDWeights
    for (int i = 0; i < accumulatedDWeights.size(); ++i) {
        for (int j = 0; j < accumulatedDWeights[i].size(); ++j) {
            accumulatedDWeights[i][j] += dWeights[i][j];
        }
    }
    //update accumulatedDBias
    for (int i = 0; i < accumulatedDBias.size(); ++i) {
        accumulatedDBias[i] += dBias[i];
    }
    //increment numProcessedSamples
    numProcessedSamples++;
    //compute dPreviousLayerOutput
    dPreviousLayerOutput.assign(layerInputSize, 0.0);
    std::vector<double> weightsColumn(weights.size());
    for (int j = 0; j < layerInputSize; ++j) {
        for (int i = 0; i < weights.size(); ++i) {
            weightsColumn[i] = weights[i][j];
        }
        double sum = 0.0;
        for (int k = 0; k < weightsColumn.size(); ++k) {
            sum += weightsColumn[k] * dPreActivation[k];
        }
        dPreviousLayerOutput[j] = sum;
    }
}
    void Layer::updateParameters(const std::vector<std::vector<double>>& dWeights, const std::vector<double>& dBias, double learningRate) {
    //update weights
    for (int i = 0; i < weights.size(); ++i) {
        for (int j = 0; j < weights[i].size(); ++j) {
            weights[i][j] -= learningRate * dWeights[i][j];
        }
    }
    //update bias
    for (int i = 0; i < bias.size(); ++i) {
        bias[i] -= learningRate * dBias[i];
    }
}
    void Layer::commitToWeights(double learningRate) {

    if (numProcessedSamples == 0) {
        return;
    }

    //prepare accumulatedDWeights
    for (int i = 0; i < accumulatedDWeights.size(); ++i) {
        for (int j = 0; j < accumulatedDWeights[i].size(); ++j) {
            accumulatedDWeights[i][j] /= numProcessedSamples;
        }
    }

    //prepare accumulatedDBias
    for (int i = 0; i < accumulatedDBias.size(); ++i) {
        accumulatedDBias[i] /= numProcessedSamples;
    }

    //update params
    updateParameters(accumulatedDWeights, accumulatedDBias, learningRate);

    //reset accumulatedDWeights, accumulatedDBias and numProcessedSamples
    accumulatedDWeights.assign(layerOutputSize, std::vector<double>(layerInputSize, 0.0));
    accumulatedDBias.assign(accumulatedDBias.size(), 0.0);
    numProcessedSamples = 0;
}
}
