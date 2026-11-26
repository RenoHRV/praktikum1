#include <Network.h>

#include <iostream>
#include <stdexcept>

namespace dnn {
    namespace {

        void validateData(const std::vector<Layer*>& layers,
                          const std::vector<std::vector<double>>& inputs,
                          const std::vector<std::vector<double>>& actualOutputs) {
            if (layers.empty()) {
                throw std::invalid_argument("Network has no layers");

            }

            if (inputs.empty() || actualOutputs.empty()) {


                throw std::invalid_argument("inputs and actualOutputs must not be empty");
            }
            if (inputs.size() != actualOutputs.size()) {
                throw std::invalid_argument("inputs and actualOutputs must have the same size (got " +std::to_string(inputs.size()) + " and " +std::to_string(actualOutputs.size()) + ")");
            }

            const auto& firstWeights = layers.front()->getWeights();
            const auto& lastWeights  = layers.back()->getWeights();
            std::size_t inputSize  = firstWeights.empty() ? 0 : firstWeights[0].size();

            std::size_t outputSize = lastWeights.size();

            for (std::size_t i = 0; i < inputs.size(); ++i) {
                if (inputs[i].size() != inputSize) {
                    throw std::invalid_argument("input " + std::to_string(i) + " has size " + std::to_string(inputs[i].size()) + ", but the first layer expects " + std::to_string(inputSize));
                }
                if (actualOutputs[i].size() != outputSize) {
                    throw std::invalid_argument("actual output " + std::to_string(i) + " has size " + std::to_string(actualOutputs[i].size()) + ", but the last layer produces " + std::to_string(outputSize));
                }
            }
        }

    }

    Network::Network(std::vector<std::tuple<std::vector<std::vector<double>>, std::vector<double>, ActivationFunction*>> layerParameters,
                 LossFunction* lossFunction)
    : ownedLossFunction(lossFunction) {

        uint64_t id = 0;
        for (auto& [weights, bias, activationFunction] : layerParameters) {
            uint64_t outputSize = weights.size();
            uint64_t inputSize  = weights.empty() ? 0 : weights[0].size();


            Layer* layer = new Layer(id, nextSeed++, inputSize, outputSize, activationFunction);

            layer->setParameters(std::move(weights), std::move(bias));
            layers.push_back(layer);

            ++id;
        }
    }




    Network::~Network() {
        for (Layer* layer : layers) {
            delete layer;
        }
        delete ownedLossFunction;
    }


    Network& Network::addLayer(uint64_t inputDim, uint64_t outputDim, ActivationFunction* activationFunction) {
        uint64_t id = layers.size();

        Layer* layer = new Layer(id, nextSeed, inputDim, outputDim, activationFunction);
        layer->init();
        layers.push_back(layer);

        ++nextSeed;

        return *this;
    }

    const std::vector<double>& Network::getLayerInput(uint64_t layerId) const {
        if (layerId == 0) {
            return initialInput;
        }
        return layers[layerId - 1]->getActivationOutput();
    }

    void Network::fullForwardPropagation(const std::vector<double>& input) {
        initialInput = input;
        const std::vector<double>* current = &initialInput;

        for (Layer* layer : layers) {
            layer->forwardPropagation(*current);

            current = &layer->getActivationOutput();

        }

        output = *current;
    }

    void Network::fullBackwardPropagation(const std::vector<double>& predictedOutput,const std::vector<double>& actualOutput) {
        if (ownedLossFunction == nullptr) {
            throw std::logic_error("fullBackwardPropagation: no loss function set");
        }

        std::vector<double> dACurr = ownedLossFunction->getGradient(predictedOutput, actualOutput);

        for (std::size_t i = layers.size(); i-- > 0; ) {

            layers[i]->backwardPropagation(dACurr, getLayerInput(i));
            dACurr = layers[i]->getDPreviousLayerOutput();
        }

    }


void Network::trainBatch(const std::vector<std::vector<double>>& inputs,
                         const std::vector<std::vector<double>>& actualOutputs,
                         std::size_t batchSize, int epochs, double learningRate) {
    if (ownedLossFunction == nullptr) {
        throw std::logic_error("trainBatch: no loss function set");
    }
    validateData(layers, inputs, actualOutputs);
    if (batchSize == 0) {
        throw std::invalid_argument("batchSize must be greater than 0");
    }

    costHistory.clear();
    accuracyHistory.clear();

    const std::size_t n = inputs.size();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double epochCost = 0.0;
        double epochAccuracy = 0.0;

        for (std::size_t i = 0; i < n; ++i) {
            fullForwardPropagation(inputs[i]);

            epochCost     += ownedLossFunction->getCostValue(output, actualOutputs[i]);
            epochAccuracy += ownedLossFunction->getAccuracyValue(output, actualOutputs[i]);

            fullBackwardPropagation(output, actualOutputs[i]);

            if (i % batchSize == 0 || i == n - 1) {
                for (Layer* layer : layers) {
                    layer->commitToWeights(learningRate);
                }
            }
        }

        costHistory.push_back(epochCost / static_cast<double>(n));
        accuracyHistory.push_back(epochAccuracy / static_cast<double>(n));
    }
}

double Network::computeAccuracy(const std::vector<std::vector<double>>& inputs,
                                const std::vector<std::vector<double>>& actualOutputs) {
    if (ownedLossFunction == nullptr) {
        throw std::logic_error("computeAccuracy: no loss function set");
    }
    validateData(layers, inputs, actualOutputs);

    double totalCorrect = 0.0;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        fullForwardPropagation(inputs[i]);
        totalCorrect += ownedLossFunction->getAccuracyValue(output, actualOutputs[i]);
    }
    return totalCorrect / static_cast<double>(inputs.size());
}


void Network::setLossFunction(LossFunction* lossFunction) {
    delete this->ownedLossFunction;
    this->ownedLossFunction = lossFunction;
}

const LossFunction* Network::getLossFunction() const {
    return ownedLossFunction;
}

const std::vector<double>& Network::getOutput() const {
    return output;
}

const std::vector<double>& Network::getCostHistory() const {
    return costHistory;
}

const std::vector<double>& Network::getAccuracyHistory() const {
    return accuracyHistory;
}

const std::vector<Layer*>& Network::getLayers() const {
    return layers;
}

void Network::setLayers(std::vector<Layer*> newLayers) {
    for (Layer* layer : layers) {
        delete layer;
    }
    layers = std::move(newLayers);
    initialInput.clear();
    output.clear();
    costHistory.clear();
    accuracyHistory.clear();
}

}
