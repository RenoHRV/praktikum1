#include <Network.h>

#include <iostream>
#include <stdexcept>

namespace dnn {
    

    Network::Network(std::vector<std::tuple<std::vector<std::vector<double>>, std::vector<double>, ActivationFunction*>> layerParameters,
                 LossFunction* lossFunction)
    : ownedLossFunction(lossFunction) {

        uint64_t id = 0;
        for (auto& [weights, bias, activationFunction] : layerParameters) {
            // Ausgabegröße = Anzahl Zeilen von weights, Eingabegröße = Länge von weights[0]
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
