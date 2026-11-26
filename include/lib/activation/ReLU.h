#pragma once

#include <activation/ActivationFunction.h>

#include <string>

namespace dnn {

    class ReLU : public ActivationFunction {
    public:
        ReLU();
        ~ReLU() override;

        double activation(double x) override;
        double activationBackwards(double x) override;
        std::string getName() const override;
    };

}