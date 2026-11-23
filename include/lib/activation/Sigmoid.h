#pragma once

#include <activation/ActivationFunction.h>

#include <string>
#include <cmath>
#include <iostream>

namespace dnn {

    class Sigmoid : public ActivationFunction {
    public:
        Sigmoid();
        ~Sigmoid() override;

        double activation(double x) override;
        double activationBackwards(double x) override;
        std::string getName() const override;
    };

}