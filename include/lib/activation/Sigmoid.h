#pragma once

#include <activation/ActivationFunction.h>

#include <string>
#include <cmath>
#include <iostream>

namespace dnn {

    class Sigmoid : public ActivationFunction {
    public:
        Sigmoid() = default;
        ~Sigmoid() override = default;

        double activation(double x) override {
            return 1.0 / (1.0 + std::exp (-x));
        }
        double activationBackwards(double x) override {
            return activation(x) * (1.0 - activation(x));
        }
        std::string getName() const override {
            return "sigmoid";
        }
    };

}