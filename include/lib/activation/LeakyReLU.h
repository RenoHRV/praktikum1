#pragma once

#include <activation/ActivationFunction.h>

#include <string>

namespace dnn {


class LeakyReLU : public ActivationFunction {
public:
    LeakyReLU() = default;
    ~LeakyReLU() override = default;

    double activation(double x) override {
        if (x > 0) {
            return x;
        }
        return 0.01 * x;
    }
    double activationBackwards(double x) override {
        if (x > 0) {
            return 1.0;
        }
        return 0.01;
    }
    std::string getName() const override {
        return "leaky_relu";
    }
};

}

