#pragma once

#include <activation/ActivationFunction.h>

#include <string>

namespace dnn {


class LeakyReLU : public ActivationFunction {
public:
    LeakyReLU();
    ~LeakyReLU() override;

    double activation(double x) override;
    double activationBackwards(double x) override;
    std::string getName() const override;
};

}

