#pragma once

#include <activation/ActivationFunction.h>

#include <string>

namespace dnn {

    class ReLU : public ActivationFunction {
    public:
        ReLU() = default;
        ~ReLU() override = default;

        double activation(double x) override {
            if (x > 0) {
                return x;
            }
            return 0.0;
        }
        double activationBackwards(double x) override {
            if (x > 0) {
                return 1.0;
            }
            return 0.0;
        }
        std::string getName() const override {
            return "relu";
        }
    };

}