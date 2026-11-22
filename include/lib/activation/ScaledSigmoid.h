#pragma once

#include <activation/ActivationFunction.h>

#include <string>
#include <bits/valarray_after.h>
#include <cmath>
#include <iostream>

namespace dnn {

    // Sigmoid whose output is linearly rescaled from (0, 1) to (min, max).
    class ScaledSigmoid : public ActivationFunction {
    public:
        ScaledSigmoid(double min, double max);

        double getMin() const {
            return min;
        }
        double getMax() const {
            return max;
        }

        double activation(double x) override {
            return min + (max - min) * (1.0 / (1.0 + std::exp(-x)));
        }
        double activationBackwards(double x) override {
            return (max-min) * (1.0 / (1.0 + std::exp(-x))) * (1.0 - 1.0 / (1.0 + std::exp(-x)));
        }
        std::string getName() const override {
            return "scaled_sigmoid";
        }

    private:
        double min;
        double max;
    };

}
