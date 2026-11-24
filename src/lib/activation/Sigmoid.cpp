#include <activation/Sigmoid.h>
namespace dnn {
    Sigmoid::Sigmoid() = default;
    Sigmoid::~Sigmoid() = default;

    double Sigmoid::activation(double x) {
        return 1.0 / (1.0 + std::exp (-x));
    }
    double Sigmoid::activationBackwards(double x) {
        return activation(x) * (1.0 - activation(x));
    }
    std::string Sigmoid::getName() const {
        return "sigmoid";
    }
}