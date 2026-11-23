#include <loss/BinaryCrossEntropy.h>

#include <MathUtils.h>

#include <algorithm>
#include <cmath>

namespace dnn {
    double BinaryCrossEntropy::getCostValue(const std::vector<double> &predictedOutput, const std::vector<double> &actualOutput) const {
        double result = 0.0;
        const std::size_t n = predictedOutput.size();
        for (std::size_t i= 0; i < n; i++) {
            double p = clampProbability(predictedOutput[i]);
            result += actualOutput[i] * std::log(p) + (1.0 - actualOutput[i]) * std::log(1.0 - p);
        }
        return -result / n;
    }

    double BinaryCrossEntropy::getAccuracyValue(const std::vector<double>& predictedOutput, const std::vector<double>& actualOutput) const {
        int correct = 0;
        const std::size_t n = predictedOutput.size();
        for (std::size_t i = 0; i < n; i++) {
            double predicted = predictedOutput[i] > 0.5 ? 1.0 : 0.0;
            if (predicted == actualOutput[i]) {
                correct++;
            }
        }
        return static_cast<double>(correct) / n;
    }

    std::vector<double> BinaryCrossEntropy::getGradient(const std::vector<double> &predictedOutput, const std::vector<double> &actualOutput) const {
        std::vector<double> gradient;
        const std::size_t n = predictedOutput.size();
        for (std::size_t i = 0; i < n; i++) {
            double p = clampProbability(predictedOutput[i]);
            double value = -(actualOutput[i] / p - (1.0 - actualOutput[i]) / (1.0 - p));
            gradient.push_back(value);
        }
        return gradient;
    }

    std::string BinaryCrossEntropy::getName() const {
    return "binary_cross_entropy";
}

}
