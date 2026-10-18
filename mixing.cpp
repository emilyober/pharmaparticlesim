#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <iomanip>

struct Counts {
    int total{0};
    int blue{0};
};

using Cell = std::tuple<int, int, int>;

int main() {
    try {
        std::ifstream input("experiments/results/local_timestepping/pilot_10rpm_6s.csv");
        if (!input) {
            throw std::runtime_error(
                "Cannot open drum_mixing.csv"
            );
        }

        std::ofstream output("experiments/results/local_timestepping/pilot_10rpm_6s_mixing.csv");
        
        output << std::setprecision(17);
        if (!output) {
            throw std::runtime_error(
                "Cannot create mixing_score.csv"
            );
        }

        output
            << "time,mixing_score,coverage,eligible_cells\n";

        constexpr double cellSize = 0.004;
        constexpr int minimumParticles = 5;

        std::map<Cell, Counts> cells;

        std::string currentStep;
        double currentTime = 0.0;
        int totalParticles = 0;

        const auto writeFrame = [&]() {
            if (totalParticles == 0) {
                return;
            }

            int included = 0;
            int includedBlue = 0;
            int eligibleCells = 0;

            for (const auto& [key, count] : cells) {
                if (count.total >= minimumParticles) {
                    included += count.total;
                    includedBlue += count.blue;
                    ++eligibleCells;
                }
            }

            const double coverage =
                static_cast<double>(included) /
                totalParticles;

            output << currentTime << ',';

            if (included == 0 ||
                includedBlue == 0 ||
                includedBlue == included) {
                output << "nan";
            } else {
                const double overallFraction =
                    static_cast<double>(includedBlue) /
                    included;

                double variance = 0.0;

                for (const auto& [key, count] : cells) {
                    if (count.total < minimumParticles) {
                        continue;
                    }

                    const double localFraction =
                        static_cast<double>(count.blue) /
                        count.total;

                    const double difference =
                        localFraction - overallFraction;

                    variance +=
                        count.total * difference * difference;
                }

                variance /= included;

                const double score =
                    1.0 - variance /
                    (overallFraction *
                     (1.0 - overallFraction));

                output << score;
            }

            output << ',' << coverage
                   << ',' << eligibleCells << '\n';
        };

        std::string line;
        std::getline(input, line); // Header.

        while (std::getline(input, line)) {
            if (line.empty()) {
                continue;
            }

            std::stringstream stream(line);
            std::vector<std::string> fields;
            std::string field;

            while (std::getline(stream, field, ',')) {
                fields.push_back(field);
            }

            if (fields.size() != 14) {
                throw std::runtime_error(
                    "Unexpected CSV row; use a completed output file"
                );
            }

            if (!currentStep.empty() &&
                fields[0] != currentStep) {
                writeFrame();
                cells.clear();
                totalParticles = 0;
            }

            currentStep = fields[0];
            currentTime = std::stod(fields[1]);

            const int group = std::stoi(fields[3]);

            if (group != 0 && group != 1) {
                throw std::runtime_error(
                    "This tool expects groups 0 and 1"
                );
            }

            const double x = std::stod(fields[4]);
            const double y = std::stod(fields[5]);
            const double z = std::stod(fields[6]);

            const Cell key{
                static_cast<int>(std::floor(x / cellSize)),
                static_cast<int>(std::floor(y / cellSize)),
                static_cast<int>(std::floor(z / cellSize))
            };

            Counts& count = cells[key];
            ++count.total;

            if (group == 0) {
                ++count.blue;
            }

            ++totalParticles;
        }

        writeFrame();

        output.close();

        if (!output) {
            throw std::runtime_error("Output write failed");
        }

        std::cout << "Saved mixing_score.csv\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }

    return 0;
}