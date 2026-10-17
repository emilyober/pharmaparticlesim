#pragma once

struct JKRParameters {
    double effectiveYoungModulus{0.0};
    double effectiveRadius{0.0};
    double surfaceEnergy{0.0};
};

struct JKRFullResult {
    bool active{false};
    double contactRadius{0.0};
    double elasticNormalForce{0.0};
};

struct JKRFull {
    public:
        static JKRFullResult evaluate(
            double overlap,
            bool previouslyActive,
            const JKRParameters& parameters
        );
};