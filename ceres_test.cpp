#include <ceres/ceres.h>
#include <iostream>

// Cost functor: y = x^2, we want to minimize x^2
struct SquareCost {
    template <typename T>
    bool operator()(const T* const x, T* residual) const {
        residual[0] = x[0] * x[0];
        return true;
    }
};

int main() {
    double x = 5.0;  // initial guess

    ceres::Problem problem;
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<SquareCost, 1, 1>(new SquareCost),
        nullptr,
        &x
    );

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << "Initial x = 5.0, optimized x = " << x << std::endl;
    std::cout << summary.BriefReport() << std::endl;

    return 0;
}
