// Times the inner loop of QuadraticModel::get_effective_linear_bias
// (dwave/optimization/src/nodes/quadratic_model.cpp in dwave-optimization)
// with different accumulator types. That function sums in `long double`,
// which is an 80-bit hardware type on x86-64 and a 128-bit software type on
// Linux aarch64.
//
// Build and run: g++ -std=c++17 -O3 -o bench long_double_bench.cpp && ./bench

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

// The compensated sum of dwave-optimization's src/kahan.hpp, reduced to what the loop needs.
struct kahan {
    kahan(double value) : value_(value), compensator_(0) {}
    kahan& operator+=(double other) {
        double y = other - compensator_;
        double t = value_ + y;
        compensator_ = (t - value_) - y;
        value_ = t;
        return *this;
    }
    explicit operator double() const { return value_; }
    double value_;
    double compensator_;
};

// One variable's interactions, laid out as in QuadraticModel::neighborhood.
struct Neighborhood {
    std::vector<int> neighbors;
    std::vector<double> biases;
};

// The loop of get_effective_linear_bias, with the accumulator type as a parameter.
template <class Accumulator>
double effective_linear_bias(const Neighborhood& adj, double linear, const std::vector<double>& state) {
    Accumulator effective_linear_bias(linear);
    for (std::size_t j = 0; j < adj.neighbors.size(); j++) {
        effective_linear_bias += state[adj.neighbors[j]] * adj.biases[j];
    }
    return static_cast<double>(effective_linear_bias);
}

// Nanoseconds per call, best of five repetitions, for random variables of a model in which
// every variable has `degree` interactions.
template <class Accumulator>
double time_calls(int num_variables, int degree, long calls, double& checksum) {
    std::mt19937_64 rng(1);
    auto bias = [&rng]() { return static_cast<double>(rng() % 2001) / 1000.0 - 1.0; };

    std::vector<double> linear(num_variables);
    std::vector<Neighborhood> adj(num_variables);
    std::vector<double> state(num_variables);
    for (int i = 0; i < num_variables; ++i) {
        linear[i] = bias();
        state[i] = static_cast<double>(rng() % 2);
        for (int k = 1; k <= degree / 2; ++k) {
            int j = (i + k) % num_variables;
            double b = bias();
            adj[i].neighbors.push_back(j);
            adj[i].biases.push_back(b);
            adj[j].neighbors.push_back(i);
            adj[j].biases.push_back(b);
        }
    }

    double best = 1e300;
    for (int repetition = 0; repetition < 5; ++repetition) {
        double sum = 0;
        auto start = std::chrono::steady_clock::now();
        for (long call = 0; call < calls; ++call) {
            int u = static_cast<int>(rng() % num_variables);
            sum += effective_linear_bias<Accumulator>(adj[u], linear[u], state);
            state[u] = 1.0 - state[u];  // flip the variable, as a move would
        }
        std::chrono::duration<double, std::nano> elapsed = std::chrono::steady_clock::now() - start;
        best = std::min(best, elapsed.count() / static_cast<double>(calls));
        checksum += sum;
    }
    return best;
}

int main() {
    std::printf("sizeof(long double) = %zu bytes, mantissa = %d bits (double: %d bits)\n\n",
                sizeof(long double), LDBL_MANT_DIG, DBL_MANT_DIG);
#if defined(__x86_64__)
    std::printf("%12s %14s %10s %10s %14s   (ns per call)\n", "interactions", "long double", "double",
                "kahan", "__float128");
#else
    std::printf("%12s %14s %10s %10s   (ns per call)\n", "interactions", "long double", "double", "kahan");
#endif

    const int num_variables = 2048;
    double checksum = 0;
    for (int degree : {4, 16, 64, 256, 1024}) {
        const long calls = 20000000L / (degree + 16);
        double t_long_double = time_calls<long double>(num_variables, degree, calls, checksum);
        double t_double = time_calls<double>(num_variables, degree, calls, checksum);
        double t_kahan = time_calls<kahan>(num_variables, degree, calls, checksum);
#if defined(__x86_64__)
        // software 128-bit on x86-64, for comparison with long double on aarch64
        double t_float128 = time_calls<__float128>(num_variables, degree, calls / 8, checksum);
        std::printf("%12d %14.1f %10.1f %10.1f %14.1f   long double / double = %.1fx\n", degree, t_long_double,
                    t_double, t_kahan, t_float128, t_long_double / t_double);
#else
        std::printf("%12d %14.1f %10.1f %10.1f   long double / double = %.1fx\n", degree, t_long_double, t_double,
                    t_kahan, t_long_double / t_double);
#endif
    }
    std::printf("\nchecksum %.6g\n", checksum);
}
