#pragma once
#include "hzl/solver.hpp"
#include <memory>

namespace hzl {
// GPU-resident RK4 solver with optional shared host-generated midpoint forcing.
class CudaSolver {
  public:
    CudaSolver(std::size_t n, Parameters parameters, const Field &initial);
    ~CudaSolver();
    CudaSolver(const CudaSolver &) = delete;
    CudaSolver &operator=(const CudaSolver &) = delete;
    void step(double dt);
    void step(double dt, const Field &midpoint_forcing);
    void synchronize();
    [[nodiscard]] Field coefficients() const;

  private:
    void advance(double dt);
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};
} // namespace hzl
