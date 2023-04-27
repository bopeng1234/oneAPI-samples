
#ifndef __DPCPP_LINEAR_SOLVER_H__
#define __DPCPP_LINEAR_SOLVER_H__

#include <memory>

#include "device_matrix.h"
#include "macro.h"
#include "sparse_block_matrix.h"

namespace dpba {

class SparseLinearSolver {
 public:
  using Ptr = std::unique_ptr<SparseLinearSolver>;
  static Ptr create();

  virtual void initialize(const GpuHplBlockMat& Hpl) = 0;
  virtual bool solve(const Scalar* d_H, Scalar* d_b, Scalar* d_x) = 0;

  virtual ~SparseLinearSolver();
};

}  // namespace dpba

#endif  // !__DPCPP_LINEAR_SOLVER_H__
