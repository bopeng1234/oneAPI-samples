
#include "dpcpp_linear_solver.h"

#include <CL/sycl.hpp>
#include <algorithm>
#include <dpct/dpct.hpp>
#include <iostream>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <tuple>
#include <vector>
// #include <execution>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <dpct/blas_utils.hpp>
#include <dpct/dpl_utils.hpp>
#include <oneapi/mkl.hpp>

#include "device_matrix.h"
#include "dpcpp_block_solver.h"

// Define the format to printf MKL_INT values
#if !defined(MKL_ILP64)
#define IFORMAT "%i"
#else
#define IFORMAT "%lli"
#endif

namespace dpba {
using namespace std;

struct H_Tri_value {
  int row, col, id;
};

struct LessRowId {
  bool operator()(const Vec3i& lhs, const Vec3i& rhs) const {
    if (lhs[0] == rhs[0]) return lhs[1] < rhs[1];
    return lhs[0] < rhs[0];
  }
};

bool sortbyrow(const H_Tri_value& a, const H_Tri_value& b) {
  if (a.row == b.row) {
    return (a.col < b.col);
  }
  return (a.row < b.row);
}

template <class I, class T>
void csc_to_csr(const I n_row, const I n_col, const I Ap[], const I Aj[],
                const T Ax[], I Bp[], I Bi[], T Bx[]) {
  const I nnz = Ap[n_row];

  // compute number of non-zero entries per column of A
  std::fill(Bp, Bp + n_col, 0);

  for (I n = 0; n < nnz; n++) {
    Bp[Aj[n]]++;
  }

  // cumsum the nnz per column to get Bp[]
  for (I col = 0, cumsum = 0; col < n_col; col++) {
    I temp = Bp[col];
    Bp[col] = cumsum;
    cumsum += temp;
  }
  Bp[n_col] = nnz;

  for (I row = 0; row < n_row; row++) {
    for (I jj = Ap[row]; jj < Ap[row + 1]; jj++) {
      I col = Aj[jj];
      I dest = Bp[col];

      Bi[dest] = row;
      Bx[dest] = Ax[jj];

      Bp[col]++;
    }
  }

  for (I col = 0, last = 0; col <= n_col; col++) {
    I temp = Bp[col];
    Bp[col] = last;
    last = temp;
  }
}

class SparseLinearSolverImpl : public SparseLinearSolver {
 public:
  using SparseMatrixCSR = Eigen::SparseMatrix<Scalar, Eigen::RowMajor>;
  using PermutationMatrix =
      Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>;
  // using Cholesky = CuSparseCholeskySolver<Scalar>;

  int size;
  int nnz;
  int nnz_new;

  int n;

  MKL_INT mtype;
  MKL_INT nrhs;
  /* Internal solver memory pointer pt, */
  /* 32-bit: int pt[64]; 64-bit: long int pt[64] */
  /* or void *pt[64] should be OK on both architectures */
  void* pt[64];
  /* Pardiso control parameters. */
  MKL_INT iparm[64];
  MKL_INT maxfct, mnum, phase, error, msglvl, info;
  /* Auxiliary variables. */
  MKL_INT i;
  double ddum;  /* Double dummy */
  MKL_INT idum; /* Integer dummy. */
  MKL_INT n_schur;

  /* LAPACKE parameters. */
  int matrix_order = LAPACK_ROW_MAJOR;
  char uplo = 'U';

  vector<int> ia;
  vector<int> ja;
  vector<float> a;
  vector<int> colInd_triangular_map;
  MKL_INT* perm = nullptr;
  MKL_INT* ipiv = nullptr;
  float* schur = nullptr;
  Scalar* d_b_temp = nullptr;
  vector<int> whole_triangular_map;

  int* ia_ptr;
  int* ja_ptr;
  float* a_ptr;

  void initialize(const GpuHplBlockMat& Hpl) override {
    #ifdef PRINTTIME
    auto t0 = std::chrono::steady_clock::now();
    #endif
    using namespace std;
    int Hpp_blocks = Hpl.rows();
    int Hll_blocks = Hpl.cols();

    int Hpl_blocks = Hpl.nnz();
    const int* Hpl_rowIdx = Hpl.innerIndices();
    const int* Hpl_colIdx = Hpl.outerIndices();

    size = Hpp_blocks * PDIM + Hll_blocks * LDIM;

    n_schur = Hpp_blocks * PDIM; /* Schur complement solution size */
    if (schur == nullptr) {
      schur = (float*)malloc(n_schur * n_schur * sizeof(float));
    }
    if (perm == nullptr) {
      perm = (MKL_INT*)malloc(size * sizeof(MKL_INT));
    }
    for (i = 0; i < size; i++) {
      if (i < n_schur)
        perm[i] = 1;
      else
        perm[i] = 0;
    }
    if (ipiv == nullptr) {
      ipiv = (MKL_INT*)malloc(n_schur * sizeof(MKL_INT));
    }

    // nnz = Hpp_blocks * PDIM * PDIM + Hll_blocks * LDIM * LDIM +

    // size = Hsc.rows();
    // nnz = Hsc.nnzSymm();

    n = size;
    if (d_b_temp == nullptr) {
      d_b_temp = (Scalar*)malloc(n * sizeof(Scalar));
    }

    mtype = 2; /* Real and symmetric positive definite */
    nrhs = 1;  /* Number of right hand sides. */

    /* -------------------------------------*/
    /* .. Setup Pardiso control parameters. */
    /* -------------------------------------*/
    for (i = 0; i < 64; i++) {
      iparm[i] = 0;
    }
    iparm[0] = 1;  /* No solver default */
    iparm[1] = 2;  /* Fill-in reordering from METIS */
    iparm[3] = 0;  /* No iterative-direct algorithm */
    iparm[4] = 0;  /* No user fill-in reducing permutation */
    iparm[5] = 0;  /* Write solution into x */
    iparm[7] = 2;  /* Max numbers of iterative refinement steps */
    iparm[9] = 13; /* Perturb the pivot elements with 1E-13 */
    iparm[10] = 0;
        /* 0: Disable scaling. Default for symmetric indefinite matrices. */ /* Use nonsymmetric permutation and scaling MPS */
    iparm[12] =
        0; /* Maximum weighted matching algorithm is switched-off (default for
              symmetric). Try iparm[12] = 1 in case of inappropriate accuracy */
    iparm[13] = 0;  /* Output: Number of perturbed pivots */
    iparm[17] = -1; /* Output: Number of nonzeros in the factor LU */
    iparm[18] = -1; /* Output: Mflops for LU factorization */
    iparm[19] = 0;  /* Output: Numbers of CG Iterations */
    iparm[26] = 0;  /* !!!!!!!!!!!!!!!!!!!!!!!!! Matrix checker. checks integer
                       arrays ia and ja. whether column indices are sorted in
                       increasing order within each row. */
    iparm[27] = 1;  /* 1 means Input arrays (a, x and b) must be presented in
                       single precision(float). 0 means double */
    iparm[34] = 1;  /* PARDISO use C-style indexing for ia and ja arrays */
    iparm[35] = 1;  /* Use Schur complement */
    maxfct = 1;     /* Maximum number of numerical factorizations. */
    mnum = 1;       /* Which factorization to use. */
    msglvl =
        0; /* !!!!!!!!!!!!!!!!!!!!!!!!! Print statistical information in file */
    error = 0; /* Initialize error flag */
    #ifdef PRINTTIME
    auto t1 = std::chrono::steady_clock::now();
    
    cout << "t1-t0 = " <<
    std::chrono::duration_cast<std::chrono::duration<double>>(t1 -
    t0).count() * 1e3 << "   , Initial malloc" << endl;
    #endif

    // /* starting computation */
    // const auto t0 = get_time_point();

    vector<H_Tri_value> H_Tri;
    // forth is flag, 0 for hpp, 1 for hll, 2 for hpl

    vector<int> Hpl_valueIdx;
    Hpl_valueIdx.resize(Hpl_blocks);
    for (int data_index = 0; data_index < Hpl_blocks; data_index++)
      Hpl_valueIdx[data_index] = data_index;

    vector<int> Hpl_csr_rowIdx;
    Hpl_csr_rowIdx.resize(Hpp_blocks);
    vector<int> Hpl_csr_colIdx;
    Hpl_csr_colIdx.resize(Hpl_blocks);
    vector<int> Hpl_csr_valueIdx;
    Hpl_csr_valueIdx.resize(Hpl_blocks);
    csc_to_csr<int, int>(Hll_blocks, Hpp_blocks, Hpl_colIdx, Hpl_rowIdx,
                         Hpl_valueIdx.data(), Hpl_csr_rowIdx.data(),
                         Hpl_csr_colIdx.data(), Hpl_csr_valueIdx.data());

    // Hpp
    nnz_new = 0;
    for (int block_idx = 0; block_idx < Hpp_blocks; block_idx++) {
      // loop the upper triangular value, and push into matrix
      for (int local_y = 0; local_y < PDIM; local_y++) {
        int block_x = block_idx * PDIM;
        int block_y = block_idx * PDIM;
        for (int local_x = local_y; local_x < PDIM; local_x++) {
          int global_y = block_y + local_y;
          int global_x = block_x + local_x;

          int data_index = block_idx * PDIM * PDIM + local_y * PDIM + local_x;
          int value = data_index;

          H_Tri.push_back({global_y, global_x, value});
          nnz_new++;
        }
        for (int col_idx = Hpl_csr_rowIdx[block_idx];
             col_idx < Hpl_csr_rowIdx[block_idx + 1]; col_idx++) {
          for (int local_x = 0; local_x < LDIM; local_x++) {
            int block_x = Hpl_csr_colIdx[col_idx] * LDIM;
            int block_y = block_idx * PDIM;

            int global_y = block_y + local_y;
            int global_x =
                block_x + local_x +
                Hpp_blocks * PDIM;  // start from right-upper rectangle

            int data_index =
                Hpl_csr_valueIdx[col_idx] * PDIM * LDIM + local_y +
                local_x * PDIM;  //////////// IMPORTANT !!!!!!!!!!!!!!!!!!  Hpl
                                 ///each block use col major save data
            int value = data_index + Hpp_blocks * PDIM * PDIM +
                        Hll_blocks * LDIM * LDIM;

            H_Tri.push_back({global_y, global_x, value});
            nnz_new++;
          }
        }
      }
    }

    // Hll
    for (int block_idx = 0; block_idx < Hll_blocks; block_idx++) {
      int block_x = block_idx * LDIM;
      int block_y = block_idx * LDIM;

      // loop the upper triangular value, and push into matrix
      for (int local_y = 0; local_y < LDIM; local_y++) {
        for (int local_x = local_y; local_x < LDIM; local_x++) {
          int global_y = block_y + local_y +
                         Hpp_blocks * PDIM;  // start from right-down rectangle
          int global_x = block_x + local_x +
                         Hpp_blocks * PDIM;  // start from right-down rectangle

          int data_index = block_idx * LDIM * LDIM + local_y * LDIM + local_x;
          int value = data_index + Hpp_blocks * PDIM * PDIM;

          H_Tri.push_back({global_y, global_x, value});
          nnz_new++;
        }
      }
    }
    #ifdef PRINTTIME
    auto t2 = std::chrono::steady_clock::now();
    cout << "t2-t1 = " <<
    std::chrono::duration_cast<std::chrono::duration<double>>(t2 -
    t1).count() * 1e3 << "   , Construct H_Tri matrix" << endl;
    #endif

    ia.clear();
    ja.clear();
    whole_triangular_map.clear();
    ia.push_back(0);
    int current_row = 0;
    int data_index = 0;
    for (; data_index < H_Tri.size(); data_index++) {
      int row = H_Tri[data_index].row;
      int col = H_Tri[data_index].col;
      int index = H_Tri[data_index].id;

      // CSR_H_value.push_back(val);
      whole_triangular_map.push_back(index);
      ja.push_back(col);

      if (row != current_row) {
        ia.push_back(data_index);
        current_row = row;
      }
    }
    ia.push_back(data_index);
    #ifdef PRINTTIME
    auto t4 = std::chrono::steady_clock::now();
    cout << "t4-t3 = " <<
    std::chrono::duration_cast<std::chrono::duration<double>>(t4 -
    t2).count() * 1e3 << "   , Compress to CSR(ia,ja,index_map)" << endl;
    #endif

    a.resize(nnz_new);
    a_ptr = a.data();
    #ifdef PRINTTIME
    auto t5 = std::chrono::steady_clock::now();
    cout << "t5-t4 = " <<
    std::chrono::duration_cast<std::chrono::duration<double>>(t5 -
    t4).count() * 1e3 << "   , a.resize()" << endl;
    #endif

    /* ----------------------------------------------------------------*/
    /* .. Initialize the internal solver memory pointer. This is only  */
    /*   necessary for the FIRST call of the PARDISO solver.           */
    /* ----------------------------------------------------------------*/
    for (i = 0; i < 64; i++) {
      pt[i] = 0;
    }
    /* --------------------------------------------------------------------*/
    /* .. Reordering and Symbolic Factorization. This step also allocates  */
    /*    all memory that is necessary for the factorization.              */
    /* --------------------------------------------------------------------*/
    ia_ptr = ia.data();
    ja_ptr = ja.data();
    phase = 11;
    #ifdef PRINTTIME
    auto t6 = std::chrono::steady_clock::now();
    cout << "t6-t5 = " <<
    std::chrono::duration_cast<std::chrono::duration<double>>(t6 -
    t5).count() * 1e3 << "   , ia.data(), ja.data()" << endl;
    #endif

    PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &n, nullptr, ia_ptr, ja_ptr,
            perm, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
    if (error != 0) {
      printf("\nERROR during symbolic factorization: " IFORMAT, error);
      exit(1);
    }
    // printf ("\nReordering completed ... ");
    // printf ("\nNumber of nonzeros in factors = " IFORMAT, iparm[17]);
    // printf ("\nNumber of factorization MFLOPS = " IFORMAT, iparm[18]);
    #ifdef PRINTTIME
    auto t7 = std::chrono::steady_clock::now();
    cout << "t7-t6 = " <<
    std::chrono::duration_cast<std::chrono::duration<double>>(t7 -
    t6).count() * 1e3 << "   , PARDISO(phase11, symbolic factorization)" <<
    endl;
    #endif
  }

  bool solve(const Scalar* d_H, Scalar* d_b, Scalar* d_x) override {
    // cout << endl << "solver(Hx=b)
    // ================================================== " << endl;
    #ifdef PRINTTIME
    auto tt0 = std::chrono::steady_clock::now();
    #endif
    for (int i = 0; i < nnz_new; i++) {
      a[i] = d_H[whole_triangular_map[i]];
    }
    #ifdef PRINTTIME
    auto tt1 = std::chrono::steady_clock::now();
    cout << "tt1-tt0 = " <<
    std::chrono::duration_cast<std::chrono::duration<double>>(tt1 -
    tt0).count() * 1e3 << "   , A value read from Hpp,Hll,Hpl" << endl;
    #endif

    /* ----------------------------*/
    /* .. Numerical factorization. */
    /* ----------------------------*/
    phase = 22;
    PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &n, a_ptr, ia_ptr, ja_ptr, perm,
            &nrhs, iparm, &msglvl, &ddum, schur, &error);
    if (error != 0) {
      printf("\nERROR during numerical factorization: " IFORMAT, error);
      exit(2);
    }
    // printf ("\nFactorization completed ... ");
    #ifdef PRINTTIME
    auto tt2 = std::chrono::steady_clock::now();
    cout << "tt2-tt1 = " <<
    std::chrono::duration_cast<std::chrono::duration<double>>(tt2 -
    tt1).count() * 1e3 << "   , PARDISO(phase 22, numerical factorization)"
    << endl;
    #endif

    /* ----------------------------------------------- */
    /* .. Reduce solving phase.                        */
    /* ----------------------------------------------- */
    phase = 331;
    // iparm[7] = 2;         /* Max numbers of iterative refinement steps. */
    /* Set right hand side to one. */
    PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &n, a_ptr, ia_ptr, ja_ptr, perm,
            &nrhs, iparm, &msglvl, d_b, d_x, &error);
    if (error != 0) {
      printf("\nERROR during solution phase 331: " IFORMAT, error);
      exit(331);
    }
    #ifdef PRINTTIME
    auto tt3 = std::chrono::steady_clock::now();
    cout << "tt3-tt2 = " <<
    std::chrono::duration_cast<std::chrono::duration<double>>(tt3 -
    tt2).count() * 1e3 << "   , PARDISO(phase 331, solution phase)" << endl;
    #endif

    for (i = 0; i < n; i++) {
      d_b_temp[i] = d_x[i];
    }
    // printf ("\nSolve phase 331 completed ... ");
    #ifdef PRINTTIME
    auto tt4 = std::chrono::steady_clock::now();
    cout << "tt4-tt3 = " <<
    std::chrono::duration_cast<std::chrono::duration<double>>(tt4 -
    tt3).count() * 1e3 << "  , d_b_temp[i] = d_x[i]" << endl;
    #endif

    /* -------------------------------------------------------------------- */
    /* .. Solving Schur complement. */
    /* -------------------------------------------------------------------- */
    for (i = 0; i < n_schur; i++) ipiv[i] = 0;
    info = LAPACKE_ssytrf(matrix_order, uplo, n_schur, schur, n_schur,
                          ipiv);  // LAPACKE_ssytrf ?
    if (info != 0) {
      printf("info after LAPACKE_dsytrf = " IFORMAT " \n", info);
      exit(0);
    }
    info = LAPACKE_ssytrs(matrix_order, uplo, n_schur, nrhs, schur, n_schur,
                          ipiv, &d_b_temp[0], nrhs);  // LAPACKE_ssytrs
    if (info != 0) {
      printf("info after LAPACKE_dsytrs = " IFORMAT " \n", info);
      exit(0);
    }
    #ifdef PRINTTIME
    auto tt5 = std::chrono::steady_clock::now();
    cout << "tt5-tt4 = " <<
    std::chrono::duration_cast<std::chrono::duration<double>>(tt5 -
    tt4).count() * 1e3 << "   , LAPACKE_ssytrf(Solving Schur complement)" <<
    endl;
    #endif
    /* -------------------------------------------------------------------- */
    /* .. Expansion solving phase. */
    /* -------------------------------------------------------------------- */
    phase = 333;
    /* Set right hand side to x one. */
    PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &n, a_ptr, ia_ptr, ja_ptr, perm,
            &nrhs, iparm, &msglvl, d_b_temp, d_x, &error);
    if (error != 0) {
      printf("\nERROR during solution: " IFORMAT, error);
      exit(333);
    }
    // printf ("\nSolve phase 333 completed ... ");
    // printf ("\nSolve completed ... ");
    #ifdef PRINTTIME
    auto tt6 = std::chrono::steady_clock::now();
    cout << "tt6-tt5 = " <<
    std::chrono::duration_cast<std::chrono::duration<double>>(tt6 -
    tt5).count() * 1e3 << "   , PARDISO(phase 333, during solution)" << endl;
    #endif

    return true;
  }

  ~SparseLinearSolverImpl() {
    free(schur);
    free(perm);
    free(ipiv);
    free(d_b_temp);
  }

 private:
  std::vector<int> P_;
  // Cholesky cholesky_;
};

SparseLinearSolver::Ptr SparseLinearSolver::create() {
  return std::make_unique<SparseLinearSolverImpl>();
}

SparseLinearSolver::~SparseLinearSolver() {}

}  // namespace dpba