
#ifndef __DEVICE_MATRIX_H__
#define __DEVICE_MATRIX_H__

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include "macro.h"

namespace dpba {

#define HOST_DEVICE inline

template <typename T, int N>
struct Vec {
  HOST_DEVICE Vec() {}
  HOST_DEVICE Vec(const T* values) {
    for (int i = 0; i < N; i++) data[i] = values[i];
  }

  template <typename U>
  HOST_DEVICE Vec(const U* values) {
    for (int i = 0; i < N; i++) data[i] = T(values[i]);
  }

  HOST_DEVICE T& operator[](int i) { return data[i]; }
  HOST_DEVICE const T& operator[](int i) const { return data[i]; }

  HOST_DEVICE void copyTo(T* rhs) const {
    for (int i = 0; i < N; i++) rhs[i] = data[i];
  }

  template <typename U>
  HOST_DEVICE void copyTo(U* rhs) const {
    for (int i = 0; i < N; i++) rhs[i] = U(data[i]);
  }

  T data[N];
};

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() : data_(nullptr), size_(0), capacity_(0), allocated_(false) {}
  DeviceBuffer(size_t size)
      : data_(nullptr), size_(0), capacity_(0), allocated_(false) {
    resize(size);
  }
  ~DeviceBuffer() { destroy(); }

  void allocate(size_t count) try {
    if (data_ && capacity_ >= count) return;

    destroy();
    DPCPP_CHECK((data_ = (T*)sycl::malloc_shared(sizeof(T) * count,
                                                 dpct::get_default_queue()),
                 0));
    capacity_ = count;
    allocated_ = true;
  } catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  void destroy() try {
    if (allocated_ && data_)
      DPCPP_CHECK((sycl::free(data_, dpct::get_default_queue()), 0));
    data_ = nullptr;
    size_ = 0;
    allocated_ = false;
  } catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  void resize(size_t size) {
    allocate(size);
    size_ = size;
  }

  void map(size_t size, void* data) {
    data_ = (T*)data;
    size_ = size;
    allocated_ = false;
  }

  void assign(size_t size, const void* h_data) {
    resize(size);
    upload((T*)h_data);
  }

  void upload(const T* h_data) try {
    DPCPP_CHECK((dpct::get_default_queue()
                     .memcpy(data_, h_data, sizeof(T) * size_)
                     .wait(),
                 0));
  } catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  void download(T* h_data) const try {
    DPCPP_CHECK((dpct::get_default_queue()
                     .memcpy(h_data, data_, sizeof(T) * size_)
                     .wait(),
                 0));
  } catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  void copyTo(T* rhs) const try {
    DPCPP_CHECK(
        (dpct::get_default_queue().memcpy(rhs, data_, sizeof(T) * size_).wait(),
         0));
  } catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  void fillZero() try {
    DPCPP_CHECK(
        (dpct::get_default_queue().memset(data_, 0, sizeof(T) * size_).wait(),
         0));
  } catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  T* data() { return data_; }
  const T* data() const { return data_; }

  size_t size() const { return size_; }
  int ssize() const { return static_cast<int>(size_); }

  operator T*() { return data_; }
  operator const T*() const { return data_; }

 private:
  T* data_;
  size_t size_, capacity_;
  bool allocated_;
};

template <typename T, int BLOCK_ROWS, int BLOCK_COLS>
class BlockPtr {
 public:
  static const int BLOCK_AREA = BLOCK_ROWS * BLOCK_COLS;
  BlockPtr(T* data) : data_(data) {}
  T* at(int i) { return data_ + i * BLOCK_AREA; }
  const T* at(int i) const { return data_ + i * BLOCK_AREA; }

 private:
  T* data_;
};

template <typename T, int BLOCK_ROWS, int BLOCK_COLS, int ORDER>
class DeviceBlockMatrix {
 public:
  static const int BLOCK_AREA = BLOCK_ROWS * BLOCK_COLS;
  using BlockPtrT = BlockPtr<T, BLOCK_ROWS, BLOCK_COLS>;

  DeviceBlockMatrix()
      : rows_(0), cols_(0), nnz_(0), outerSize_(0), innerSize_(0) {}
  DeviceBlockMatrix(int rows, int cols) : nnz_(0) { resize(rows, cols); }

  void resize(int rows, int cols) {
    rows_ = rows;
    cols_ = cols;
    outerSize_ = ORDER == ROW_MAJOR ? rows : cols;
    innerSize_ = ORDER == ROW_MAJOR ? cols : rows;
    outerIndices_.resize(outerSize_ + 1);
  }

  void resizeNonZeros(int nnz) {
    nnz_ = nnz;
    values_.resize(nnz * BLOCK_AREA);
    innerIndices_.resize(nnz);
  }

  void mapNonZeros(int nnz, T* data) {
    nnz_ = nnz;
    values_.map(nnz * BLOCK_AREA, data);
    innerIndices_.resize(nnz);
  }

  void upload(const T* values, const int* outerIndices,
              const int* innerIndices) {
    if (values) values_.upload(values);
    if (outerIndices) outerIndices_.upload(outerIndices);
    if (innerIndices) innerIndices_.upload(innerIndices);
  }

  void download(T* values, int* outerIndices, int* innerIndices) const {
    if (values) values_.download(values);
    if (outerIndices) outerIndices_.download(outerIndices);
    if (innerIndices) innerIndices_.download(innerIndices);
  }

  void fillZero() { values_.fillZero(); }

  T* values() { return values_.data(); }
  int* outerIndices() { return outerIndices_.data(); }
  int* innerIndices() { return innerIndices_.data(); }

  const T* values() const { return values_.data(); }
  const int* outerIndices() const { return outerIndices_.data(); }
  const int* innerIndices() const { return innerIndices_.data(); }

  int rows() const { return rows_; }
  int cols() const { return cols_; }
  int nnz() const { return nnz_; }

  operator BlockPtrT() const { return BlockPtrT((T*)values_.data()); }

 private:
  DeviceBuffer<T> values_;
  DeviceBuffer<int> outerIndices_, innerIndices_;
  int rows_, cols_, nnz_, outerSize_, innerSize_;
};

template <typename T, int BLOCK_ROWS, int BLOCK_COLS>
class DeviceBlockVector {
 public:
  static const int BLOCK_AREA = BLOCK_ROWS * BLOCK_COLS;
  using BlockPtrT = BlockPtr<T, BLOCK_ROWS, BLOCK_COLS>;

  DeviceBlockVector() : size_(0) {}
  DeviceBlockVector(int size) { resize(size); }

  void resize(int size) {
    size_ = size;
    values_.resize(size * BLOCK_AREA);
  }

  void map(int size, T* data) {
    size_ = size;
    values_.map(size * BLOCK_AREA, data);
  }

  void fillZero() { values_.fillZero(); }

  void copyTo(DeviceBlockVector& rhs) const { values_.copyTo(rhs.values()); }

  T* values() { return values_.data(); }
  const T* values() const { return values_.data(); }

  int size() const { return size_; }
  int elemSize() const { return size_ * BLOCK_AREA; }

  operator BlockPtrT() const { return BlockPtrT((T*)values_.data()); }

 private:
  DeviceBuffer<T> values_;
  int size_;
};

using Vec2d = Vec<Scalar, 2>;
using Vec3d = Vec<Scalar, 3>;
using Vec4d = Vec<Scalar, 4>;

using Vec2i = Vec<int, 2>;
using Vec3i = Vec<int, 3>;
using Vec4i = Vec<int, 4>;

template <typename T>
using GpuVec = DeviceBuffer<T>;

using GpuVec1d = GpuVec<Scalar>;
using GpuVec2d = GpuVec<Vec2d>;
using GpuVec3d = GpuVec<Vec3d>;
using GpuVec4d = GpuVec<Vec4d>;

using GpuVec1i = GpuVec<int>;
using GpuVec2i = GpuVec<Vec2i>;
using GpuVec3i = GpuVec<Vec3i>;
using GpuVec4i = GpuVec<Vec4i>;

using GpuVec1b = GpuVec<uint8_t>;

using GpuHplBlockMat = DeviceBlockMatrix<Scalar, PDIM, LDIM, COL_MAJOR>;
using GpuHscBlockMat = DeviceBlockMatrix<Scalar, PDIM, PDIM, ROW_MAJOR>;

using GpuPxPBlockVec = DeviceBlockVector<Scalar, PDIM, PDIM>;
using GpuLxLBlockVec = DeviceBlockVector<Scalar, LDIM, LDIM>;
using GpuPxLBlockVec = DeviceBlockVector<Scalar, PDIM, LDIM>;
using GpuPx1BlockVec = DeviceBlockVector<Scalar, PDIM, 1>;
using GpuLx1BlockVec = DeviceBlockVector<Scalar, LDIM, 1>;

}  // namespace dpba

#endif  // !__DEVICE_MATRIX_H__
