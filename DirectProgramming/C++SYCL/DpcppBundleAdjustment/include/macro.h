
#ifndef __MACRO_H__
#define __MACRO_H__

#include <CL/sycl.hpp>
#include <cstdio>
#include <dpct/dpct.hpp>
#include <type_traits>
#include <vector>

#define DPCPP_CHECK(err)                                                       \
  do {                                                                         \
    if (err != 0) {                                                            \
      printf("[DPCPP Error] %s (code: %d) at %s:%d\n",                         \
             "dpcppGetErrorString not supported" /*dpcppGetErrorString(err)*/, \
             err, __FILE__, __LINE__);                                         \
    }                                                                          \
  } while (0)

namespace dpba {

// Constant

static constexpr int PDIM = 6;
static constexpr int LDIM = 3;

enum StorageOrder { ROW_MAJOR, COL_MAJOR };

enum EdgeFlag { EDGE_FLAG_FIXED_L = 1, EDGE_FLAG_FIXED_P = 2 };

// Create Object
struct BaseObject {
  virtual ~BaseObject() {}
};

template <class T>
struct Object : public BaseObject {
  Object(T* ptr = nullptr) : ptr(ptr) {}
  ~Object() { delete ptr; }
  T* ptr;
};

class ObjectCreator {
 public:
  ObjectCreator() {}
  ~ObjectCreator() { release(); }

  template <class T, class... Args>
  T* create(Args&&... args) {
    T* ptr = new T(std::forward<Args>(args)...);
    objects_.push_back(new Object(ptr));
    return ptr;
  }

  void release() {
    for (auto ptr : objects_) delete ptr;
    objects_.clear();
  }

 private:
  std::vector<BaseObject*> objects_;
};

// Define Scalar
using Scalar = float;

static_assert(std::is_same<Scalar, float>::value ||
                  std::is_same<Scalar, double>::value,
              "Scalar must be float or double.");

}  // namespace dpba

#endif  // !__MACRO_H__
