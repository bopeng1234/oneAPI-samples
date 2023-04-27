
#ifndef __DPCPP_BUNDLE_ADJUSTMENT_H__
#define __DPCPP_BUNDLE_ADJUSTMENT_H__

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace dpba {
////////////////////////////////////////////////////////////////////////////////////
// Type alias
////////////////////////////////////////////////////////////////////////////////////

template <class T, int N>
using Array = Eigen::Matrix<T, N, 1>;

template <class T>
using Set = std::unordered_set<T>;

template <class T>
using UniquePtr = std::unique_ptr<T>;

////////////////////////////////////////////////////////////////////////////////////
// Edge
////////////////////////////////////////////////////////////////////////////////////

struct PoseVertex;
struct LandmarkVertex;

/** @brief Base edge struct.
 */
struct BaseEdge {
  /** @brief Returns the connected pose vertex.
   */
  virtual PoseVertex* poseVertex() const = 0;

  /** @brief Returns the connected landmark vertex.
   */
  virtual LandmarkVertex* landmarkVertex() const = 0;

  /** @brief Returns the dimension of measurement.
   */
  virtual int dim() const = 0;

  /** @brief the destructor.
   */
  virtual ~BaseEdge() {}
};

/** @brief Edge with N-dimensional measurement.
@tparam DIM dimension of the measurement vector.
*/
template <int DIM>
struct Edge : BaseEdge {
  using Measurement = Array<double, DIM>;
  using Information = double;

  /** @brief The constructor.
   */
  Edge()
      : measurement(Measurement()),
        information(Information()),
        vertexP(nullptr),
        vertexL(nullptr) {}

  /** @brief The constructor.
  @param m measurement vector.
  @param I information matrix.
  @param vertexP connected pose vertex.
  @param vertexL connected landmark vertex.
  */
  Edge(const Measurement& m, Information I, PoseVertex* vertexP,
       LandmarkVertex* vertexL)
      : measurement(m), information(I), vertexP(vertexP), vertexL(vertexL) {}

  /** @brief Returns the connected pose vertex.
   */
  PoseVertex* poseVertex() const override { return vertexP; }

  /** @brief Returns the connected landmark vertex.
   */
  LandmarkVertex* landmarkVertex() const override { return vertexL; }

  /** @brief Returns the dimension of measurement.
   */
  int dim() const override { return DIM; }

  Measurement measurement;  //!< measurement vector.
  Information information;  //!< information matrix (represented by a scalar for
                            //!< performance).
  PoseVertex* vertexP;      //!< connected pose vertex.
  LandmarkVertex* vertexL;  //!< connected landmark vertex.
};

/** @brief Edge with 2-dimensional measurement (monocular observation).
 */
using MonoEdge = Edge<2>;

/** @brief Edge with 3-dimensional measurement (stereo observation).
 */
using StereoEdge = Edge<3>;

////////////////////////////////////////////////////////////////////////////////////
// Vertex
////////////////////////////////////////////////////////////////////////////////////

/** @brief Pose vertex struct.
 */
struct PoseVertex {
  using Quaternion = Eigen::Quaterniond;
  using Rotation = Quaternion;
  using Translation = Array<double, 3>;

  /** @brief The constructor.
   */
  PoseVertex()
      : q(Rotation()), t(Translation()), fixed(false), id(-1), iP(-1) {}

  /** @brief The constructor.
  @param id ID of the vertex.
  @param q rotational component of the pose, represented by quaternions.
  @param t translational component of the pose.
  @param fixed if true, the state variables are fixed during optimization.
  */
  PoseVertex(int id, const Rotation& q, const Translation& t,
             bool fixed = false)
      : q(q), t(t), fixed(fixed), id(id), iP(-1) {}

  Rotation
      q;  //!< rotational component of the pose, represented by quaternions.
  Translation t;  //!< translational component of the pose.
  bool fixed;  //!< if true, the state variables are fixed during optimization.
  int id;      //!< ID of the vertex.
  int iP;      //!< ID of the vertex (internally used).
  Set<BaseEdge*> edges;  //!< connected edges.
};

/** @brief Landmark vertex struct.
 */
struct LandmarkVertex {
  using Point3D = Array<double, 3>;

  /** @brief The constructor.
   */
  LandmarkVertex() : Xw(Point3D()), fixed(false), id(-1), iL(-1) {}

  /** @brief The constructor.
  @param id ID of the vertex.
  @param Xw 3D position of the landmark.
  @param fixed if true, the state variables are fixed during optimization.
  */
  LandmarkVertex(int id, const Point3D& Xw, bool fixed = false)
      : Xw(Xw), fixed(fixed), id(id), iL(-1) {}

  Point3D Xw;  //!< 3D position of the landmark.
  bool fixed;  //!< if true, the state variables are fixed during optimization.
  int id;      //!< ID of the vertex.
  int iL;      //!< ID of the vertex (internally used).
  Set<BaseEdge*> edges;  //!< connected edges.
};

////////////////////////////////////////////////////////////////////////////////////
// Camera parameters
////////////////////////////////////////////////////////////////////////////////////

/** @brief Camera parameters struct.
 */
struct CameraParams {
  double fx;  //!< focal length x (pixel)
  double fy;  //!< focal length y (pixel)
  double cx;  //!< principal point x (pixel)
  double cy;  //!< principal point y (pixel)
  double bf;  //!< stereo baseline times fx

  /** @brief The constructor.
   */
  CameraParams() : fx(0), fy(0), cx(0), cy(0), bf(0) {}
};

////////////////////////////////////////////////////////////////////////////////////
// Statistics
////////////////////////////////////////////////////////////////////////////////////

/** @brief information about optimization.
 */
struct BatchInfo {
  int iteration;  //!< iteration number
  double chi2;    //!< total chi2 (objective function value)
};

using BatchStatistics = std::vector<BatchInfo>;

/** @brief Time profile.
 */
using TimeProfile = std::map<std::string, double>;

////////////////////////////////////////////////////////////////////////////////////
// Type alias
////////////////////////////////////////////////////////////////////////////////////

using VertexP = PoseVertex;
using VertexL = LandmarkVertex;
using Edge2D = MonoEdge;
using Edge3D = StereoEdge;

/** @brief DPCPP implementation of Bundle Adjustment.

The class implements a Bundle Adjustment algorithm with DPCPP.
It optimizes camera poses and landmarks (3D points) represented by a graph.

@attention This class doesn't take responsibility for deleting pointers to
vertices and edges added in the graph.

*/
class DpcppBundleAdjustment {
 public:
  using Ptr = UniquePtr<DpcppBundleAdjustment>;

  /** @brief Creates an instance of DpcppBundleAdjustment.
   */
  static Ptr create();

  /** @brief Adds a pose vertex to the graph.
   */
  virtual void addPoseVertex(PoseVertex* v) = 0;

  /** @brief Adds a landmark vertex to the graph.
   */
  virtual void addLandmarkVertex(LandmarkVertex* v) = 0;

  /** @brief Adds an edge with monocular observation to the graph.
   */
  virtual void addMonocularEdge(MonoEdge* e) = 0;

  /** @brief Adds an edge with stereo observation to the graph.
   */
  virtual void addStereoEdge(StereoEdge* e) = 0;

  /** @brief Returns the pose vertex with specified id.
   */
  virtual PoseVertex* poseVertex(int id) const = 0;

  /** @brief Returns the landmark vertex with specified id.
   */
  virtual LandmarkVertex* landmarkVertex(int id) const = 0;

  /** @brief Removes a pose vertex from the graph.
   */
  virtual void removePoseVertex(PoseVertex* v) = 0;

  /** @brief Removes a landmark vertex from the graph.
   */
  virtual void removeLandmarkVertex(LandmarkVertex* v) = 0;

  /** @brief Removes an edge from the graph.
   */
  virtual void removeEdge(BaseEdge* e) = 0;

  /** @brief Sets a camera parameters to the graph.
  @note The camera parameters are the same in all edges.
  */
  virtual void setCameraPrams(const CameraParams& camera) = 0;

  /** @brief Returns the number of poses in the graph.
   */
  virtual size_t nposes() const = 0;

  /** @brief Returns the number of landmarks in the graph.
   */
  virtual size_t nlandmarks() const = 0;

  /** @brief Returns the total number of edges in the graph.
   */
  virtual size_t nedges() const = 0;

  /** @brief Initializes the graph.
   */
  virtual void initialize() = 0;

  /** @brief Optimizes the graph.
  @param niterations number of iterations for Levenberg-Marquardt algorithm.
  */
  virtual void optimize(int niterations) = 0;

  /** @brief Clears the graph.
   */
  virtual void clear() = 0;

  /** @brief Returns the batch statistics.
   */
  virtual const BatchStatistics& batchStatistics() const = 0;

  /** @brief Returns the time profile.
   */
  virtual const TimeProfile& timeProfile() = 0;

  /** @brief the destructor.
   */
  virtual ~DpcppBundleAdjustment();
};

}  // namespace dpba

#endif  // !__DPCPP_BUNDLE_ADJUSTMENT_H__
