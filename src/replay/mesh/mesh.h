#pragma once

#include "libavutil/spherical.h"
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <Eigen/Core>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "replay/third_party/theia/sfm/types.h"
#include "replay/util/types.h"

namespace replay {

// This mesh class provides a simple interface to a mesh structure. The OpenMesh
// library is used under the hood to maintain an efficient half-edge data
// structure. This provides efficient access for adjacency in the mesh and a
// handful of other useful functionalities.
//
// NOTE: One peculiarity of this class is that deleted elements are not actually
// deleted until CollectGarbage() is called. Delete elements with care.
class Mesh {
public:
  Mesh();

  // Methods to load/save the mesh to/from disk. The file type is deduced from
  // the extention of the provided filename. Texture file defines the filename
  // For details on supported formats, see io/read_ply_file.h and
  // io/write_ply_file.h
  bool Load(const std::string &mesh_file);
  void Save(const std::string &output_file);

  // Append all the information of another mesh into this one. Does not modify
  // the other mesh.
  void Append(const Mesh &rhs_mesh);

  int NumVertices() const;
  int NumTriangleFaces() const;
  bool HasUVs() const;

  // Vertex getter/setters.
  VertexId AddVertex(const Eigen::Vector3f &vertex);

  // This marks the vertex for deletion. The vertex data will still live in the
  // mesh (so that vertex ids, etc. are not destroyed) but no triangle/edge data
  // will point to the vertex. The vertex will be truly deleted when
  // CollectGarbage() is called;
  bool RemoveVertex(const VertexId vertex_id);

  // Removes all isolated vertices in the mesh. Returns the number of vertices
  // removed.
  int RemoveIsolatedVertices();

  // Set/get the vertex positions.
  void SetVertexPosition(const VertexId vertex_id,
                         const Eigen::Vector3f &position);
  Eigen::Vector3f VertexPosition(const VertexId vertex_id) const;

  // Returns the valence (also known as "degree") of the vertex.
  int ValenceOfVertex(const VertexId vertex_id) const;

  // Returns true if the vertex/face lies on the boundary of the mesh.
  bool IsVertexOnBoundary(const VertexId vertex_id) const;
  bool IsFaceOnBoundary(const TriangleFaceId face_id) const;

  // Triangle face getter/setters. The vertex order is assumed to be clockwise
  // (this matters for back-face culling).
  TriangleFaceId AddTriangleFace(const VertexId vertex1, const VertexId vertex2,
                                 const VertexId vertex3);

  // This marks the triangle for deletion. The triangle data will still live in
  // the mesh (so that triangle ids, etc. are not destroyed) but no verte/edge
  // data will point to the triangl. The triangle will be truly deleted when
  // CollectGarbage() is called;
  bool RemoveTriangleFace(const TriangleFaceId face_id);

  // Cleans up the mesh and removes any vertices or faces marked for deletion.
  void CollectGarbage();

  // Similar to above, the mesh face ids are stored in continuous memory in an
  // implementation-specific format. We return a pointer to the continuous block
  // of 3 vertex ids that the user may
  Eigen::Matrix<VertexId, 3, 1>
  GetVertexIdsForTriangleFace(const TriangleFaceId face_id) const;

  // Returns the median edge length of all edges.
  float MedianEdgeLength() const;
  // Returns the mean edge length.
  float MeanEdgeLength() const;

  // Compute the opposite edge of a vertex on a face.
  Eigen::Vector3f ComputeOppositeEdge(const TriangleFaceId face_id,
                                      const VertexId vertex_id) const;

  // Computes the normal of the given triangle face.
  Eigen::Vector3f ComputeFaceNormal(const TriangleFaceId face_id) const;

  // Computes the normal of the vertex from the 1-ring neighborhood of the
  // vertex.
  Eigen::Vector3f ComputeVertexNormal(const VertexId vertex_id) const;

  // Computes the normals for all triangles and stores the result in a map.
  std::unordered_map<TriangleFaceId, Eigen::Vector3f>
  ComputeAllFaceNormals() const;

  // Computes the normal of all vertices from the 1-ring neighborhood of each
  // vertex.
  std::unordered_map<VertexId, Eigen::Vector3f> ComputeAllVertexNormals() const;

  // Computes the face centroid.
  Eigen::Vector3f ComputeFaceCentroid(const TriangleFaceId face_id) const;

  // Computes the centroid of the 1-ring around the given vertex.
  Eigen::Vector3f ComputeVertexCentroid(const VertexId vertex_id) const;

  // Computes the areas for all triangles and stores the result in a map.
  std::unordered_map<TriangleFaceId, float> ComputeAllFaceAreas() const;

  // Return all triangles that contain this vertex.
  std::unordered_set<TriangleFaceId>
  TrianglesAtVertex(const VertexId vertex_id) const;

  // Return all vertices that are contain edges to the given vertex
  std::unordered_set<VertexId> EdgesToVertex(const VertexId vertex_id) const;

  // Return all neighbors of the triangle face.
  std::unordered_set<TriangleFaceId>
  NeighborsOfTriangle(const TriangleFaceId face_id) const;

  // Texture coordinates for vertices
  void SetVertexUV(const VertexId &vertex, const float &u, const float &v);
  void SetVertexUV(const VertexId &vertex, const Eigen::Vector2f &uv);

  Eigen::Vector2f VertexUV(const VertexId &vertex) const;

  // Applies a 4x4 transformation matrix to all points in the mesh
  void ApplyTransform(const Eigen::Matrix4f &transform);

  // Flips the normals of all the faces in the mesh (i.e. from CW to CCW and
  // vice versa)
  void FlipFaceOrientation();

  void SubdivideTriangle(const TriangleFaceId triangle);

  // Decimates the mesh util the mesh is reduced to the desired ratio of
  // edges. For instance, if desired_ratio = 0.2, the decimation stops when the
  // number of edges in the mesh reaches 0.2 * the original number of
  // edges. Returns true if any edges were removed and false otherwise.
  bool DecimateMesh(const double desired_ratio);

  bool GetSubMesh(const std::unordered_set<TriangleFaceId> &faces,
                  Mesh *new_mesh) const;

  // Split the mesh into submeshes where each sub-mesh has at most
  // max_num_vertices_per_mesh vertices.
  bool SplitMeshByMaxVertexCount(const int max_num_vertices_per_mesh,
                                 std::vector<Mesh> *meshes);

  // Returns a pointer to the mesh vertex positions. This is mainly for use with
  // OpenGL.
  const float *vertex_positions() const;

  // Returns all UV coordinates. Also for use with OpenGL.
  const float *uvs() const;

  // Returns all triangles by their vertex ids.
  std::vector<Eigen::Matrix<VertexId, 3, 1>> triangles() const;

  // Comments in string form, if loaded from PLY
  std::vector<std::string> comments;

private:
  bool has_uvs_;
  // TODO(csweeney): Need to do more research to understand if exact predicates
  // are needed here.
  typedef CGAL::Simple_cartesian<float> CGALKernel;
  typedef CGALKernel::Point_3 CGALPoint3;
  typedef CGALKernel::Point_2 CGALPoint2;
  typedef CGAL::Surface_mesh<CGALPoint3> CGALMesh;
  typedef CGALMesh::Edge_index EdgeIndex;
  typedef CGALMesh::Face_index FaceIndex;
  typedef CGALMesh::Vertex_index VertexIndex;

  Eigen::Vector3f CGALPoint3ToEigenVector3f(const CGALPoint3 &vertex) const;
  CGALPoint3 EigenVector3fToCGALVertex(const Eigen::Vector3f &point) const;
  CGALPoint2 EigenVector2fToCGALPoint2(const Eigen::Vector2f &point) const;
  Eigen::Vector2f CGALPoint2ToEigenVector2f(const CGALPoint2 &vertex) const;

  // The interal representation of the mesh is and CGAL triangular surface
  // mesh. This allows for efficient storage and access internally, and a
  // simplified interface wrapped in the REPLAY Mesh class.
  CGALMesh mesh_;
  CGALMesh::Property_map<VertexIndex, CGALPoint2> uv_map_;
};

} // namespace replay
