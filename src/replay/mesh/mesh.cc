#include "replay/mesh/mesh.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/compute_normal.h>
#include <CGAL/Polygon_mesh_processing/fair.h>
#include <CGAL/Polygon_mesh_processing/repair.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Surface_mesh_simplification/Edge_collapse_visitor_base.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Count_ratio_stop_predicate.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Midpoint_and_length.h>
#include <CGAL/Surface_mesh_simplification/edge_collapse.h>
#include <CGAL/boost/graph/Euler_operations.h>
#include <CGAL/boost/graph/iterator.h>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <glog/logging.h>

#include <algorithm>
#include <math.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "replay/io/read_ply_file.h"
#include "replay/io/write_ply_file.h"
#include "replay/third_party/theia/util/map_util.h"

namespace replay {
	Mesh::Mesh() : has_uvs_(false) {}

	int Mesh::NumVertices() const { return mesh_.number_of_vertices(); }

	int Mesh::NumTriangleFaces() const { return mesh_.number_of_faces(); }

	bool Mesh::HasUVs() const { return has_uvs_; }

	// Methods to load/save the mesh to/from disk. The file type is deduced from
	// the extention of the provided filename.
	bool Mesh::Load(const std::string &mesh_file) {
		return ReadPLYFile(mesh_file, this);
	}

	void Mesh::Save(const std::string &output_file) {
		CHECK(WritePLYFile(output_file, *this, true));
	}

	void Mesh::Append(const Mesh &rhs_mesh) {
		std::unordered_map<VertexId, VertexId> rhs_vertex_to_lhs;
		for (int i = 0; i < rhs_mesh.NumVertices(); i++) {
			rhs_vertex_to_lhs[i] = AddVertex(rhs_mesh.VertexPosition(i));
			if (rhs_mesh.HasUVs() && HasUVs()) {
				SetVertexUV(rhs_vertex_to_lhs[i], rhs_mesh.VertexUV(i));
			}
		}
		for (int i = 0; i < rhs_mesh.NumTriangleFaces(); i++) {
			const Eigen::Matrix<VertexId, 3, 1> &triangle_face =
				rhs_mesh.GetVertexIdsForTriangleFace(i);
			AddTriangleFace(rhs_vertex_to_lhs[triangle_face[0]],
				rhs_vertex_to_lhs[triangle_face[1]],
				rhs_vertex_to_lhs[triangle_face[2]]);
		}
	}

	// Vertex getter/setters.
	VertexId Mesh::AddVertex(const Eigen::Vector3f &vertex) {
		const auto &vertex_index =
			mesh_.add_vertex(EigenVector3fToCGALVertex(vertex));
		return mesh_.is_valid(vertex_index) ? vertex_index : kInvalidVertexId;
	}

	bool Mesh::RemoveVertex(const VertexId vertex_id) {
		DCHECK_NE(vertex_id, kInvalidVertexId);
		DCHECK(mesh_.is_valid(vertex_id));
		mesh_.remove_vertex(VertexIndex(vertex_id));
		return true;
	}

	// Removes all isolated vertices in the mesh.
	int Mesh::RemoveIsolatedVertices() {
		return CGAL::Polygon_mesh_processing::remove_isolated_vertices(mesh_);
	}

	void Mesh::SetVertexPosition(const VertexId vertex_id,
		const Eigen::Vector3f &position) {
		DCHECK(mesh_.is_valid(VertexIndex(vertex_id)));
		mesh_.point(VertexIndex(vertex_id)) = EigenVector3fToCGALVertex(position);
	}

	Eigen::Vector3f Mesh::VertexPosition(const VertexId vertex_id) const {
		DCHECK(mesh_.is_valid(VertexIndex(vertex_id)));
		return CGALPoint3ToEigenVector3f(mesh_.point(VertexIndex(vertex_id)));
	}

	int Mesh::ValenceOfVertex(const VertexId vertex_id) const {
		return mesh_.degree(VertexIndex(vertex_id));
	}

	bool Mesh::IsVertexOnBoundary(const VertexId vertex_id) const {
		return mesh_.is_border(VertexIndex(vertex_id));
	}

	bool Mesh::IsFaceOnBoundary(const TriangleFaceId face_id) const {
		for (const auto &vertex :
			CGAL::vertices_around_face(mesh_.halfedge(FaceIndex(face_id)), mesh_)) {
			if (mesh_.is_border(vertex)) {
				return true;
			}
		}
		return false;
	}

	TriangleFaceId Mesh::AddTriangleFace(const VertexId vertex1_id,
		const VertexId vertex2_id,
		const VertexId vertex3_id) {
		CHECK(mesh_.is_valid(VertexIndex(vertex1_id)));
		CHECK(mesh_.is_valid(VertexIndex(vertex2_id)));
		CHECK(mesh_.is_valid(VertexIndex(vertex3_id)));

		// Fetch the vertices.
		const auto triangle_index =
			mesh_.add_face(VertexIndex(vertex1_id), VertexIndex(vertex2_id),
				VertexIndex(vertex3_id));

		// Unlike the method for adding vertices, CGAL's add_face function will go
		// berserk when you add a topologically invalid triangle. For some reason it
		// even throws errors to mesh_.is_valid(triangle_index) so we must explicitly
		// detect a failure here.
		if (triangle_index > NumTriangleFaces() || !mesh_.is_valid(triangle_index)) {
			LOG(WARNING) << "Attempted to add a topologically invalid triangle to "
				"vertices with ids "
				<< vertex1_id << ", " << vertex2_id << ", " << vertex3_id
				<< ". Skipping this triangle face.";
			return kInvalidTriangleFaceId;
		}

		return triangle_index;
	}

	bool Mesh::RemoveTriangleFace(const TriangleFaceId face_id) {
		DCHECK(mesh_.is_valid(FaceIndex(face_id)));
		mesh_.remove_face(FaceIndex(face_id));
		return true;
	}

	void Mesh::CollectGarbage() { mesh_.collect_garbage(); }

	Eigen::Matrix<VertexId, 3, 1>
		Mesh::GetVertexIdsForTriangleFace(const TriangleFaceId face_id) const {
		DCHECK(mesh_.is_valid(FaceIndex(face_id)));

		Eigen::Matrix<VertexId, 3, 1> vertex_ids;
		int i = 0;
		for (const auto &vertex :
			CGAL::vertices_around_face(mesh_.halfedge(FaceIndex(face_id)), mesh_)) {
			vertex_ids[i] = vertex;
			++i;
		}
		return vertex_ids;
	}

	float Mesh::MedianEdgeLength() const {
		std::vector<float> sq_edge_lengths;
		sq_edge_lengths.reserve(mesh_.number_of_edges());
		for (const auto &edge : mesh_.edges()) {
			const CGALPoint3 &p = mesh_.point(mesh_.vertex(edge, 0));
			const CGALPoint3 &q = mesh_.point(mesh_.vertex(edge, 1));
			sq_edge_lengths.emplace_back(CGAL::squared_distance(p, q));
		}
		const int median_index = sq_edge_lengths.size() / 2;
		std::nth_element(sq_edge_lengths.begin(),
			sq_edge_lengths.begin() + median_index,
			sq_edge_lengths.end());
		return std::sqrt(sq_edge_lengths[median_index]);
	}

	float Mesh::MeanEdgeLength() const {
		float mean_edge_length = 0;
		for (const auto &edge : mesh_.edges()) {
			const CGALPoint3 &p = mesh_.point(mesh_.vertex(edge, 0));
			const CGALPoint3 &q = mesh_.point(mesh_.vertex(edge, 1));
			mean_edge_length += CGAL::sqrt(CGAL::squared_distance(p, q));
		}
		return mean_edge_length / static_cast<float>(mesh_.number_of_edges());
	}

	Eigen::Vector3f Mesh::ComputeFaceNormal(const TriangleFaceId face_id) const {
		const auto &normal = CGAL::Polygon_mesh_processing::compute_face_normal(
			FaceIndex(face_id), mesh_);
		return Eigen::Vector3f(normal.x(), normal.y(), normal.z());
	}

	Eigen::Vector3f Mesh::ComputeVertexNormal(const VertexId vertex_id) const {
		const auto &normal = CGAL::Polygon_mesh_processing::compute_vertex_normal(
			VertexIndex(vertex_id), mesh_);
		return Eigen::Vector3f(normal.x(), normal.y(), normal.z());
	}

	std::unordered_map<TriangleFaceId, Eigen::Vector3f>
		Mesh::ComputeAllFaceNormals() const {
		std::unordered_map<TriangleFaceId, Eigen::Vector3f> normals;
		normals.reserve(NumTriangleFaces());
		for (const auto &face_index : mesh_.faces()) {
			normals[face_index] = ComputeFaceNormal(face_index);
		}
		return normals;
	}

	std::unordered_map<VertexId, Eigen::Vector3f>
		Mesh::ComputeAllVertexNormals() const {
		std::unordered_map<VertexId, Eigen::Vector3f> normals;
		normals.reserve(NumVertices());
		for (const auto &vertex_index : mesh_.vertices()) {
			normals[vertex_index] = ComputeVertexNormal(vertex_index);
		}
		return normals;
	}

	Eigen::Vector3f Mesh::ComputeFaceCentroid(const TriangleFaceId face_id) const {
		// Compute the triangle centroid.
		const auto &halfedge_index = mesh_.halfedge(FaceIndex(face_id));
		const CGALPoint3 centroid =
			CGAL::centroid(mesh_.point(mesh_.source(halfedge_index)),
				mesh_.point(mesh_.target(halfedge_index)),
				mesh_.point(mesh_.target(mesh_.next(halfedge_index))));
		return CGALPoint3ToEigenVector3f(centroid);
	}

	Eigen::Vector3f Mesh::ComputeVertexCentroid(const VertexId vertex_id) const {
		DCHECK(mesh_.is_valid(VertexIndex(vertex_id)));

		Eigen::Vector3f centroid(0, 0, 0);
		int num_neighbors = 0;
		for (const auto &neighbor_id : CGAL::vertices_around_target(
			mesh_.halfedge(VertexIndex(vertex_id)), mesh_)) {
			centroid +=
				CGALPoint3ToEigenVector3f(mesh_.point(VertexIndex(neighbor_id)));
			++num_neighbors;
		}
		return centroid / static_cast<float>(num_neighbors);
	}

	std::unordered_map<TriangleFaceId, float> Mesh::ComputeAllFaceAreas() const {
		std::unordered_map<TriangleFaceId, float> areas;
		areas.reserve(NumTriangleFaces());
		for (const auto &face_index : mesh_.faces()) {
			const auto &vertex_ids = GetVertexIdsForTriangleFace(face_index);
			areas[face_index] =
				CGALKernel::Compute_area_3()(mesh_.point(VertexIndex(vertex_ids[0])),
					mesh_.point(VertexIndex(vertex_ids[1])),
					mesh_.point(VertexIndex(vertex_ids[2])));
		}
		return areas;
	}

	// Return all triangles that contain this vertex.
	std::unordered_set<TriangleFaceId>
		Mesh::TrianglesAtVertex(const VertexId vertex_id) const {
		CHECK(mesh_.is_valid(VertexIndex(vertex_id)));

		std::unordered_set<TriangleFaceId> triangle_ids;
		const auto &halfedge = mesh_.halfedge(VertexIndex(vertex_id));
		for (const auto &face_index : CGAL::faces_around_target(halfedge, mesh_)) {
			triangle_ids.emplace(face_index);
		}
		// Some vertices have a connection to an invalid triangle to indicate that
		// they lie on the hull of the mesh, so we must remove any references to
		// invalid triangles.
		triangle_ids.erase(kInvalidTriangleFaceId);

		return triangle_ids;
	}

	// Return all vertices that are contain edges to the given vertex
	std::unordered_set<VertexId>
		Mesh::EdgesToVertex(const VertexId vertex_id) const {
		DCHECK(mesh_.is_valid(VertexIndex(vertex_id)));

		std::unordered_set<VertexId> neighbors;
		for (const auto &neighbor_id : CGAL::vertices_around_target(
			mesh_.halfedge(VertexIndex(vertex_id)), mesh_)) {
			neighbors.emplace(neighbor_id);
		}
		return neighbors;
	}

	std::unordered_set<TriangleFaceId>
		Mesh::NeighborsOfTriangle(const TriangleFaceId face_id) const {
		DCHECK(mesh_.is_valid(FaceIndex(face_id)));

		std::unordered_set<TriangleFaceId> neighbor_faces;
		for (const auto &neighbor_face_id :
			CGAL::faces_around_face(mesh_.halfedge(FaceIndex(face_id)), mesh_)) {
			neighbor_faces.emplace(neighbor_face_id);
		}
		neighbor_faces.erase(kInvalidTriangleFaceId);

		return neighbor_faces;
	}

	void Mesh::SetVertexUV(const VertexId &vertex, const float &u, const float &v) {
		SetVertexUV(vertex, Eigen::Vector2f(u, v));
	}

	void Mesh::SetVertexUV(const VertexId &vertex, const Eigen::Vector2f &uv) {
		DCHECK(mesh_.is_valid(VertexIndex(vertex)));
		if (!has_uvs_) {
			std::pair<CGALMesh::Property_map<VertexIndex, CGALPoint2>, bool> added_map =
				mesh_.add_property_map<VertexIndex, CGALPoint2>("uv", CGALPoint2());
			CHECK(added_map.second);
			uv_map_ = added_map.first;
			has_uvs_ = true;
		}
		mesh_.property_map<VertexIndex, CGALPoint2>("uv").first[VertexIndex(vertex)] =
			EigenVector2fToCGALPoint2(uv);
	}

	Eigen::Vector2f Mesh::VertexUV(const VertexId &vertex) const {
		CHECK(mesh_.is_valid(VertexIndex(vertex)));
		CHECK(has_uvs_) << "Mesh does not have UVs!";
		return CGALPoint2ToEigenVector2f(
			mesh_.property_map<VertexIndex, CGALPoint2>("uv")
			.first[VertexIndex(vertex)]);
	}

	void Mesh::ApplyTransform(const Eigen::Matrix4f& transform) {
		for (int i = 0; i < NumVertices(); i++) {
			SetVertexPosition(i, (transform * VertexPosition(i).homogeneous()).hnormalized());
		}
	}

	void Mesh::FlipFaceOrientation() {
		CGAL::Polygon_mesh_processing::reverse_face_orientations(mesh_);
	}

void Mesh::SubdivideTriangle(const TriangleFaceId triangle) {
  // Compute the triangle centroid.
  const auto &halfedge_index = mesh_.halfedge(FaceIndex(triangle));
  CGALPoint3 old_face_centroid =
      CGAL::centroid(mesh_.point(mesh_.source(halfedge_index)),
                     mesh_.point(mesh_.target(halfedge_index)),
                     mesh_.point(mesh_.target(mesh_.next(halfedge_index))));

  // Split the triangle if the edge is not on a border.
  if (mesh_.is_valid(halfedge_index) && !mesh_.is_border(halfedge_index)) {
    // Adding the vertex does not set the point to the centroid so we must do
    // that after.
    const auto &new_edge =
        CGAL::Euler::add_center_vertex(halfedge_index, mesh_);
    mesh_.point(mesh_.target(new_edge)) = old_face_centroid;
  }
}

bool Mesh::DecimateMesh(const double desired_ratio) {
  CHECK_GT(desired_ratio, 0.0)
      << "The desired decimation ratio must be between 0.0 and 1.0";
  CHECK_LT(desired_ratio, 1.0)
      << "The desired decimation ratio must be between 0.0 and 1.0";
  namespace SMS = CGAL::Surface_mesh_simplification;

  SMS::Count_ratio_stop_predicate<CGALMesh> stop(desired_ratio);
  const int num_edges_removed = SMS::edge_collapse(
      mesh_, stop,
      CGAL::parameters::get_cost(SMS::Edge_length_cost<CGALMesh>())
          .get_placement(SMS::Midpoint_placement<CGALMesh>()));
  CollectGarbage();
  return num_edges_removed > 0;
}

// Given a list of faces for a submesh, creates a new mesh from only those
// triangles.
bool Mesh::GetSubMesh(const std::unordered_set<TriangleFaceId> &faces,
                      Mesh *new_mesh) const {
  std::unordered_map<VertexId, VertexId> old_to_new_vertex_id_map;

  // Loop over all triangles we want to add to the new mesh. For each triangle,
  // add the vertices (if they have not already been added) and the triangle
  // face.
  for (const TriangleFaceId &face_id : faces) {
    // Get the "old" vertex ids of the 3 triangle vertices.
    const Eigen::Matrix<VertexId, 3, 1> &triangle_vertex_ids =
        GetVertexIdsForTriangleFace(face_id);

    // For each "old" vertex, add it to the new mesh if it has not already been
    // added.
    for (int i = 0; i < 3; i++) {
      // If the vertex has already been added to our new mesh, skip this vertex.
      if (theia::ContainsKey(old_to_new_vertex_id_map,
                             triangle_vertex_ids[i])) {
        continue;
      }

      // Add the vertex to the new mesh and store the mapping between "old" and
      // "new" vertex ids.
      const VertexId new_vertex_id =
          new_mesh->AddVertex(VertexPosition(triangle_vertex_ids[i]));
      old_to_new_vertex_id_map[triangle_vertex_ids[i]] = new_vertex_id;
      if (HasUVs()) {
        new_mesh->SetVertexUV(new_vertex_id, VertexUV(triangle_vertex_ids[i]));
      }
    }
    // Add the triangle to the new mesh.
    new_mesh->AddTriangleFace(old_to_new_vertex_id_map[triangle_vertex_ids[0]],
                              old_to_new_vertex_id_map[triangle_vertex_ids[1]],
                              old_to_new_vertex_id_map[triangle_vertex_ids[2]]);
  }

  return true;
}

// Splits the mesh into submeshes that each have less than max_count vertices
bool Mesh::SplitMeshByMaxVertexCount(int max_count, std::vector<Mesh> *meshes) {
  CHECK_NOTNULL(meshes)->clear();
  meshes->resize(static_cast<float>(NumVertices()) / max_count);
  for (int i = 0; i < meshes->size(); i++) {
    std::unordered_set<TriangleFaceId> faces;

    if (!GetSubMesh(faces, &(*meshes)[i])) {
      return false;
    }
  }
  return true;
}

// Returns a pointer to the mesh vertex positions. This is mainly for use with
// OpenGL.
const float *Mesh::vertex_positions() const {
  return &(mesh_.points()[VertexIndex(0)][0]);
}

const float *Mesh::uvs() const {
  return &(mesh_.property_map<VertexIndex, CGALPoint2>("uv")
               .first[VertexIndex(0)][0]);
}

// Returns all triangles by their vertex ids.
std::vector<Eigen::Matrix<VertexId, 3, 1>> Mesh::triangles() const {
  std::vector<Eigen::Matrix<VertexId, 3, 1>> triangles_vertices;
  triangles_vertices.reserve(NumTriangleFaces());

  for (const auto &face_index : mesh_.faces()) {
    triangles_vertices.emplace_back(GetVertexIdsForTriangleFace(face_index));
  }

  return triangles_vertices;
}

Eigen::Vector3f
Mesh::CGALPoint3ToEigenVector3f(const CGALPoint3 &vertex) const {
  return Eigen::Vector3f(vertex.x(), vertex.y(), vertex.z());
}

Mesh::CGALPoint3
Mesh::EigenVector3fToCGALVertex(const Eigen::Vector3f &point) const {
  return CGALPoint3(point.x(), point.y(), point.z());
}

Mesh::CGALPoint2
Mesh::EigenVector2fToCGALPoint2(const Eigen::Vector2f &point) const {
  return CGALPoint2(point.x(), point.y());
}

Eigen::Vector2f
Mesh::CGALPoint2ToEigenVector2f(const CGALPoint2 &vertex) const {
  return Eigen::Vector2f(vertex.x(), vertex.y());
}

} // namespace replay
