#include <glog/logging.h>
#include <replay/geometry/mesh.h>
#include <replay/geometry/plane.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

namespace replay {

Plane::Plane(const float nx, const float ny, const float nz, const float d)
    : normal_(nx, ny, nz), d_(d) {
  normal_.normalize();
}

Plane::Plane(const Eigen::Vector3f &point, const Eigen::Vector3f &normal) {
  normal_ = normal.normalized();
  d_ = (-point).dot(normal_);
}

Plane::Plane(const std::vector<Eigen::Vector3f> &points) {
  LOG(INFO) << "Fitting plane to " << points.size() << " points.";
  CHECK_GE(points.size(), 3) << "Not enough points to fit plane.";

  std::vector<Eigen::Vector3f> points_ransac = points;
  std::unordered_set<int> inliers;
  int top_inlier_count = 0;
  const float ransac_inlier_distance = 0.1;
  for (int ransac_iteration = 0; ransac_iteration < 100; ransac_iteration++) {
    LOG(INFO) << "Ransac iteration " << ransac_iteration;
    std::random_shuffle(points_ransac.begin(), points_ransac.end());
    if ((points_ransac[1] - points_ransac[0])
            .normalized()
            .dot((points_ransac[2] - points_ransac[0]).normalized()) >= 0.99) {
      continue;
    }
    *this = Plane(points_ransac[0], points_ransac[1], points_ransac[2]);
    int num_inliers = 0;
    std::unordered_set<int> hypothesis_inliers;
    for (int i = 0; i < points.size(); i++) {
      if (UnsignedDistance(points[i]) < ransac_inlier_distance) {
        num_inliers++;
        hypothesis_inliers.insert(i);
      }
    }
    if (num_inliers > top_inlier_count) {
      top_inlier_count = num_inliers;
      inliers.clear();
      inliers.insert(hypothesis_inliers.begin(), hypothesis_inliers.end());
    }
    LOG(INFO) << "Inliers: " << num_inliers;
    if (num_inliers > 0.50 * points.size()) {
      break;
    }
  }

  CHECK_GE(inliers.size(), 3);

  Eigen::Vector3f Basis1, Basis2;
  std::vector<Eigen::Vector4f> WeightedPoints(inliers.size());
  int i = 0;
  for (auto inlier : inliers) {
    WeightedPoints[i] = Eigen::Vector4f(points[inlier][0], points[inlier][1],
                                        points[inlier][2], 1.0f);
    i++;
  }
  LOG(INFO) << "Fitting least-squares plane to " << WeightedPoints.size() << " points";
  float NormalEigenvalue;
  float residual_error;
  CHECK(FitToPoints(WeightedPoints, Basis1, Basis2, NormalEigenvalue,
                    residual_error));
  LOG(INFO) << "Fit plane with residual error " << residual_error;
}

void Find_ScatterMatrix(const std::vector<Eigen::Vector4f> &Points,
                        const Eigen::Vector3f &Centroid,
                        float ScatterMatrix[3][3], int Order[3]) {
  int i, TempI;
  float TempD;

  /*    To compute the correct scatter matrix, the centroid must be
  **    subtracted from all points.  If the plane is to be forced to pass
  **    through the origin (0,0,0), then the Centroid was earlier set
  **    equal to (0,0,0).  This, of course, is NOT the true Centroid of
  **    the set of points!  Since the matrix is symmetrical about its
  **    diagonal, one-third of it is redundant and is simply found at
  **    the end.
  */
  for (i = 0; i < 3; i++) {
    ScatterMatrix[i][0] = ScatterMatrix[i][1] = ScatterMatrix[i][2] = 0;
  }

  for (int i = 0; i < Points.size(); i++) {
    const Eigen::Vector4f &P = Points[i];
    Eigen::Vector3f d = Eigen::Vector3f(P[0], P[1], P[2]) - Centroid;
    float Weight = P[3];
    ScatterMatrix[0][0] += d[0] * d[0] * Weight;
    ScatterMatrix[0][1] += d[0] * d[1] * Weight;
    ScatterMatrix[0][2] += d[0] * d[2] * Weight;
    ScatterMatrix[1][1] += d[1] * d[1] * Weight;
    ScatterMatrix[1][2] += d[1] * d[2] * Weight;
    ScatterMatrix[2][2] += d[2] * d[2] * Weight;
  }
  ScatterMatrix[1][0] = ScatterMatrix[0][1];
  ScatterMatrix[2][0] = ScatterMatrix[0][2];
  ScatterMatrix[2][1] = ScatterMatrix[1][2];

  /*    Now, perform a sort of "Matrix-sort", whereby all the larger elements
  **    in the matrix are relocated towards the lower-right portion of the
  **    matrix.  This is done as a requisite of the tred2 and tqli algorithms,
  **    for which the scatter matrix is being computed as an input.
  **    "Order" is a 3 element array that will keep track of the xyz order
  **    in the ScatterMatrix.
  */
  Order[0] = 0; /* Beginning order is x-y-z, as found above */
  Order[1] = 1;
  Order[2] = 2;
  if (ScatterMatrix[0][0] > ScatterMatrix[1][1]) {
    TempD = ScatterMatrix[0][0];
    ScatterMatrix[0][0] = ScatterMatrix[1][1];
    ScatterMatrix[1][1] = TempD;
    TempD = ScatterMatrix[0][2];
    ScatterMatrix[0][2] = ScatterMatrix[2][0] = ScatterMatrix[1][2];
    ScatterMatrix[1][2] = ScatterMatrix[2][1] = TempD;
    TempI = Order[0];
    Order[0] = Order[1];
    Order[1] = TempI;
  }
  if (ScatterMatrix[1][1] > ScatterMatrix[2][2]) {
    TempD = ScatterMatrix[1][1];
    ScatterMatrix[1][1] = ScatterMatrix[2][2];
    ScatterMatrix[2][2] = TempD;
    TempD = ScatterMatrix[0][1];
    ScatterMatrix[0][1] = ScatterMatrix[1][0] = ScatterMatrix[0][2];
    ScatterMatrix[0][2] = ScatterMatrix[2][0] = TempD;
    TempI = Order[1];
    Order[1] = Order[2];
    Order[2] = TempI;
  }
  if (ScatterMatrix[0][0] > ScatterMatrix[1][1]) {
    TempD = ScatterMatrix[0][0];
    ScatterMatrix[0][0] = ScatterMatrix[1][1];
    ScatterMatrix[1][1] = TempD;
    TempD = ScatterMatrix[0][2];
    ScatterMatrix[0][2] = ScatterMatrix[2][0] = ScatterMatrix[1][2];
    ScatterMatrix[1][2] = ScatterMatrix[2][1] = TempD;
    TempI = Order[0];
    Order[0] = Order[1];
    Order[1] = TempI;
  }
}

/*
**    This code is taken from ``Numerical Recipes in C'', 2nd
**    and 3rd editions, by Press, Teukolsky, Vetterling and
**    Flannery, Cambridge University Press, 1992, 1994.
**
*/

/*
**    tred2 Householder reduction of a float, symmetric matrix a[1..n][1..n].
**    On output, a is replaced by the orthogonal matrix q effecting the
**    transformation. d[1..n] returns the diagonal elements of the
**    tridiagonal matrix, and e[1..n] the off-diagonal elements, with
**    e[1]=0.
**
**    For my problem, I only need to handle a 3x3 symmetric matrix,
**    so it can be simplified.
**    Therefore n=3.
**
**    Attention: in the book, the index for array starts from 1,
**    but in C, index should start from zero. so I need to modify it.
**    I think it is very simple to modify, just substract 1 from all the
**    index.
*/

#define SIGN(a, b) ((b) < 0 ? -fabs(a) : fabs(a))

void tred2(float a[3][3], float d[3], float e[3]) {
  int l, k, i, j;
  float scale, hh, h, g, f;

  for (i = 3; i >= 2; i--) {
    l = i - 1;
    h = scale = 0.0;
    if (l > 1) {
      for (k = 1; k <= l; k++) scale += fabs(a[i - 1][k - 1]);
      if (scale == 0.0) /* skip transformation */
        e[i - 1] = a[i - 1][l - 1];
      else {
        for (k = 1; k <= l; k++) {
          a[i - 1][k - 1] /= scale; /* use scaled a's for transformation. */
          h += a[i - 1][k - 1] * a[i - 1][k - 1]; /* form sigma in h. */
        }
        f = a[i - 1][l - 1];
        g = f > 0 ? -sqrt(h) : sqrt(h);
        e[i - 1] = scale * g;
        h -= f * g;              /* now h is equation (11.2.4) */
        a[i - 1][l - 1] = f - g; /* store u in the ith row of a. */
        f = 0.0;
        for (j = 1; j <= l; j++) {
          a[j - 1][i - 1] =
              a[i - 1][j - 1] / h; /* store u/H in ith column of a. */
          g = 0.0;                 /* form an element of A.u in g */
          for (k = 1; k <= j; k++) g += a[j - 1][k - 1] * a[i - 1][k - 1];
          for (k = j + 1; k <= l; k++) g += a[k - 1][j - 1] * a[i - 1][k - 1];
          e[j - 1] =
              g / h; /* form element of p in temorarliy unused element of e. */
          f += e[j - 1] * a[i - 1][j - 1];
        }
        hh = f / (h + h);        /* form K, equation (11.2.11) */
        for (j = 1; j <= l; j++) /* form q and store in e overwriting p. */
        {
          f = a[i - 1][j - 1]; /* Note that e[l]=e[i-1] survives */
          e[j - 1] = g = e[j - 1] - hh * f;
          for (k = 1; k <= j; k++) /* reduce a, equation (11.2.13) */
            a[j - 1][k - 1] -= (f * e[k - 1] + g * a[i - 1][k - 1]);
        }
      }
    } else
      e[i - 1] = a[i - 1][l - 1];
    d[i - 1] = h;
  }

  /*
  **    For computing eigenvector.
  */
  d[0] = 0.0;
  e[0] = 0.0;

  for (i = 1; i <= 3; i++) /* begin accumualting of transfomation matrices */
  {
    l = i - 1;
    if (d[i - 1]) /* this block skipped when i=1 */
    {
      for (j = 1; j <= l; j++) {
        g = 0.0;
        for (k = 1; k <= l; k++) /* use u and u/H stored in a to form P.Q */
          g += a[i - 1][k - 1] * a[k - 1][j - 1];
        for (k = 1; k <= l; k++) a[k - 1][j - 1] -= g * a[k - 1][i - 1];
      }
    }
    d[i - 1] = a[i - 1][i - 1];
    a[i - 1][i - 1] = 1.0; /* reset row and column of a to identity matrix for
                              next iteration */
    for (j = 1; j <= l; j++) a[j - 1][i - 1] = a[i - 1][j - 1] = 0.0;
  }
}

/*
**    QL algo with implicit shift, to determine the eigenvalues and
**    eigenvectors of a float,symmetric  tridiagonal matrix, or of a float,
**    symmetric matrix previously reduced by algo tred2.
**    On input , d[1..n] contains the diagonal elements of the tridiagonal
**    matrix. On output, it returns the eigenvalues. The vector e[1..n]
**    inputs the subdiagonal elements of the tridiagonal matrix, with e[1]
**    arbitrary. On output e is destroyed. If the eigenvectors of a
**    tridiagonal matrix are desired, the matrix z[1..n][1..n] is input
**    as the identity matrix. If the eigenvectors of a matrix that has
**    been reduced by tred2 are required, then z is input as the matrix
**    output by tred2. In either case, the kth column of z returns the
**    normalized eigenvector corresponding to d[k].
**
*/
void tqli(float d[3], float e[3], float z[3][3]) {
  int m, l, iter, i, k;
  float s, r, p, g, f, dd, c, b;

  for (i = 2; i <= 3; i++)
    e[i - 2] = e[i - 1]; /* convenient to renumber the elements of e */
  e[2] = 0.0;
  for (l = 1; l <= 3; l++) {
    iter = 0;
    do {
      for (m = l; m <= 2; m++) {
        /*
        **    Look for a single small subdiagonal element
        **    to split the matrix.
        */
        dd = fabs(d[m - 1]) + fabs(d[m]);
        if (fabs(e[m - 1]) + dd == dd) break;
      }
      if (m != l) {
        if (iter++ == 30) {
          printf("\nToo many iterations in TQLI");
        }
        g = (d[l] - d[l - 1]) / (2.0f * e[l - 1]); /* form shift */
        r = sqrt((g * g) + 1.0f);
        g = d[m - 1] - d[l - 1] +
            e[l - 1] / (g + SIGN(r, g)); /* this is dm-ks */
        s = c = 1.0;
        p = 0.0;
        for (i = m - 1; i >= l; i--) {
          /*
          **    A plane rotation as in the original
          **    QL, followed by Givens rotations to
          **    restore tridiagonal form.
          */
          f = s * e[i - 1];
          b = c * e[i - 1];
          if (fabs(f) >= fabs(g)) {
            c = g / f;
            r = sqrt((c * c) + 1.0f);
            e[i] = f * r;
            c *= (s = 1.0f / r);
          } else {
            s = f / g;
            r = sqrt((s * s) + 1.0f);
            e[i] = g * r;
            s *= (c = 1.0f / r);
          }
          g = d[i] - p;
          r = (d[i - 1] - g) * s + 2.0f * c * b;
          p = s * r;
          d[i] = g + p;
          g = c * r - b;
          for (k = 1; k <= 3; k++) {
            /*
            **    Form eigenvectors
            */
            f = z[k - 1][i];
            z[k - 1][i] = s * z[k - 1][i - 1] + c * f;
            z[k - 1][i - 1] = c * z[k - 1][i - 1] - s * f;
          }
        }
        d[l - 1] = d[l - 1] - p;
        e[l - 1] = g;
        e[m - 1] = 0.0f;
      }
    } while (m != l);
  }
}

namespace {
bool IsValid(const Eigen::Vector3f &vec) {
  for (int i = 0; i < 3; i++) {
    if (isnan(vec[i])) {
      return false;
    }
  }
  return true;
}

}  // namespace

bool Plane::FitToPoints(const std::vector<Eigen::Vector4f> &Points,
                        Eigen::Vector3f &Basis1, Eigen::Vector3f &Basis2,
                        float &NormalEigenvalue, float &ResidualError) {
  Eigen::Vector3f Centroid, Normal;

  float ScatterMatrix[3][3];
  int Order[3];
  float DiagonalMatrix[3];
  float OffDiagonalMatrix[3];

  // Find centroid
  Centroid = Eigen::Vector3f::Zero();
  float TotalWeight = 0.0f;
  for (size_t i = 0; i < Points.size(); i++) {
    TotalWeight += Points[i][3];
    Centroid += Eigen::Vector3f(Points[i][0], Points[i][1], Points[i][2]) *
                Points[i][3];
  }
  Centroid /= TotalWeight;

  // Compute scatter matrix
  Find_ScatterMatrix(Points, Centroid, ScatterMatrix, Order);

  tred2(ScatterMatrix, DiagonalMatrix, OffDiagonalMatrix);
  tqli(DiagonalMatrix, OffDiagonalMatrix, ScatterMatrix);

  /*
  **    Find the smallest eigenvalue first.
  */
  float Min = DiagonalMatrix[0];
  float Max = DiagonalMatrix[0];
  size_t MinIndex = 0;
  size_t MiddleIndex = 0;
  size_t MaxIndex = 0;
  for (size_t i = 1; i < 3; i++) {
    if (DiagonalMatrix[i] < Min) {
      Min = DiagonalMatrix[i];
      MinIndex = i;
    }
    if (DiagonalMatrix[i] > Max) {
      Max = DiagonalMatrix[i];
      MaxIndex = i;
    }
  }
  for (size_t i = 0; i < 3; i++) {
    if (MinIndex != i && MaxIndex != i) {
      MiddleIndex = i;
    }
  }
  /*
  **    The normal of the plane is the smallest eigenvector.
  */
  for (size_t i = 0; i < 3; i++) {
    Normal[Order[i]] = ScatterMatrix[i][MinIndex];
    Basis1[Order[i]] = ScatterMatrix[i][MiddleIndex];
    Basis2[Order[i]] = ScatterMatrix[i][MaxIndex];
  }
  NormalEigenvalue = std::fabs(DiagonalMatrix[MinIndex]);
  Basis1.normalize();
  Basis2.normalize();
  Basis1 *= (DiagonalMatrix[MiddleIndex]);
  Basis2 *= (DiagonalMatrix[MaxIndex]);

  if (!IsValid(Basis1) || !IsValid(Basis2) || !IsValid(Normal)) {
    *this = Plane(Centroid, Eigen::Vector3f::UnitX());
    Basis1 = Eigen::Vector3f::UnitY();
    Basis2 = Eigen::Vector3f::UnitZ();
  } else {
    *this = Plane(Centroid, Normal);
  }

  ResidualError = 0.0f;
  for (size_t i = 0; i < Points.size(); i++) {
    ResidualError += UnsignedDistance(
        Eigen::Vector3f(Points[i][0], Points[i][1], Points[i][2]));
  }
  ResidualError /= Points.size();

  return true;
}

Plane::Plane(const Eigen::Vector3f &p1, const Eigen::Vector3f &p2,
             const Eigen::Vector3f &p3) {
  const Eigen::Vector3f v1 = (p2 - p1).normalized();
  const Eigen::Vector3f v2 = (p3 - p1).normalized();
  CHECK_GT(0.99, v1.dot(v2));
  normal_ = v1.cross(v2);
  normal_.normalize();
  *this = Plane(p1, normal_);
  CHECK_LT(UnsignedDistance(p1), 0.0001);
  CHECK_LT(UnsignedDistance(p2), 0.0001);
  CHECK_LT(UnsignedDistance(p3), 0.0001);
}

float Plane::UnsignedDistance(const Eigen::Vector3f &point) const {
  return std::fabs(normal_[0] * point.x() + normal_[1] * point.y() +
                   normal_[2] * point.z() + d_);
}

Eigen::Vector3f Plane::ProjectToPlane(const Eigen::Vector3f &point) const {
  return point - normal_ * (normal_[0] * point.x() + normal_[1] * point.y() +
                            normal_[2] * point.z() + d_);
}

Eigen::Vector3f Plane::Intersect(const Eigen::Vector3f &ray_origin,
                                 const Eigen::Vector3f &ray_direction) const {
  const Eigen::Vector3f direction = ray_direction.normalized();
  const Eigen::Vector3f p0(0, 0, d_ / (-normal_[2]));
  const float n_dot_v = normal_.dot(direction);
  const float distance_to_plane = normal_.dot(p0 - ray_origin);
  const float t = distance_to_plane / n_dot_v;
  const Eigen::Vector3f intersection = ray_origin + ray_direction * t;
  return intersection;
}

Mesh Plane::GetMesh(const Eigen::Vector3f &center, const float extent) const {
  const Eigen::Vector3f p0 = ProjectToPlane(center);
  CHECK_LT(UnsignedDistance(p0), 0.0001);
  Eigen::Vector3f random_vector(std::rand(), std::rand(), std::rand());
  random_vector.normalize();
  while (random_vector.dot(normal_) > 0.99 || random_vector.norm() < 0.01) {
    random_vector = Eigen::Vector3f(std::rand(), std::rand(), std::rand());
    random_vector.normalize();
  }
  const Eigen::Vector3f random_point =
      ProjectToPlane(p0 + normal_.cross(random_vector));
  const Eigen::Vector3f u = (p0 - random_point).normalized();
  const Eigen::Vector3f v = u.cross(normal_);
  const Eigen::Vector3f uo = u * extent;
  const Eigen::Vector3f vo = v * extent;
  Eigen::Vector3f tr = p0 + uo + vo;
  Eigen::Vector3f tl = p0 - uo + vo;
  Eigen::Vector3f bl = p0 - uo - vo;
  Eigen::Vector3f br = p0 + uo - vo;
  return Mesh::Plane(tl, tr, bl, br);
}

Eigen::Vector4f Plane::GetCoefficients() const {
  return Eigen::Vector4f(normal_[0], normal_[1], normal_[2], d_);
}

}  // namespace replay
