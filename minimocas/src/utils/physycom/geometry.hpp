#ifndef PHYSYCOM_UTILS_GEOMETRY_HPP
#define PHYSYCOM_UTILS_GEOMETRY_HPP

#include <type_traits>
#include <tuple>

namespace physycom
{
  enum
  {
    X = 0,
    Y
  };

  template <typename point_t>
  static auto to_coords_tuple(point_t &pt)
  {
    return std::make_tuple(&pt.x, &pt.y);
  }

  template <typename point_t>
  point_t centroid(std::vector<point_t> &points)
  {
    point_t c;
    auto tc = to_coords_tuple(c);
    double x = 0.;
    double y = 0.;
    for (auto &p : points)
    {
      auto tp = to_coords_tuple(p);
      x += double(*std::get<X>(tp));
      y += double(*std::get<Y>(tp));
      //std::cout << "x " << *std::get<X>(tc) << std::endl;
    }
    using vtype = typename std::remove_reference<decltype(*std::get<X>(tc))>::type;
    *std::get<X>(tc) = vtype( x / double(points.size()));
    *std::get<Y>(tc) = vtype( y / double(points.size()));
    return c;
  }

  template <typename point_t>
  bool is_left(point_t &p0, point_t &p1, point_t &p) // fast check : point p is LEFT of segment p1-p2?
  {
    auto tp0 = to_coords_tuple(p0);
    auto tp1 = to_coords_tuple(p1);
    auto tp = to_coords_tuple(p);
    return (*std::get<Y>(tp1) - *std::get<Y>(tp0)) * (*std::get<X>(tp) - *std::get<X>(tp0) ) - ( *std::get<Y>(tp) - *std::get<Y>(tp0) ) * ( *std::get<X>(tp1) - *std::get<X>(tp0) ) > 0;
  }

  template <typename point_t>
  bool point_in_polygon(std::vector<point_t> &polygon, point_t &p)
  {
    // winding number inclusion algorithm
    int wn = 0;
    auto tp = to_coords_tuple(p);
    auto px = *std::get<X>(tp);
    for (int i = 0; i < int(polygon.size()) - 1; ++i)
    {
      auto tp0 = to_coords_tuple(polygon[i]);
      auto tp1 = to_coords_tuple(polygon[i+1]);
      if (*std::get<X>(tp0) <= px)
      {
        if (*std::get<X>(tp1) > px)                 // an upward crossing
          if (is_left(polygon[i], polygon[i+1], p)) // P left of  edge
            ++wn;                                   // have  a valid up intersect
      }
      else
      {
        if (*std::get<X>(tp1) <= px)                   // a downward crossing
          if (! is_left(polygon[i], polygon[i+1], p) ) // P right of  edge
            --wn;                                      // have  a valid down intersect
      }
    }
    return wn != 0; // wn=0 <=> p is outside
  }

  template <typename point_t>
  bool intersection(point_t a0, point_t &a1, point_t &b0, point_t &b1)
  {
    auto ta0 = to_coords_tuple(a0);
    auto ta1 = to_coords_tuple(a1);
    auto tb0 = to_coords_tuple(b0);
    auto tb1 = to_coords_tuple(b1);
    auto a0x = *std::get<X>(ta0);
    auto a1x = *std::get<X>(ta1);
    auto b0x = *std::get<X>(tb0);
    auto b1x = *std::get<X>(tb1);
    auto a0y = *std::get<Y>(ta0);
    auto a1y = *std::get<Y>(ta1);
    auto b0y = *std::get<Y>(tb0);
    auto b1y = *std::get<Y>(tb1);

    auto cross = (b1x - b0x) * (a1y - a0y) - (a1x - a0x) * (b1y - b0y);
    if (!int(cross)) return false;

    float inv_cross = 1.f / float(cross);
    float sa = ( (a1x - b0x) * (a1y - a0y) - (a1x - a0x) * (a1y - b0y) ) * inv_cross;
    float sb = ( (a1x - b0x) * (b1y - b0y) - (b1x - b0x) * (a1y - b0y) ) * inv_cross;

    if (sa > 0 && sa < 1 && sb > 0 && sb < 1)  // segment-segment crossing
      return true;

    return false;
  }
}

#endif // PHYSYCOM_UTILS_GEOMETRY_HPP
