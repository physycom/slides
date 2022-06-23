#ifndef PHYSYCOM_UTILS_VORONOI_HPP
#define PHYSYCOM_UTILS_VORONOI_HPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <vector>

#include <boost/polygon/voronoi.hpp>

using boost::polygon::voronoi_builder;
using boost::polygon::voronoi_diagram;
using boost::polygon::x;
using boost::polygon::y;
using boost::polygon::low;
using boost::polygon::high;


namespace physycom
{
  constexpr double PI = 3.1415926535;
  inline double atan2_2pi(const double &y, const double &x)
  {
    double th = std::atan2(y,x);
    return (th >= 0) ? th : 2*PI + th;
  }

  struct Point
  {
    int x, y;
    Point() : x(0), y(0) {}
    Point(int x, int y) : x(x), y(y) {}
    bool operator==(const Point &p)
    {
      return x == p.x && y == p.y;
    }
    bool operator!=(const Point &p)
    {
      return ! (*this == p);
    }
    friend std::ostream &operator<<(std::ostream &stream, const Point &p)
    {
      stream << "[" << p.x << "," << p.y << "]";
      return stream;
    }
  };

  struct Segment
  {
    Point p0, p1;
    Segment(int x1, int y1, int x2, int y2) : p0(x1, y1), p1(x2, y2) {}
  };

  struct Box
  {
    int xmin, xmax, ymin, ymax;
    Box() : xmin(0), xmax(0), ymin(0), ymax(0) {}
    Box(const int &xmin, const int &xmax, const int &ymin, const int &ymax) : xmin(xmin), xmax(xmax), ymin(ymin), ymax(ymax) {}
    bool contains(const Point &p)
    {
      return p.x > xmin && p.x < xmax && p.y > ymin && p.y < ymax;
    }
    void encompass(const Point &p)
    {
      if (xmin > p.x)
        xmin = (p.x > 0) ? 0.9*p.x : 1.1*p.x;
      if (xmax < p.x)
        xmax = (p.x > 0) ? 1.1*p.x : 0.9*p.x;
      if (ymin > p.y)
        ymin = (p.y > 0) ? 0.9*p.y : 1.1*p.y;
      if (xmax < p.x)
        ymax = (p.y > 0) ? 1.1*p.y : 0.9*p.y;
    }
    friend std::ostream &operator<<(std::ostream &stream, const Box &b)
    {
      stream << "[" << b.xmin << "," << b.xmax << "]x[" << b.ymin << "," << b.ymax << "]";
      return stream;
    }
  };

  struct Polygon
  {
    std::vector<Point> points;
    Polygon() {}
    Polygon(std::vector<Point> &points) : points(points) {}
  };

} // physycom

namespace boost
{
  namespace polygon
  {
    using namespace physycom;

    template <>
    struct geometry_concept<Point>
    {
      typedef point_concept type;
    };

    template <>
    struct point_traits<Point>
    {
      typedef int coordinate_type;

      static inline coordinate_type get(const Point& point, orientation_2d orient)
      {
        return (orient == HORIZONTAL) ? point.x : point.y;
      }
    };

    template <>
    struct geometry_concept<Segment>
    {
      typedef segment_concept type;
    };

    template <>
    struct segment_traits<Segment>
    {
      typedef int coordinate_type;
      typedef Point point_type;

      static inline point_type get(const Segment& segment, direction_1d dir)
      {
        return dir.to_int() ? segment.p1 : segment.p0;
      }
    };
  }  // polygon
}  // boost

namespace physycom
{
  struct voronoi
  {
    std::vector<Point> points, points_voro;
    Box bounds;
    std::vector<Polygon> cells;
    voronoi_diagram<double> vd;

    voronoi() {}
    voronoi(const std::vector<Point> &points, const Box &bounds) : points(points), bounds(bounds)
    {
      construct_voronoi(this->points.begin(), this->points.end(), &(this->vd));
      int cell_index = 0;
      for (auto cell = vd.cells().begin(); cell != vd.cells().end(); ++cell)
      {
        Polygon pg;
        if (cell->contains_point())
        {
          if (cell->source_category() == boost::polygon::SOURCE_CATEGORY_SINGLE_POINT)
          {
            // retrieve point in cell
            std::size_t index = cell->source_index();
            Point p = this->points[index];
            points_voro.push_back(p);
            std::cout << "cell #" << cell_index << std::endl;

            // loop over cell edges
            auto *edge = cell->incident_edge();
            int edge_cnt = 0;
            do
            {
              edge = edge->next();
              if (edge->is_finite())
              {
                Point p0 = Point(edge->vertex0()->x(), edge->vertex0()->y());
                Point p1 = Point(edge->vertex1()->x(), edge->vertex1()->y());
                std::cout << "FINITE " << p0 << " " << p1 << std::endl;
                if (!pg.points.size())
                  pg.points.push_back(p0);
                pg.points.push_back(p1);
              }
              else if (edge->is_infinite())
              {
                auto cell1 = edge->cell();
                auto cell2 = edge->twin()->cell();
                Point origin, direction;
                std::vector<Point> clipped_edge;
                if (cell1->contains_point() && cell2->contains_point())
                {
                  Point p1 = points[edge->cell()->source_index()];
                  Point p2 = points[edge->twin()->cell()->source_index()];
                  origin.x = (p1.x + p2.x) * 0.5;
                  origin.y = (p1.y + p2.y) * 0.5;
                  direction.x = p1.y - p2.y;
                  direction.y = p2.x - p1.x;
                }

                double theta1 = physycom::atan2_2pi(bounds.ymax - origin.y, bounds.xmax - origin.x);
                double theta2 = physycom::atan2_2pi(bounds.ymax - origin.y, bounds.xmin - origin.x);
                double theta3 = physycom::atan2_2pi(bounds.ymin - origin.y, bounds.xmin - origin.x);
                double theta4 = physycom::atan2_2pi(bounds.ymin - origin.y, bounds.xmax - origin.x);

/*
                std::cout << "t1 " << theta1 * 180 / PI << " " << bounds.ymax - origin.y << " " << bounds.xmax - origin.x << std::endl
                          << "t2 " << theta2 * 180 / PI << " " << bounds.ymax - origin.y << " " << bounds.xmin - origin.x << std::endl
                          << "t3 " << theta3 * 180 / PI << " " << bounds.ymin - origin.y << " " << bounds.xmin - origin.x << std::endl
                          << "t4 " << theta4 * 180 / PI << " " << bounds.ymin - origin.y << " " << bounds.xmax - origin.x << std::endl;
*/

                if (edge->vertex0() == NULL)
                {
                  direction.x *= -1;
                  direction.y *= -1;
                }
                double thetad = physycom::atan2_2pi(direction.y, direction.x);

                Point clipped;
                if ( thetad <= theta1 || thetad > theta4 )
                  clipped = Point(bounds.xmax, origin.y + ( direction.y * ( bounds.xmax - origin.x) ) / direction.x );
                else if ( thetad <= theta2 )
                  clipped = Point(origin.x + ( direction.x * ( bounds.ymax - origin.y) ) / direction.y, bounds.ymax );
                else if ( thetad <= theta3 )
                  clipped = Point(bounds.xmin, origin.y - ( direction.y * ( bounds.xmin - origin.x) ) / direction.x );
                else if ( thetad <= theta4 )
                  clipped = Point(origin.x - ( direction.x * ( bounds.ymin - origin.y) ) / direction.y, bounds.ymin );
                std::cout << "clip " << origin << " " << direction << " " << clipped << std::endl;
                std::cout << "INFINITE " << origin << " " << clipped <<  std::endl;

                if (edge->vertex0() != NULL)
                {
                  if (pg.points.size() == 0)
                    pg.points.push_back(origin);
                  if (pg.points.back() != origin)
                    pg.points.push_back(origin);
                  pg.points.push_back(clipped);
                }
                else
                {
                  pg.points.push_back(clipped);
                  pg.points.push_back(origin);
                }

              }
              edge_cnt++;
            } while (edge != cell->incident_edge());
          }
        }
        ++cell_index;
        this->cells.push_back(pg);
      }
    }
  };

} // end of namespace physycom

#endif // PHYSYCOM_UTILS_VORONOI_HPP
