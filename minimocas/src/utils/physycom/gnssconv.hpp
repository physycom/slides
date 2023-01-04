/* Copyright 2015-2017 - Alessandro Fabbri, Stefano Sinigardi */

// Distributed under the Boost Software License, Version 1.0.
// See accompanying file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt


#ifndef PHYSYCOM_UTILS_GNSSCONV_HPP
#define PHYSYCOM_UTILS_GNSSCONV_HPP

#include <iostream>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950  // pi :)
#endif

namespace physycom{

  class Gnss_Coordinate{
  public:
    double deg;
    double rad;
    double iso6709;
    int dms[3];
    void Deg_To_Rad(){
      rad = deg * M_PI / 180;
    }
    void Rad_To_Deg(){
      deg = rad / M_PI * 180;
    }
    void Deg_To_DMS(){
      dms[0] = (int) deg;
      dms[1] = (int) ( (deg-dms[0])*60 );
      dms[2] = (int) ( ( (deg-dms[0])*60-dms[1] )*60 );
    }
    void DMS_To_Deg(){
      deg = dms[0]+dms[1]/60.+dms[2]/3600.;
    }
    void Deg_To_Iso6709(){
      iso6709 = dms[0]*100+(deg-dms[0])*60;
    }
    void Iso6709_To_DMS(){
      dms[0] = (int) (iso6709 / 1e2);
      dms[1] = (int) (iso6709 - dms[0]*1e2);
      dms[2] = (int) (( iso6709 - floor(iso6709))*60);
    }
    Gnss_Coordinate(){
      deg = 0.0;
      Deg_To_Rad();
      Deg_To_DMS();
      Deg_To_Iso6709();    
    }
    Gnss_Coordinate(int deg_int, int min_int, int sec_int){ 
      dms[0] = deg_int;
      dms[1] = min_int;
      dms[2] = sec_int;
    }
    /* Constructor's flag 
     * 'd' = degree  = dd.dddddd
     * 'r' = radians = rr.rrrrrr
     * 'i' = iso6709 = ddmm.mmmm
     */
    Gnss_Coordinate(double number, char flag = 'd'){ 
      switch(flag){
        case 'd':
          deg = number;
          Deg_To_Rad();
          Deg_To_DMS();
          Deg_To_Iso6709();
          break;
        case 'r':
          rad = number;
          Rad_To_Deg();
          Deg_To_DMS();
          Deg_To_Iso6709();
          break;
        case 'i':
          iso6709 = number;
          Iso6709_To_DMS();
          DMS_To_Deg();      
          Deg_To_Rad();
          break;
        default:
          std::cerr << "Encoding not found. Default to zero." << std::endl;
          deg = 0.0;
          Deg_To_Rad();
          Deg_To_DMS();
          Deg_To_Iso6709();      
          break;
      }
    }
  };

} // end of namespace physycom

#endif // PHYSYCOM_UTILS_GNSSCONV_HPP
