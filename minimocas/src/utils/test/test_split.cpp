/* Copyright 2017 - Alessandro Fabbri, Stefano Sinigardi */

// Distributed under the Boost Software License, Version 1.0.
// See accompanying file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt


#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include <boost/algorithm/string.hpp>
#include <chrono>
#include "physycom/string.hpp"

using namespace std;
using namespace physycom;


// Timing template function
template<typename TimeT>
struct elapsed_time
{
  template<typename F, typename ...Args>
  static typename TimeT::rep exec(F&& func, Args&&... args)
  {
    auto start = std::chrono::steady_clock::now();
    std::forward<decltype(func)>(func)(std::forward<Args>(args)...);
    auto duration = std::chrono::duration_cast<TimeT> (std::chrono::steady_clock::now() - start);
    return duration.count();
  }
};
typedef  elapsed_time<std::chrono::microseconds> time_us;


TEST_CASE( "Timing " ) {
	constexpr size_t num_passes=10000;
	time_t boost_splitting_time;
	time_t physycom_splitting_time;
	string line;
	string separators;
	std::vector<std::string> tokens, result;

    boost_splitting_time = 0;
    physycom_splitting_time = 0;
    line = "string\tto\tsplit\tbecause\twe\twant\tto";
	result = vector<string>({"string", "to", "split","because", "we", "want","to"});
    separators = "\t";
    SECTION( "Simple case, compress on, single separator\nstring = " + line + "\nseparators = " + separators ) {
        for (size_t i=0; i<num_passes; i++) {
            boost_splitting_time += time_us::exec(boost::split<std::vector<std::string>, std::string, decltype(boost::is_any_of(separators))>, tokens, line, boost::is_any_of(separators), boost::token_compress_on);
            physycom_splitting_time += time_us::exec(split<std::string>, tokens, line, separators, token_compress_on);
        }
		REQUIRE( physycom_splitting_time <= boost_splitting_time );
    }

    boost_splitting_time = 0;
    physycom_splitting_time = 0;
    line = "string\tto\tsplit\tbecause\twe\twant\tto";
	result = vector<string>({"string", "to", "split","because", "we", "want","to"});
    separators = "\t";
    SECTION( "Simple case, compress off, single separator\nstring = " + line + "\nseparators = " + separators ) {
        for (size_t i=0; i<num_passes; i++) {
            boost_splitting_time += time_us::exec(boost::split<std::vector<std::string>, std::string, decltype(boost::is_any_of(separators))>, tokens, line, boost::is_any_of(separators), boost::token_compress_off);
            physycom_splitting_time += time_us::exec(split<std::string>, tokens, line, separators, token_compress_off);
        }
		REQUIRE( physycom_splitting_time <= boost_splitting_time );
    }

    boost_splitting_time = 0;
    physycom_splitting_time = 0;
    line = "string\t\tto\tsplit\t\tbecause\twe\t\twant\tto\t\t";
	result = vector<string>({"string", "to", "split","because", "we", "want","to"});
    separators = "\t";
    SECTION( "Simple case, compress on, single separator\nstring = " + line + "\nseparators = " + separators ) {
        for (size_t i=0; i<num_passes; i++) {
            boost_splitting_time += time_us::exec(boost::split<std::vector<std::string>, std::string, decltype(boost::is_any_of(separators))>, tokens, line, boost::is_any_of(separators), boost::token_compress_on);
            physycom_splitting_time += time_us::exec(split<std::string>, tokens, line, separators, token_compress_on);
        }
		REQUIRE( physycom_splitting_time <= boost_splitting_time );
    }

    boost_splitting_time = 0;
    physycom_splitting_time = 0;
    line = "string\t\tto\tsplit\t\tbecause\twe\t\twant\tto\t\t";
	result = vector<string>({"string", "", "to", "split","", "because", "we", "", "want","to", "", ""});
    separators = "\t";
    SECTION( "Simple case, compress off, single separator\nstring = " + line + "\nseparators = " + separators ) {
        for (size_t i=0; i<num_passes; i++) {
            boost_splitting_time += time_us::exec(boost::split<std::vector<std::string>, std::string, decltype(boost::is_any_of(separators))>, tokens, line, boost::is_any_of(separators), boost::token_compress_off);
            physycom_splitting_time += time_us::exec(split<std::string>, tokens, line, separators, token_compress_off);
        }
		REQUIRE( physycom_splitting_time <= boost_splitting_time );
    }

    boost_splitting_time = 0;
    physycom_splitting_time = 0;
    line = "\t\t\t\t\t\t\t\t\t\t";
	result = vector<string>(10,std::string(""));
    separators = "\t";
    SECTION( "Simple case, compress on, single separator\nstring = " + line + "\nseparators = " + separators ) {
        for (size_t i=0; i<num_passes; i++) {
            boost_splitting_time += time_us::exec(boost::split<std::vector<std::string>, std::string, decltype(boost::is_any_of(separators))>, tokens, line, boost::is_any_of(separators), boost::token_compress_on);
            physycom_splitting_time += time_us::exec(split<std::string>, tokens, line, separators, token_compress_on);
        }
		REQUIRE( physycom_splitting_time <= boost_splitting_time );
    }

}

TEST_CASE( "Correctness " ) {
	string line;
	string separators;
	std::vector<std::string> tokens, result;

	line = "string_to_split";
	separators = "_";
    SECTION( "Simple case, compress off, single separator\nstring = " + line + "\nseparators = " + separators ) {
		result = vector<string>({"string", "to", "split"});
		split(tokens, line, separators);

		REQUIRE( tokens.size() == result.size() );
		for(size_t i=0; i<tokens.size(); i++){
			REQUIRE( tokens[i] == result[i] );
		}
	}
	line = "__string_to__split__";
	separators = "_";
    SECTION( "Empty tokens, compress off, single separator\nstring = " + line + "\nseparators = " + separators ) {
		result = vector<string>({"", "", "string", "to", "", "split", "", ""});
		split(tokens, line, separators);

		REQUIRE( tokens.size() == result.size() );
		for(size_t i=0; i<tokens.size(); i++){
			REQUIRE( tokens[i] == result[i] );
		}
    }
	line = "string_to_split";
	separators = "_";
    SECTION( "Simple case, compress on, single separator\nstring = " + line + "\nseparators = " + separators ) {
		result = vector<string>({"string", "to", "split"});
		split(tokens, line, separators, token_compress_on);

		REQUIRE( tokens.size() == result.size() );
		for(size_t i=0; i<tokens.size(); i++){
			REQUIRE( tokens[i] == result[i] );
		}
	}
	line = "__string_to__split__";
	separators = "_";
    SECTION( "Empty tokens, compress on, single separator\nstring = " + line + "\nseparators = " + separators ) {
		result = vector<string>({"string", "to", "split"});
		split(tokens, line, separators, token_compress_on);

		REQUIRE( tokens.size() == result.size() );
		for(size_t i=0; i<tokens.size(); i++){
			REQUIRE( tokens[i] == result[i] );
		}
	}
}
