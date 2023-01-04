TEST_SPLIT_EXE=test_split.exe
TEST_GNSS_EXE=test_gnssconv.exe

all: dirtree test_gnss test_split

test: test_gnss test_split
	./bin/$(TEST_SPLIT_EXE)
	./bin/$(TEST_GNSS_EXE)

test_gnss: dirtree physycom/gnssconv.hpp test/test_gnssconv.cpp
	$(CXX) -std=c++11 -I. -o bin/$(TEST_GNSS_EXE) test/test_gnssconv.cpp 

test_split: dirtree physycom/split.hpp test/test_split.cpp
	$(CXX) -std=c++11 -I. -o bin/$(TEST_SPLIT_EXE) test/test_split.cpp 

dirtree:
	@mkdir -p bin

clean:
	rm -f bin/*.exe

cleanall:
	rm -rf bin obj
