PROJECT(GraphPartitioner)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
SET(CMAKE_SOURCE_DIR .)

#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lboost_program_options -lboost_filesystem -lboost_random -lboost_iostreams -ltcmalloc")

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -m64 ")

SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS} $ENV{CXXFLAGS} -O0 -Wall -g -ggdb -std=c++0x")
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS} $ENV{CXXFLAGS} -O3 -Wall -std=c++0x")

set(BOOST_LIBS_DIR /home/xcgoner/usr/lib)
set(Boost_LIBRARIES 
  ${BOOST_LIBS_DIR}/libboost_filesystem.a 
  ${BOOST_LIBS_DIR}/libboost_program_options.a
  ${BOOST_LIBS_DIR}/libboost_system.a 
  ${BOOST_LIBS_DIR}/libboost_iostreams.a 
  ${BOOST_LIBS_DIR}/libboost_date_time.a)

SET(SRC_LIST GraphPartitioner.cpp partition_by_patoh.hpp partition_strategy.hpp basic_graph.hpp builtin_parsers.hpp fs_util.cpp graph_basic_types.hpp memory_info.cpp util.hpp)
#SET(SRC_LIST GraphPartitioner.cpp basic_graph.hpp builtin_parsers.hpp graph_basic_types.hpp)
ADD_EXECUTABLE(demo ${SRC_LIST})
include_directories(/home/xcgoner/usr/include .)
TARGET_LINK_LIBRARIES( demo ${Boost_LIBRARIES} /home/xcgoner/usr/lib/libpatoh.a)