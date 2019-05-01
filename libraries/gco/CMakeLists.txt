project(gco)

include_directories(./)
set(gco_sources
  GCoptimization.cpp
  LinkedBlockList.cpp
  graph.cpp
  maxflow.cpp)

add_library(gco STATIC ${gco_sources})
