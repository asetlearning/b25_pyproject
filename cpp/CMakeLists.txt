pybind11_add_module(tcgraph_ext bindings.cpp bindings_pattern_boost.cpp)
target_link_libraries(tcgraph_ext PRIVATE tc_graph pattern_boost)

install(TARGETS tcgraph_ext DESTINATION .)
