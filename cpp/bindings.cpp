// Created by: Alex Myasnikov
// Date: March 9, 2026

#include <optional>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "word.h"
#include "word_problem.h"
#include "logger.h"

namespace py = pybind11;

PYBIND11_MODULE(tcgraph_ext, m) {
    m.doc() = "tcgraph C++ bindings for B(2,5) computations";

    // Flush spdlog after every message so output appears in real time
    tcgraph::getLogger().get()->flush_on(spdlog::level::trace);

    py::class_<Word>(m, "Word")
        .def(py::init<>())
        .def(py::init<std::vector<int>>())
        .def(py::init<int>())
        .def("get_letters", &Word::getLetters)
        .def("size", &Word::size)
        .def("reduce", &Word::reduce)
        .def("is_cyclically_reduced", &Word::isCyclicallyReduced)
        .def("to_string", &Word::toString)
        .def("to_gap_string", &Word::toGapString, py::arg("x"))
        .def("__len__", &Word::size)
        .def("__getitem__", &Word::operator[])
        .def("__mul__", &Word::operator*)
        .def("__neg__", [](const Word& w) { return -w; })
        .def("__pow__", [](const Word& w, int p) { return w ^ p; })
        .def("__repr__", &Word::toCString)
        .def("__str__", &Word::toString)
        .def("__eq__", &Word::operator==);

    m.def("construct_cayley_approximation",
          &constructCayleyApproximationForB_2_5,
          py::arg("word"),
          py::arg("coset_bound") = 20000000,
          py::call_guard<py::gil_scoped_release>(),
          R"doc(
Test if word is trivial in B(2,5) using Todd-Coxeter approximation.
Returns: True (trivial), False (non-trivial), None (inconclusive).
          )doc");
}
