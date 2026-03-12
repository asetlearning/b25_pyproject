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

// Must be at module scope so the Python exception type lives long enough
static py::exception<ShorterLengthFound>* py_shorter_length_found = nullptr;

PYBIND11_MODULE(tcgraph_ext, m) {
    m.doc() = "tcgraph C++ bindings for B(2,5) computations";

    // Flush spdlog after every message so output appears in real time
    tcgraph::getLogger().get()->flush_on(spdlog::level::trace);

    // Expose ShorterLengthFound so Python callers can catch it specifically
    py_shorter_length_found = new py::exception<ShorterLengthFound>(m, "ShorterLengthFound");

    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const ShorterLengthFound& e) {
            // Include both the message and the reduced word in the exception text
            std::string msg = std::string(e.what()) + " | word: " + e.getWord().toString();
            py_shorter_length_found->operator()(msg.c_str());
        } catch (const std::exception& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        } catch (...) {
            PyErr_SetString(PyExc_RuntimeError, "unknown C++ exception");
        }
    });

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

    m.def("attempt_to_decide_if_trivial",
          &attemptToDecideIfTrivial_in_B_2_5,
          py::arg("word"),
          py::arg("coset_bound") = 2000000,
          py::call_guard<py::gil_scoped_release>(),
          R"doc(
Attempt to decide if a word is trivial in B(2,5) by enumerating relators.
Returns: True (trivial), False (non-trivial), None (inconclusive).
          )doc");

    m.def("attempt_to_reduce_using_relations",
          [](Word w, std::vector<Word> relators) {
              return attemptToReduceUsingRelations(w, relators);
          },
          py::arg("word"),
          py::arg("relators"),
          py::call_guard<py::gil_scoped_release>(),
          R"doc(
Attempt to reduce word using the given relators.
Returns: True (trivial), False (non-trivial), None (inconclusive).
          )doc");

    m.def("attempt_to_reduce_using_relations_greedy",
          [](Word w, std::vector<Word> relators,
             const std::string& word_name,
             std::optional<std::string> check_point_file_name) {

              return attemptToReduceUsingRelationsGreedy(w, relators, word_name, check_point_file_name);
          },
          py::arg("word"),
          py::arg("relators"),
          py::arg("word_name") = "R",
          py::arg("check_point_file_name") = std::nullopt,
          py::call_guard<py::gil_scoped_release>(),
          R"doc(
Attempt to reduce word using the given relators (greedy strategy).
Returns: True (trivial), False (non-trivial), None (inconclusive).
Raises ShorterLengthFound if a shorter equivalent word is discovered.
          )doc");
}
