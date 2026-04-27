// Created by: Alex Myasnikov
// Date: April 24, 2026

#include <memory>
#include <random>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pb_relator_map.h"
#include "pb_sample_generator.h"
#include "pb_scoring.h"
#include "pb_local_search.h"

namespace py = pybind11;
using namespace pattern_boost;

void init_pattern_boost(py::module_& parent) {
    auto pb = parent.def_submodule("pattern_boost",
        "PatternBoost: sample/score/search trivial B(2,5) words");

    // --- Factor ---
    py::class_<Factor>(pb, "Factor")
        .def(py::init<>())
        .def_readwrite("relator_idx", &Factor::relator_idx)
        .def_readwrite("conjugator",  &Factor::conjugator);

    // --- FactoredWord ---
    py::class_<FactoredWord>(pb, "FactoredWord")
        .def(py::init<>())
        .def_readwrite("factors", &FactoredWord::factors);

    // --- RelatorMap ---
    py::class_<RelatorMap>(pb, "RelatorMap")
        .def(py::init<std::vector<Word>>(), py::arg("relators"))
        .def(py::init<std::vector<std::pair<std::string, Word>>>(), py::arg("named_relators"))
        .def("size",          &RelatorMap::size)
        .def("__len__",       &RelatorMap::size)
        .def("name",          &RelatorMap::name,         py::arg("idx"))
        .def("expand_factor", &RelatorMap::expandFactor, py::arg("factor"))
        .def("expand",        &RelatorMap::expand,       py::arg("factors"));

    pb.def("build_standard_relators", &buildStandardRelators,
           py::arg("max_len"), py::arg("num_generators") = 2,
           py::call_guard<py::gil_scoped_release>());

    // --- ConjugatorLenSampler (opaque wrapper around std::function) ---
    py::class_<ConjugatorLenSampler>(pb, "ConjugatorLenSampler");
    pb.def("uniform_len",    &uniformLen,    py::arg("max_len"));
    pb.def("zero_heavy_len", &zeroHeavyLen,  py::arg("max_len"), py::arg("zero_prob") = 0.3);
    pb.def("geometric_len",  &geometricLen,  py::arg("max_len"), py::arg("p") = 0.5);
    pb.def("constant_len",   &constantLen,   py::arg("len"));

    // --- SampleConfig ---
    py::class_<SampleConfig>(pb, "SampleConfig")
        .def(py::init<>())
        .def_readwrite("min_factors",      &SampleConfig::min_factors)
        .def_readwrite("max_factors",      &SampleConfig::max_factors)
        .def_readwrite("conj_len",         &SampleConfig::conj_len)
        .def_readwrite("num_generators",   &SampleConfig::num_generators)
        .def_readwrite("min_expanded_len", &SampleConfig::min_expanded_len);

    // --- Generation (seed accepted; mt19937 created internally) ---
    pb.def("generate_word",
        [](const RelatorMap& rm, const SampleConfig& cfg, uint32_t seed) {
            std::mt19937 rng(seed);
            return generateWord(rm, cfg, rng);
        },
        py::arg("rm"), py::arg("cfg"), py::arg("seed") = 42,
        py::call_guard<py::gil_scoped_release>());

    pb.def("generate_words",
        [](const RelatorMap& rm, const SampleConfig& cfg, size_t count, uint32_t seed) {
            std::mt19937 rng(seed);
            return generateWords(rm, cfg, count, rng);
        },
        py::arg("rm"), py::arg("cfg"), py::arg("count"), py::arg("seed") = 42,
        py::call_guard<py::gil_scoped_release>());

    // --- JSON serialization ---
    pb.def("to_json",
        [](const FactoredWord& fw, const RelatorMap& rm, int version) {
            return toJson(fw, rm, version).dump();
        },
        py::arg("fw"), py::arg("rm"), py::arg("version") = 1);

    // Accept both JSON string and Python dict
    pb.def("from_json",
        [](py::object obj) {
            if (py::isinstance<py::str>(obj)) {
                return fromJson(nlohmann::json::parse(obj.cast<std::string>()));
            }
            auto json_str = py::module_::import("json")
                                .attr("dumps")(obj)
                                .cast<std::string>();
            return fromJson(nlohmann::json::parse(json_str));
        },
        py::arg("json_or_str"));

    // --- BigramFrequencies ---
    py::class_<BigramFrequencies>(pb, "BigramFrequencies")
        .def(py::init<size_t>(), py::arg("num_generators"))
        .def(py::init<const Word&, size_t>(), py::arg("word"), py::arg("num_generators"))
        .def_readwrite("num_generators", &BigramFrequencies::num_generators)
        .def_readwrite("data",           &BigramFrequencies::data);

    pb.def("hellinger_distance", &hellingerDistance, py::arg("p"), py::arg("q"));
    pb.def("hamming_distance", &hammingDistance, py::arg("a"), py::arg("b"));
    pb.def("edit_distance", &editDistance, py::arg("a"), py::arg("b"));

    pb.def("load_challenges",
        [](const std::string& path) {
            return loadChallenges(path);
        },
        py::arg("json_path"),
        py::call_guard<py::gil_scoped_release>());

    // --- WordScorer (abstract base) ---
    py::class_<WordScorer>(pb, "WordScorer");

    // --- HellingerScorer ---
    py::class_<HellingerScorer, WordScorer>(pb, "HellingerScorer")
        .def(py::init<std::vector<Word>, size_t>(),
             py::arg("challenges"), py::arg("num_generators"))
        .def("score_word",
             &HellingerScorer::scoreWord,
             py::arg("word"))
        .def("score_sample",
             &HellingerScorer::scoreSample,
             py::arg("fw"), py::arg("rm"));

    // --- HammingScorer ---
    py::class_<HammingScorer, WordScorer>(pb, "HammingScorer")
        .def(py::init<std::vector<Word>>(),
             py::arg("challenges"))
        .def("score_word",
             &HammingScorer::scoreWord,
             py::arg("word"))
        .def("score_sample",
             &HammingScorer::scoreSample,
             py::arg("fw"), py::arg("rm"));

    // --- EditScorer ---
    py::class_<EditScorer, WordScorer>(pb, "EditScorer")
        .def(py::init<std::vector<Word>>(),
             py::arg("challenges"))
        .def("score_word",
             &EditScorer::scoreWord,
             py::arg("word"))
        .def("score_sample",
             &EditScorer::scoreSample,
             py::arg("fw"), py::arg("rm"));

    // --- LocalSearchConfig ---
    py::class_<LocalSearchConfig>(pb, "LocalSearchConfig")
        .def(py::init<>())
        .def_readwrite("beam_width",         &LocalSearchConfig::beam_width)
        .def_readwrite("max_steps",          &LocalSearchConfig::max_steps)
        .def_readwrite("num_conj_samples",   &LocalSearchConfig::num_conj_samples)
        .def_readwrite("num_insert_samples", &LocalSearchConfig::num_insert_samples)
        .def_readwrite("conj_max_len",       &LocalSearchConfig::conj_max_len)
        .def_readwrite("min_factors",        &LocalSearchConfig::min_factors)
        .def_readwrite("max_factors",        &LocalSearchConfig::max_factors)
        .def_readwrite("num_generators",     &LocalSearchConfig::num_generators)
        .def("to_dict", [](const LocalSearchConfig& c) {
            py::dict d;
            d["beam_width"]         = c.beam_width;
            d["max_steps"]          = c.max_steps;
            d["num_conj_samples"]   = c.num_conj_samples;
            d["num_insert_samples"] = c.num_insert_samples;
            d["conj_max_len"]       = c.conj_max_len;
            d["min_factors"]        = c.min_factors;
            d["max_factors"]        = c.max_factors;
            d["num_generators"]     = c.num_generators;
            return d;
        })
        .def("__repr__", [](const LocalSearchConfig& c) {
            return "LocalSearchConfig(" +
                std::string("beam_width=")         + std::to_string(c.beam_width) +
                ", max_steps="                     + std::to_string(c.max_steps) +
                ", num_conj_samples="              + std::to_string(c.num_conj_samples) +
                ", num_insert_samples="            + std::to_string(c.num_insert_samples) +
                ", conj_max_len="                  + std::to_string(c.conj_max_len) +
                ", min_factors="                   + std::to_string(c.min_factors) +
                ", max_factors="                   + std::to_string(c.max_factors) +
                ", num_generators="                + std::to_string(c.num_generators) +
                ")";
        });

    // --- LocalSearcher — scorer bakes in challenges at construction ---
    py::class_<LocalSearcher>(pb, "LocalSearcher")
        .def(py::init(
            [](const RelatorMap& rm, HellingerScorer scorer, LocalSearchConfig cfg) {
                return std::make_unique<LocalSearcher>(
                    rm, std::make_unique<HellingerScorer>(std::move(scorer)), cfg);
            }),
            py::arg("rm"), py::arg("scorer"), py::arg("config") = LocalSearchConfig{})
        .def(py::init(
            [](const RelatorMap& rm, HammingScorer scorer, LocalSearchConfig cfg) {
                return std::make_unique<LocalSearcher>(
                    rm, std::make_unique<HammingScorer>(std::move(scorer)), cfg);
            }),
            py::arg("rm"), py::arg("scorer"), py::arg("config") = LocalSearchConfig{})
        .def(py::init(
            [](const RelatorMap& rm, EditScorer scorer, LocalSearchConfig cfg) {
                return std::make_unique<LocalSearcher>(
                    rm, std::make_unique<EditScorer>(std::move(scorer)), cfg);
            }),
            py::arg("rm"), py::arg("scorer"), py::arg("config") = LocalSearchConfig{})
        .def("score_word", &LocalSearcher::scoreWord, py::arg("word"))
        .def("neighbors",
            [](const LocalSearcher& ls, const FactoredWord& fw, uint32_t seed) {
                std::mt19937 rng(seed);
                return ls.neighbors(fw, rng);
            },
            py::arg("fw"), py::arg("seed") = 42,
            py::call_guard<py::gil_scoped_release>())
        .def("search",
            [](const LocalSearcher& ls, const FactoredWord& initial, uint32_t seed) {
                std::mt19937 rng(seed);
                return ls.search(initial, rng);
            },
            py::arg("initial"), py::arg("seed") = 42,
            py::call_guard<py::gil_scoped_release>());
}
