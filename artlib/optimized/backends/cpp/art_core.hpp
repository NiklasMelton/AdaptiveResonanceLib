#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace art_core {

struct NoLabels {};

// A non-owning view lets every model keep its native weight type while the search
// engine and the Python boundary use the same state layout.
template <typename Weights, typename Labels, typename Dimension> struct StateView {
    Weights& weights;
    Labels& labels;
    Dimension& dimension;
};

template <typename Weights, typename Labels, typename Dimension>
StateView<Weights, Labels, Dimension> state_view(Weights& weights, Labels& labels,
                                                 Dimension& dimension) {
    return {weights, labels, dimension};
}

enum class MatchTracking { Plus, Minus, Zero, One, Leave };

inline MatchTracking parse_match_tracking(const std::string& name) {
    if (name == "MT+")
        return MatchTracking::Plus;
    if (name == "MT-")
        return MatchTracking::Minus;
    if (name == "MT0")
        return MatchTracking::Zero;
    if (name == "MT1")
        return MatchTracking::One;
    if (name == "MT~")
        return MatchTracking::Leave;
    throw std::invalid_argument("Invalid match tracking mode: " + name);
}

inline bool passes_match(double match, double vigilance, MatchTracking mode) {
    return (mode == MatchTracking::Zero || mode == MatchTracking::Leave) ? match > vigilance
                                                                         : match >= vigilance;
}

inline bool track_match(double match, double& vigilance, double epsilon, MatchTracking mode) {
    switch (mode) {
    case MatchTracking::Plus:
        vigilance = match + epsilon;
        break;
    case MatchTracking::Minus:
        vigilance = match - epsilon;
        break;
    case MatchTracking::Zero:
        vigilance = match;
        break;
    case MatchTracking::One:
        return false;
    case MatchTracking::Leave:
        break;
    }
    return vigilance <= 1.0;
}

inline bool passes_match(std::uint32_t match, std::uint32_t vigilance, MatchTracking mode) {
    return (mode == MatchTracking::Zero || mode == MatchTracking::Leave) ? match > vigilance
                                                                         : match >= vigilance;
}

inline bool track_match(std::uint32_t match, std::uint32_t& vigilance, std::uint32_t epsilon,
                        std::uint32_t maximum, MatchTracking mode) {
    switch (mode) {
    case MatchTracking::Plus:
        vigilance = (epsilon > maximum - std::min(match, maximum)) ? maximum + 1 : match + epsilon;
        break;
    case MatchTracking::Minus:
        vigilance = match > epsilon ? match - epsilon : 0;
        break;
    case MatchTracking::Zero:
        vigilance = match;
        break;
    case MatchTracking::One:
        return false;
    case MatchTracking::Leave:
        break;
    }
    return vigilance <= maximum;
}

template <typename Score>
std::vector<std::size_t> descending_order(const std::vector<Score>& scores) {
    std::vector<std::size_t> order(scores.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        if (scores[a] != scores[b])
            return scores[a] > scores[b];
        return a < b;
    });
    return order;
}

// Policy callbacks are templates, so the hot path remains inlineable. The ranking
// callback can use floating point scores or exact fraction ordering.
template <typename Score, typename Match> struct SearchScratch {
    std::vector<Score> scores;
    std::vector<Match> matches;
};

template <typename State, typename Choice, typename Match, typename Rank, typename Pass,
          typename Accept, typename Track, typename Learn, typename Create>
std::size_t fit_one_cached(State state, Choice choice, Match match, Rank rank, Pass pass,
                           Accept accept, Track track, Learn learn, Create create,
                           SearchScratch<std::invoke_result_t<Choice, std::size_t>,
                                         std::invoke_result_t<Match, std::size_t>>& scratch) {
    if (state.dimension == 0)
        throw std::invalid_argument("feature dimension must be positive");
    const std::size_t count = state.weights.size();
    if (count == 0)
        return create();

    auto& scores = scratch.scores;
    auto& matches = scratch.matches;
    scores.clear();
    matches.clear();
    scores.reserve(count);
    matches.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        scores.push_back(choice(index));
        matches.push_back(match(index));
    }

    for (std::size_t index : rank(scores, matches)) {
        if (!pass(matches[index]))
            continue;
        if (!accept(index)) {
            if (!track(matches[index]))
                break;
            continue;
        }
        learn(index);
        return index;
    }
    return create();
}

template <typename State, typename Choice, typename Match, typename Rank, typename Pass,
          typename Accept, typename Track, typename Learn, typename Create>
std::size_t fit_one(State state, Choice choice, Match match, Rank rank, Pass pass, Accept accept,
                    Track track, Learn learn, Create create) {
    SearchScratch<std::invoke_result_t<Choice, std::size_t>,
                  std::invoke_result_t<Match, std::size_t>>
        scratch;
    return fit_one_cached(state, choice, match, rank, pass, accept, track, learn, create, scratch);
}

template <typename State, typename Choice, typename Better>
std::size_t predict_one(State state, Choice choice, Better better) {
    if (state.weights.empty())
        throw std::runtime_error("Model has no clusters");
    if (state.dimension == 0)
        throw std::invalid_argument("feature dimension must be positive");
    std::size_t best = 0;
    auto best_score = choice(0);
    for (std::size_t index = 1; index < state.weights.size(); ++index) {
        auto score = choice(index);
        if (better(score, best_score)) {
            best = index;
            best_score = score;
        }
    }
    return best;
}

} // namespace art_core
