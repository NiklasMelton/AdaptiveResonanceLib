#pragma once

#include <algorithm>
#include <cmath>
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

template <typename Score>
void descending_order_into(const std::vector<Score>& scores, std::vector<std::size_t>& order) {
    order.resize(scores.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        if (scores[a] != scores[b])
            return scores[a] > scores[b];
        return a < b;
    });
}

// Reused between samples so the search does not allocate on every call.
template <typename Score> struct LazySearchScratch {
    std::vector<Score> scores;
    std::vector<std::size_t> order;
};

// Same search as fit_one with descending_order ranking, but the match is computed
// only for the candidates actually visited. ``choice`` may cache work (such as the
// fuzzy AND sum) for ``match`` to reuse, since match(index) always follows
// choice(index) for the same sample.
template <typename State, typename Choice, typename Match, typename Pass, typename Accept,
          typename Track, typename Learn, typename Create>
std::size_t fit_one_lazy(State state, Choice choice, Match match, Pass pass, Accept accept,
                         Track track, Learn learn, Create create,
                         LazySearchScratch<std::invoke_result_t<Choice, std::size_t>>& scratch) {
    if (state.dimension == 0)
        throw std::invalid_argument("feature dimension must be positive");
    const std::size_t count = state.weights.size();
    if (count == 0)
        return create();

    auto& scores = scratch.scores;
    scores.resize(count);
    for (std::size_t index = 0; index < count; ++index)
        scores[index] = choice(index);
    descending_order_into(scores, scratch.order);

    for (std::size_t index : scratch.order) {
        const auto value = match(index);
        if (!pass(value))
            continue;
        if (!accept(index)) {
            if (!track(value))
                break;
            continue;
        }
        learn(index);
        return index;
    }
    return create();
}

// True when match tracking can only raise the vigilance during one sample's
// search: then a category whose match fails the starting vigilance fails every
// later check too, and can be dropped before the search sorts the candidates.
inline bool vigilance_only_rises(MatchTracking mode, double epsilon) {
    switch (mode) {
    case MatchTracking::Plus:
        return epsilon >= 0.0; // vigilance = match + epsilon, and match >= vigilance
    case MatchTracking::Minus:
        return epsilon <= 0.0; // vigilance = match - epsilon
    case MatchTracking::Zero:  // vigilance = match, and match > vigilance
    case MatchTracking::One:   // the search stops
    case MatchTracking::Leave: // vigilance is unchanged
        return true;
    }
    return false;
}

// The search of fit_one_lazy, given the choice score and match value of every
// one of ``count`` categories (computed in bulk, so match no longer has to be
// lazy). Categories are visited by descending score, ties to the lowest index.
//
// With ``prefilter`` (see vigilance_only_rises), only categories that pass the
// starting vigilance are sorted: the others would be skipped wherever they fell
// in the order, and with no NaN score the order is a strict total order, so the
// visited categories come in the same order as before. A NaN score makes the
// comparison inconsistent, so then every category is sorted exactly as
// fit_one_lazy sorts them.
template <typename Pass, typename Accept, typename Track, typename Learn, typename Create>
std::size_t fit_one_scored(std::size_t count, const double* scores, const double* matches,
                           bool prefilter, Pass pass, Accept accept, Track track, Learn learn,
                           Create create, std::vector<std::size_t>& order) {
    if (count == 0)
        return create();

    auto descending = [scores](std::size_t a, std::size_t b) {
        if (scores[a] != scores[b])
            return scores[a] > scores[b];
        return a < b;
    };
    bool filter = prefilter;
    for (std::size_t index = 0; filter && index < count; ++index)
        filter = !std::isnan(scores[index]);
    if (filter) {
        order.clear();
        for (std::size_t index = 0; index < count; ++index) {
            if (pass(matches[index]))
                order.push_back(index);
        }
    } else {
        order.resize(count);
        std::iota(order.begin(), order.end(), std::size_t{0});
    }
    std::sort(order.begin(), order.end(), descending);

    for (std::size_t index : order) {
        const double value = matches[index];
        if (!pass(value))
            continue;
        if (!accept(index)) {
            if (!track(value))
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
