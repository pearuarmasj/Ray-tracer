/**
 * @file profiler.hpp
 * @brief Simple profiling utilities for performance analysis
 * 
 * Thread-safe timing instrumentation to identify bottlenecks.
 * Enable with ENABLE_PROFILING define or via CMake.
 */

#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <atomic>
#include <mutex>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

namespace raytracer {

/**
 * @brief Performance profiler with per-category timing
 * 
 * Usage:
 *   Profiler::instance().start("category");
 *   // ... work ...
 *   Profiler::instance().stop("category");
 * 
 * Or use RAII:
 *   { ProfileScope scope("category"); ... }
 */
class Profiler {
public:
    using Clock = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<double, std::milli>;
    
    struct Stats {
        std::atomic<uint64_t> total_ns{0};      // Total time in nanoseconds
        std::atomic<uint64_t> call_count{0};    // Number of calls
    };
    
    static Profiler& instance() {
        static Profiler inst;
        return inst;
    }
    
    /**
     * @brief Reset all timers
     */
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.clear();
        enabled_ = true;
    }
    
    /**
     * @brief Enable/disable profiling (disabled = zero overhead)
     */
    void set_enabled(bool enabled) { enabled_ = enabled; }
    bool is_enabled() const { return enabled_; }
    
    /**
     * @brief Record a timing manually (thread-safe)
     */
    void record(const std::string& category, Duration duration) {
        if (!enabled_) return;
        
        auto ns = static_cast<uint64_t>(duration.count() * 1e6);
        get_stats(category).total_ns.fetch_add(ns, std::memory_order_relaxed);
        get_stats(category).call_count.fetch_add(1, std::memory_order_relaxed);
    }
    
    /**
     * @brief Get stats for a category
     */
    Stats& get_stats(const std::string& category) {
        std::lock_guard<std::mutex> lock(mutex_);
        return stats_[category];
    }
    
    /**
     * @brief Print profiling report
     */
    void report() const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (stats_.empty()) {
            std::cout << "Profiler: No data collected.\n";
            return;
        }
        
        // Collect and sort by total time
        std::vector<std::pair<std::string, const Stats*>> sorted;
        for (const auto& [name, stats] : stats_) {
            sorted.emplace_back(name, &stats);
        }
        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
            return a.second->total_ns.load() > b.second->total_ns.load();
        });
        
        // Calculate total time for percentage
        uint64_t grand_total_ns = 0;
        for (const auto& [name, stats] : sorted) {
            grand_total_ns += stats->total_ns.load();
        }
        
        std::cout << "\n===== PERFORMANCE PROFILE =====\n";
        std::cout << std::left << std::setw(25) << "Category"
                  << std::right << std::setw(12) << "Total (ms)"
                  << std::setw(10) << "Calls"
                  << std::setw(12) << "Avg (us)"
                  << std::setw(8) << "%" << "\n";
        std::cout << std::string(67, '-') << "\n";
        
        for (const auto& [name, stats] : sorted) {
            uint64_t total_ns = stats->total_ns.load();
            uint64_t calls = stats->call_count.load();
            double total_ms = total_ns / 1e6;
            double avg_us = calls > 0 ? (total_ns / 1e3) / calls : 0.0;
            double pct = grand_total_ns > 0 ? (100.0 * total_ns / grand_total_ns) : 0.0;
            
            std::cout << std::left << std::setw(25) << name
                      << std::right << std::fixed << std::setprecision(1)
                      << std::setw(12) << total_ms
                      << std::setw(10) << calls
                      << std::setprecision(2) << std::setw(12) << avg_us
                      << std::setprecision(1) << std::setw(7) << pct << "%\n";
        }
        
        std::cout << std::string(67, '-') << "\n";
        std::cout << "Grand total: " << std::fixed << std::setprecision(1) 
                  << (grand_total_ns / 1e6) << " ms\n";
        std::cout << "===============================\n\n";
    }
    
private:
    Profiler() = default;
    Profiler(const Profiler&) = delete;
    Profiler& operator=(const Profiler&) = delete;
    
    mutable std::mutex mutex_;
    std::unordered_map<std::string, Stats> stats_;
    bool enabled_ = true;
};

/**
 * @brief RAII scope timer
 */
class ProfileScope {
public:
    explicit ProfileScope(const std::string& category) 
        : category_(category), start_(Profiler::Clock::now()) {}
    
    ~ProfileScope() {
        auto end = Profiler::Clock::now();
        Profiler::Duration duration = end - start_;
        Profiler::instance().record(category_, duration);
    }
    
private:
    std::string category_;
    Profiler::Clock::time_point start_;
};

/**
 * @brief Simple one-shot timer for larger sections
 */
class Timer {
public:
    using Clock = std::chrono::high_resolution_clock;
    
    Timer() : start_(Clock::now()) {}
    
    void reset() { start_ = Clock::now(); }
    
    double elapsed_ms() const {
        auto end = Clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
    
    double elapsed_sec() const {
        return elapsed_ms() / 1000.0;
    }
    
private:
    Clock::time_point start_;
};

// Convenience macros (can be disabled for zero overhead)
#ifdef ENABLE_PROFILING
    #define PROFILE_SCOPE(name) raytracer::ProfileScope _profile_##__LINE__(name)
    #define PROFILE_FUNCTION() raytracer::ProfileScope _profile_func_(__func__)
#else
    #define PROFILE_SCOPE(name) ((void)0)
    #define PROFILE_FUNCTION() ((void)0)
#endif

} // namespace raytracer
