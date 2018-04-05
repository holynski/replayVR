#include <chrono>
#include "glog/logging.h"
#include "replay/util/timer.h"

namespace replay {

SimpleTimer::SimpleTimer() { last_time_ = std::chrono::system_clock::now(); }

double SimpleTimer::ElapsedTime() {
  std::chrono::time_point<std::chrono::system_clock> new_time =
      std::chrono::system_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                       new_time - last_time_)
                       .count();
  last_time_ = new_time;
  return elapsed;
}

Stopwatch::Stopwatch() : count_(0), running_(false) {}

void Stopwatch::Start() {
  CHECK(!running_) << "Called Start() twice!";
  running_ = true;
  last_start_time_ = std::chrono::system_clock::now();
}

void Stopwatch::Stop() {
  CHECK(running_) << "Called Stop() twice, or didn't call Start()!";
  running_ = false;
  count_ += std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now() - last_start_time_)
                .count();
}

void Stopwatch::Clear() {
  running_ = false;
  count_ = 0;
}

double Stopwatch::Count() const {
  CHECK(!running_) << "Must call Stop() before Count()!";
  return count_;
}

}  // namespace replay
