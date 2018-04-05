#pragma once

#include <chrono>

namespace replay {

// A simple timer.
// The first call to ElapsedTime() will return the time since the constructor
// was called. Subsequent calls to ElapsedTime() will return the time since the
// previous call to ElapsedTime(). All times are in milliseconds.
//
// Sample usage:
//
// SimpleTimer timer;
// A: ** do something here **
// timer.ElapsedTime(); // returns the duration of A
// B: ** do something here **
// C: ** do something here **
// timer.ElapsedTime(); // returns the duration of B
class SimpleTimer {
 public:
  SimpleTimer();
  double ElapsedTime();

 private:
  std::chrono::time_point<std::chrono::system_clock> last_time_;
};

// A cumulative stopwatch, i.e. the amount of time between Start() and Stop()
// calls will be added up and returned with Count(), until Clear() is called.
// Count() will fail if the stopwatch is currently running, that is, if Stop()
// has not been called. All times are in milliseconds.
//
// Sample usage:
//
// Stopwatch watch;
// watch.Start();
// A: **do something you want to measure here**
// watch.Stop();
// B: **do something you don't want to measure here**
// watch.Start();
// C: **do something you want to measure here**
// watch.Stop();
// int duration = watch.Count(); // returns the value of A + C
//
class Stopwatch {
 public:
  Stopwatch();
  void Start();
  void Stop();
  void Clear();
  double Count() const;

 private:
  double count_;
  std::chrono::time_point<std::chrono::system_clock> last_start_time_;
  bool running_;
};

}  // namespace replay
