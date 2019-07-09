#include <glog/logging.h>
#include <math.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace replay {

namespace {
const std::string kPreText = ":\t";
const std::string kProgressSymbol = "#";
const std::string kFillSymbol = " ";
const std::string kPostText = "]";
const size_t kProgressBarWidth = 50u;
const double kProgressBarWidthFloatingPoint =
    static_cast<double>(kProgressBarWidth);
}  // namespace

void PrintProgress(const size_t current, const size_t maximum,
                   const std::string& pre_message,
                   const std::string& post_message) {
  double num_progress_symbols = kProgressBarWidthFloatingPoint *
                                static_cast<double>(current) /
                                static_cast<double>(maximum);
  double percentage = 100.0 * static_cast<double>(current) / maximum;

  size_t num_progress_symbols_integer = floor(num_progress_symbols);
  CHECK_LE(num_progress_symbols_integer, kProgressBarWidth)
      << "Number of "
      << "progress bar symbols is greater than the progress bar width. Make "
      << "sure the number of processed elements (" << current
      << ") is at most equal to the total number of elements to be processed ("
      << maximum << ")";

  std::stringstream ss;
  ss << "\033[34m" << (pre_message.empty() ? "Progress" : pre_message)
     << "\033[0m" << kPreText << std::setfill(' ') << std::fixed << std::setw(5)
     << std::setprecision(1) << percentage << "% [";
  for (size_t idx = 0; idx < kProgressBarWidth; ++idx) {
    if (idx < num_progress_symbols_integer) {
      ss << kProgressSymbol;
    } else {
      ss << kFillSymbol;
    }
  }
  ss << kPostText << " " << post_message
     << (current == maximum
             ? " [\033[32mComplete!\033[0m]"
             : "");

  if (current > 1) {
    // LOG(INFO) << "\33[2K\033[A\33[2K\033[A\33[2K\r";
    LOG(INFO) << "\33[2K\033[A\33[2K\033[A";
  }
  LOG(INFO) << ss.str();
}

}  // namespace replay
