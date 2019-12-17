#include <gflags/gflags.h>
#include <gflags/gflags_completions.h>
#include <glog/logging.h>
#include <ostream>

#include <gtest/gtest.h>

DEFINE_string(test_datadir, "", "The location of the test data.");

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  ::testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  LOG(INFO) << "Running tests with logs enabled.";
  return RUN_ALL_TESTS();
}
