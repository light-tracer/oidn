#include <gtest/gtest.h>

class SkipListener : public ::testing::EmptyTestEventListener {
public:
  void OnTestPartResult(const ::testing::TestPartResult& result) override {
    if (result.type() == ::testing::TestPartResult::kSkip)
      skipped = true;
  }
  static bool skipped;
};

bool SkipListener::skipped = false;

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::UnitTest::GetInstance()->listeners().Append(new SkipListener);
  int ret = RUN_ALL_TESTS();
  if (ret == 0 && SkipListener::skipped)
    return 77; // special code for skipped
  return ret;
}
