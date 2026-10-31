// sherpa-onnx/csrc/offline-stream-test.cc
//
// Copyright (c)  2026  mitsu-h

#include "sherpa-onnx/csrc/offline-stream.h"

#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace sherpa_onnx {

static std::vector<OfflineRecognitionHypothesis> MakeHypotheses(
    const std::vector<std::string> &texts) {
  std::vector<OfflineRecognitionHypothesis> hyps;
  for (const auto &text : texts) {
    OfflineRecognitionHypothesis hyp;
    hyp.text = text;
    hyps.push_back(hyp);
  }
  return hyps;
}

static std::vector<std::string> Texts(
    const std::vector<OfflineRecognitionHypothesis> &hyps) {
  std::vector<std::string> texts;
  for (const auto &hyp : hyps) {
    texts.push_back(hyp.text);
  }
  return texts;
}

TEST(TruncateHypotheses, ZeroKeepsOne) {
  auto hyps = MakeHypotheses({"a", "b", "c", "d"});
  TruncateHypotheses(&hyps, 0, 4);
  EXPECT_EQ(Texts(hyps), (std::vector<std::string>{"a"}));
}

TEST(TruncateHypotheses, OneKeepsOne) {
  auto hyps = MakeHypotheses({"a", "b", "c", "d"});
  TruncateHypotheses(&hyps, 1, 4);
  EXPECT_EQ(Texts(hyps), (std::vector<std::string>{"a"}));
}

TEST(TruncateHypotheses, ThreeKeepsFirstThreeInOrder) {
  auto hyps = MakeHypotheses({"a", "b", "c", "d"});
  TruncateHypotheses(&hyps, 3, 4);
  EXPECT_EQ(Texts(hyps), (std::vector<std::string>{"a", "b", "c"}));
}

TEST(TruncateHypotheses, LargerThanListKeepsAll) {
  auto hyps = MakeHypotheses({"a", "b"});
  TruncateHypotheses(&hyps, 3, 4);
  EXPECT_EQ(Texts(hyps), (std::vector<std::string>{"a", "b"}));
}

TEST(TruncateHypotheses, LargerThanMaxActivePathsKeepsMaxActivePaths) {
  auto hyps = MakeHypotheses({"a", "b", "c", "d"});
  TruncateHypotheses(&hyps, 10, 3);
  EXPECT_EQ(Texts(hyps), (std::vector<std::string>{"a", "b", "c"}));
}

TEST(TruncateHypotheses, EmptyListStaysEmpty) {
  std::vector<OfflineRecognitionHypothesis> hyps;
  TruncateHypotheses(&hyps, 2, 4);
  EXPECT_TRUE(hyps.empty());
}

TEST(OfflineRecognitionResult, SerializesNBestHypotheses) {
  OfflineRecognitionResult result;
  result.text = "top one";
  result.tokens = {"top", " one"};

  OfflineRecognitionHypothesis first;
  first.text = "top one";
  first.tokens = {"top", " one"};
  first.timestamps = {0.0f, 0.4f};
  first.ys_log_probs = {-0.1f, -0.2f};
  first.score = -0.15;

  OfflineRecognitionHypothesis second;
  second.text = "top won";
  second.tokens = {"top", " won"};
  second.timestamps = {0.0f, 0.4f};
  second.ys_log_probs = {-0.1f, -0.3f};
  second.score = -0.2;

  result.hypotheses = {first, second};

  std::string json = result.AsJsonString();
  EXPECT_NE(json.find("\"hypotheses\": [{"), std::string::npos);
  EXPECT_NE(json.find("\"text\": \"top one\""), std::string::npos);
  EXPECT_NE(json.find("\"text\": \"top won\""), std::string::npos);
  EXPECT_NE(json.find("\"score\": -0.150000"), std::string::npos);
  EXPECT_NE(json.find("\"score\": -0.200000"), std::string::npos);
}

TEST(OfflineRecognitionResult, SerializesEmptyHypothesisList) {
  OfflineRecognitionResult result;
  std::string json = result.AsJsonString();
  EXPECT_NE(json.find("\"hypotheses\": []"), std::string::npos);
}

}  // namespace sherpa_onnx
