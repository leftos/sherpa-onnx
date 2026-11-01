// yaat/smoke-nbest.c
//
// Smoke test for the yaat-nbest natives: decodes <model-dir>/test_wavs/0.wav
// with modified_beam_search and four return paths, prints the result JSON and
// exits non-zero unless it carries a non-empty "text" and a non-empty
// "hypotheses" array.
//
// Usage: smoke-nbest <model-dir>
// <model-dir> is sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8 (or any NeMo
// transducer laid out the same way).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-onnx/c-api/c-api.h"

#define PATH_SIZE 1024

static int32_t JoinPath(char *out, const char *dir, const char *name) {
  int32_t n = snprintf(out, PATH_SIZE, "%s/%s", dir, name);
  return n > 0 && n < PATH_SIZE;
}

// Returns 1 when `key` is followed by a value whose first character differs
// from `empty_marker`, i.e. a non-empty string (marker '"') or array (']').
static int32_t HasNonEmptyValue(const char *json, const char *key,
                                char empty_marker) {
  const char *p = strstr(json, key);
  if (p == NULL) {
    return 0;
  }
  p += strlen(key);
  return *p != '\0' && *p != empty_marker;
}

static int32_t CheckJson(const char *json) {
  int32_t ok = 1;
  if (!HasNonEmptyValue(json, "\"text\": \"", '"')) {
    fprintf(stderr, "smoke-nbest: result JSON has no non-empty \"text\"\n");
    ok = 0;
  }
  if (!HasNonEmptyValue(json, "\"hypotheses\": [", ']')) {
    fprintf(stderr,
            "smoke-nbest: result JSON has no non-empty \"hypotheses\" array; "
            "is this the yaat-nbest native?\n");
    ok = 0;
  }
  return ok;
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <model-dir>\n", argv[0]);
    return 2;
  }
  const char *dir = argv[1];

  char wav[PATH_SIZE], encoder[PATH_SIZE], decoder[PATH_SIZE],
      joiner[PATH_SIZE], tokens[PATH_SIZE], bpe_vocab[PATH_SIZE];
  if (!JoinPath(wav, dir, "test_wavs/0.wav") ||
      !JoinPath(encoder, dir, "encoder.int8.onnx") ||
      !JoinPath(decoder, dir, "decoder.int8.onnx") ||
      !JoinPath(joiner, dir, "joiner.int8.onnx") ||
      !JoinPath(tokens, dir, "tokens.txt") ||
      !JoinPath(bpe_vocab, dir, "bpe.vocab")) {
    fprintf(stderr, "smoke-nbest: model dir path too long: %s\n", dir);
    return 2;
  }

  if (!SherpaOnnxFileExists(wav)) {
    fprintf(stderr, "smoke-nbest: file not found: %s\n", wav);
    return 1;
  }
  const SherpaOnnxWave *wave = SherpaOnnxReadWave(wav);
  if (wave == NULL) {
    fprintf(stderr, "smoke-nbest: cannot read %s\n", wav);
    return 1;
  }

  SherpaOnnxOfflineRecognizerConfig config;
  memset(&config, 0, sizeof(config));
  config.model_config.num_threads = 2;
  config.model_config.provider = "cpu";
  config.model_config.tokens = tokens;
  config.model_config.transducer.encoder = encoder;
  config.model_config.transducer.decoder = decoder;
  config.model_config.transducer.joiner = joiner;
  config.decoding_method = "modified_beam_search";
  config.max_active_paths = 4;
  if (SherpaOnnxFileExists(bpe_vocab)) {
    config.model_config.modeling_unit = "bpe";
    config.model_config.bpe_vocab = bpe_vocab;
  } else {
    printf("smoke-nbest: no %s; decoding without modeling_unit/bpe_vocab\n",
           bpe_vocab);
  }

  const SherpaOnnxOfflineRecognizer *recognizer =
      SherpaOnnxCreateOfflineRecognizer(&config);
  if (recognizer == NULL) {
    fprintf(stderr, "smoke-nbest: cannot create the recognizer\n");
    SherpaOnnxFreeWave(wave);
    return 1;
  }

  const SherpaOnnxOfflineStream *stream =
      SherpaOnnxCreateOfflineStream(recognizer);
  if (stream == NULL) {
    fprintf(stderr, "smoke-nbest: cannot create the stream\n");
    SherpaOnnxDestroyOfflineRecognizer(recognizer);
    SherpaOnnxFreeWave(wave);
    return 1;
  }
  SherpaOnnxOfflineStreamSetOption(stream, "num_return_paths", "4");

  SherpaOnnxAcceptWaveformOffline(stream, wave->sample_rate, wave->samples,
                                  wave->num_samples);
  SherpaOnnxDecodeOfflineStream(recognizer, stream);

  const char *json = SherpaOnnxGetOfflineStreamResultAsJson(stream);
  int32_t ok = 0;
  if (json == NULL) {
    fprintf(stderr, "smoke-nbest: no result JSON\n");
  } else {
    printf("%s\n", json);
    ok = CheckJson(json);
    SherpaOnnxDestroyOfflineStreamResultJson(json);
  }

  SherpaOnnxDestroyOfflineStream(stream);
  SherpaOnnxDestroyOfflineRecognizer(recognizer);
  SherpaOnnxFreeWave(wave);

  return ok ? 0 : 1;
}
