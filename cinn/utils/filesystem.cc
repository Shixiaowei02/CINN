// Copyright (c) 2023 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "cinn/utils/filesystem.h"

#include <cstdlib>

#include "glog/logging.h"

#ifdef __linux__
#include <errno.h>
#include <sys/stat.h>
#endif

namespace cinn {
namespace filesystem {

bool is_directory(const char* path) {
#if defined _WIN32
  LOG(FATAL) << "The is_directory method does not support Windows system yet.";
#elif defined __linux__
  struct stat s;
  int ret = lstat(path, &s);
  return !ret && S_ISDIR(s.st_mode);
#endif
  return false;
}

const char* temp_directory_path() {
#if defined _WIN32
  LOG(FATAL) << "The temp_directory_path method does not support Windows system yet.";
  return nullptr;
#elif defined __linux__
  const char* val = nullptr;
  (val = std::getenv("TMPDIR")) || (val = std::getenv("TEMPDIR"));
#ifdef __ANDROID__
  const char* default_tmp = "/data/local/tmp";
#else
  const char* default_tmp = "/tmp";
#endif
  const char* path        = val ? val : default_tmp;
  CHECK(is_directory(path)) << "Can not get the temp directory path because the path " << path
                            << " is not a directory.";
  return path;
#endif
}

bool create_directory(const char* path, int mode) {
#if defined _WIN32
  LOG(FATAL) << "The create_directory method does not support Windows system yet.";
#elif defined __linux__
  struct stat st;
  int status = 0;

  if (stat(path, &st) != 0) {
    /* Directory does not exist. EEXIST for race condition */
    if (mkdir(path, mode) != 0 && errno != EEXIST) status = -1;
  } else if (!S_ISDIR(st.st_mode)) {
    errno  = ENOTDIR;
    status = -1;
  }
  return !status;
#endif
}

}  // namespace filesystem
}  // namespace cinn
