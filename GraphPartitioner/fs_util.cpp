/**  
 * Copyright (c) 2009 Carnegie Mellon University. 
 *     All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing,
 *  software distributed under the License is distributed on an "AS
 *  IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *  express or implied.  See the License for the specific language
 *  governing permissions and limitations under the License.
 *
 */


#include <boost/version.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string/predicate.hpp>


#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

#include "fs_util.hpp"

using namespace std;

void graphp::fs_util::
list_files_with_suffix(const string& pathname,
                       const string& suffix,
                       vector<string>& files) {
  namespace fs = boost::filesystem;
  fs::path dir_path(pathname);
  fs::directory_iterator end_iter;
  files.clear();
  if ( fs::exists(dir_path) && fs::is_directory(dir_path)) {
    for( fs::directory_iterator dir_iter(dir_path) ; 
         dir_iter != end_iter ; ++dir_iter) {
      if (fs::is_regular_file(dir_iter->status()) ) {
#if BOOST_FILESYSTEM_VERSION >= 3 
        const string filename = dir_iter->path().filename().string();
#else
        const string filename = dir_iter->leaf();
#endif
        if (suffix.size() > 0 && !boost::ends_with(filename, suffix)) 
          continue;
        files.push_back(filename);
      }
    }
  }
  sort(files.begin(), files.end());
//   namespace fs = boost::filesystem;
//   fs::path path(pathname);
//   assert(fs::exists(path));
//   for(fs::directory_iterator iter( path ), end_iter; 
//       iter != end_iter; ++iter) {
//     if( ! fs::is_directory(iter->status()) ) {

// #if BOOST_FILESYSTEM_VERSION >= 3
//       string filename(iter->path().filename().string());
// #else
//       string filename(iter->path().filename());
// #endif
//       size_t pos = 
//         filename.size() >= suffix.size()?
//         filename.size() - suffix.size() : 0;
//       string ending(filename.substr(pos));
//       if(ending == suffix) {
// #if BOOST_FILESYSTEM_VERSION >= 3
//         files.push_back(iter->path().filename().string());
// #else
//         files.push_back(iter->path().filename());
// #endif
//       }
//     }
//   }
//  sort(files.begin(), files.end());
} // end of list files with suffix  



void graphp::fs_util::
list_files_with_prefix(const string& pathname,
                       const string& prefix,
                       vector<string>& files) {
  namespace fs = boost::filesystem;  
  fs::path dir_path(pathname);
  fs::directory_iterator end_iter;
  files.clear();
  if ( fs::exists(dir_path) && fs::is_directory(dir_path)) {
    for( fs::directory_iterator dir_iter(dir_path) ; 
         dir_iter != end_iter ; ++dir_iter) {
      if (fs::is_regular_file(dir_iter->status()) ) {
        const string filename = dir_iter->path().filename().string();
        if (prefix.size() > 0 && !boost::starts_with(filename, prefix)) {
          continue;
        }
        files.push_back(dir_iter->path().string());
      }
    }
  }
  sort(files.begin(), files.end());
} // end of list files with prefix





string graphp::fs_util::
change_suffix(const string& fname,
              const string& new_suffix) {             
  size_t pos = fname.rfind('.');
  assert(pos != string::npos); 
  const string new_base(fname.substr(0, pos));
  return new_base + new_suffix;
} // end of change_suffix


