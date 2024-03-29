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


#ifndef GRAPH_PARTITIONER_FS_UTIL
#define GRAPH_PARTITIONER_FS_UTIL

#include <string>
#include <vector>

using namespace std;

namespace graphp {

  namespace fs_util {

    /**
     * List all the files with the given suffix at the pathname
     * location
     */
    void list_files_with_suffix(const string& pathname,
                                const string& suffix,
                                vector<string>& files);


    /**
     * List all the files with the given prefix at the pathname
     * location
     */
    void list_files_with_prefix(const string& pathname,
                                const string& prefix,
                                vector<string>& files);


    /// \ingroup util_internal
    string change_suffix(const string& fname,
                                     const string& new_suffix);

  }; // end of fs_utils


}; // end of graphp
#endif

