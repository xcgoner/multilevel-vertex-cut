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

#include <iostream>
#include <string>

#ifdef HAS_TCMALLOC
#include <google/malloc_extension.h>
#endif

using namespace std;

namespace graphp {
  namespace memory_info {

    bool available() { 
#ifdef HAS_TCMALLOC
      return true;
#else
      return false;
#endif
    } // end of available



    size_t heap_bytes() {
      size_t heap_size(0);
#ifdef HAS_TCMALLOC
      MallocExtension::instance()->
        GetNumericProperty("generic.heap_size", &heap_size);
#else
      cerr << 
        "memory_info::heap_bytes() requires tcmalloc" << endl;
#endif
      return heap_size;
    } // end of heap size
    


    size_t allocated_bytes() {
      size_t allocated_size(0);
#ifdef HAS_TCMALLOC
      MallocExtension::instance()->
        GetNumericProperty("generic.current_allocated_bytes", 
                           &allocated_size);
#else
      cerr << 
        "memory_info::allocated_bytes() requires tcmalloc" << endl;
#endif
      return allocated_size;
    } // end of allocated bytes



    void print_usage(const string& label) {
#ifdef HAS_TCMALLOC
        const double BYTES_TO_MB = double(1) / double(1024 * 1024);
        cout 
          << "Memory Info: " << label << endl
          << "\t Heap: " << (heap_bytes() * BYTES_TO_MB) << " MB"
          << endl
          << "\t Allocated: " << (allocated_bytes() * BYTES_TO_MB) << " MB" 
          << endl;
#else
        cerr 
          << "Unable to print memory info for: " << label << ". "
          << "No memory extensions api available." << endl;
#endif
    } // end of print_usage

    void log_usage(const string& label) {
#ifdef HAS_TCMALLOC
        const double BYTES_TO_MB = double(1) / double(1024 * 1024);
        cout
          << "Memory Info: " << label
          << "\n\t Heap: " << (heap_bytes() * BYTES_TO_MB) << " MB"
          << "\n\t Allocated: " << (allocated_bytes() * BYTES_TO_MB) << " MB" 
          << endl;
#else
        cerr 
          << "Unable to print memory info for: " << label << ". "
          << "No memory extensions api available." << endl;
#endif
    } // end of log usage


  }; // end of namespace memory info

}; // end of graphp namespace


