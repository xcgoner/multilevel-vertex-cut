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
 * For more about this software visit:
 *
 *      http://www.graphlab.ml.cmu.edu
 *
 */

#include <set>
#include <iostream>
#include <fstream>

#include <boost/random.hpp>
#include <boost/integer_traits.hpp>

#include "timer.hpp"
#include "random.hpp"
#include "util.hpp"

namespace graphp {
  namespace random {

    /**
     * A truely nondeterministic generator
     */
    class nondet_generator {
    public:
      static nondet_generator& global() {
        static nondet_generator global_gen;
        return global_gen;
      }

      typedef size_t result_type;
      BOOST_STATIC_CONSTANT(result_type, min_value = 
                            boost::integer_traits<result_type>::const_min);
      BOOST_STATIC_CONSTANT(result_type, max_value = 
                            boost::integer_traits<result_type>::const_max);
      result_type min BOOST_PREVENT_MACRO_SUBSTITUTION () const { return min_value; }
      result_type max BOOST_PREVENT_MACRO_SUBSTITUTION () const { return max_value; }
      
      nondet_generator() {
        rnd_dev.open("/dev/urandom", std::ios::binary | std::ios::in);
      }
      // Close the random number generator
      ~nondet_generator() { rnd_dev.close(); }
      // read a size_t from the source
      result_type operator()() {
        // read a machine word into result
        result_type result(0);
        rnd_dev.read(reinterpret_cast<char*>(&result), sizeof(result_type));
        //        std::cout << result << std::endl;
        return result;
      }      
    private:
      std::ifstream rnd_dev;
    };
    //nondet_generator global_nondet_rng;

    void pdf2cdf(std::vector<double>& pdf) {
      double Z = 0;
      for(size_t i = 0; i < pdf.size(); ++i) Z += pdf[i];
      for(size_t i = 0; i < pdf.size(); ++i)
        pdf[i] = pdf[i]/Z + ((i>0)? pdf[i-1] : 0);
    } // end of pdf2cdf



  
  }; // end of namespace random

};// end of namespace graphp

