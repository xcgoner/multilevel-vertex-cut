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


#ifndef GRAPH_PARTITIONER_RANDOM_HPP
#define GRAPH_PARTITIONER_RANDOM_HPP

#include <cstdlib>
#include <stdint.h>


#include <vector>
#include <limits>
#include <algorithm>

#include <boost/random.hpp>
#include "timer.hpp"

namespace graphp {

  /**
   * \ingroup random
   * A collection of thread safe random number routines.  Each thread
   * is assigned its own generator however assigning a seed affects
   * all current and future generators.
   */
  namespace random {        


    ///////////////////////////////////////////////////////////////////////
    //// Underlying generator definition



    namespace distributions {
      /**
       * The uniform distribution struct is used for partial function
       * specialization. Generating uniform random real numbers is
       * accomplished slightly differently than for integers.
       * Therefore the base case is for integers and we then
       * specialize the two real number types (floats and doubles).
       */
      template<typename IntType>
      struct uniform {
        typedef boost::uniform_int<IntType> distribution_type;
        template<typename RealRNG, typename DiscreteRNG>
        static inline IntType sample(RealRNG& real_rng, 
                                     DiscreteRNG& discrete_rng, 
                                     const IntType& min, const IntType& max) {
          return distribution_type(min, max)(discrete_rng);
        }
      };
      template<>
      struct uniform<double> {
        typedef boost::uniform_real<double> distribution_type;
        template<typename RealRNG, typename DiscreteRNG>
        static inline double sample(RealRNG& real_rng, 
                                    DiscreteRNG& discrete_rng, 
                                    const double& min, const double& max) {
          return distribution_type(min, max)(real_rng);
        }
      };
      template<>
      struct uniform<float> {
        typedef boost::uniform_real<float> distribution_type;
        template<typename RealRNG, typename DiscreteRNG>
        static inline float sample(RealRNG& real_rng, 
                                  DiscreteRNG& discrete_rng, 
                                  const float& min, const float& max) {
          return distribution_type(min, max)(real_rng);
        }
      };
    }; // end of namespace distributions

    /**
     * The generator class is the base underlying type used to
     * generate random numbers.  User threads should use the functions
     * provided in the random namespace.
     */
    class generator {
    public:
      // base Generator types
      typedef boost::lagged_fibonacci607 real_rng_type;
      typedef boost::mt11213b            discrete_rng_type;  
      typedef boost::rand48              fast_discrete_rng_type;       
    
      generator() {
        time_seed();
      }
    
      //! Seed the generator using the default seed
      inline void seed() {
        real_rng.seed();
        discrete_rng.seed();
        fast_discrete_rng.seed();
      }

      //! Seed the generator nondeterministically
      void nondet_seed();


      //! Seed the generator using the current time in microseconds
      inline void time_seed() {
        seed( graphp::timer::usec_of_day() );
      }

      //! Seed the random number generator based on a number
      void seed(size_t number) {
        fast_discrete_rng.seed(number);
        real_rng.seed(fast_discrete_rng);
        discrete_rng.seed(fast_discrete_rng);
      }
      
      //! Seed the generator using another generator
      void seed(generator& other){
        real_rng.seed(other.real_rng);
        discrete_rng.seed(other.discrete_rng);
        fast_discrete_rng.seed(other.fast_discrete_rng());
      } 
   
      /**
       * Generate a random number in the uniform real with range [min,
       * max) or [min, max] if the number type is discrete.
       */
      template<typename NumType>
      inline NumType uniform(const NumType min, const NumType max) { 
        const NumType result = distributions::uniform<NumType>::
          sample(real_rng, discrete_rng, min, max);
        return result;
      } // end of uniform

      /**
       * Generate a random number in the uniform real with range [min,
       * max) or [min, max] if the number type is discrete.
       */
      template<typename NumType>
      inline NumType fast_uniform(const NumType min, const NumType max) { 
        const NumType result = distributions::uniform<NumType>::
          sample(real_rng, fast_discrete_rng, min, max);
        return result;
      } // end of fast_uniform


      /**
       * Generate a random number in the uniform real with range [min,
       * max);
       */
      inline double gamma(const double alpha = double(1)) {
        boost::gamma_distribution<double> gamma_dist(alpha);
        const double result = gamma_dist(real_rng);
        return result;
      } // end of gamma


      /**
       * Generate a gaussian random variable with zero mean and unit
       * variance.
       */
      inline double gaussian(const double mean = double(0), 
                             const double stdev = double(1)) {
        boost::normal_distribution<double> normal_dist(mean,stdev);
        const double result = normal_dist(real_rng);
        return result;
      } // end of gaussian

      /**
       * Generate a gaussian random variable with zero mean and unit
       * variance.
       */
      inline double normal(const double mean = double(0), 
                           const double stdev = double(1)) {
        return gaussian(mean, stdev);
      } // end of normal


      inline bool bernoulli(const double p = double(0.5)) {
        boost::bernoulli_distribution<double> dist(p);
        const double result(dist(discrete_rng));
        return result;
      } // end of bernoulli

      inline bool fast_bernoulli(const double p = double(0.5)) {
        boost::bernoulli_distribution<double> dist(p);
        const double result(dist(fast_discrete_rng));
        return result;
      } // end of bernoulli


      /**
       * Draw a random number from a multinomial
       */
      template<typename Double>
      size_t multinomial(const std::vector<Double>& prb) {
        if (prb.size() == 1) { return 0; }
        Double sum(0);
        for(size_t i = 0; i < prb.size(); ++i) {
          sum += prb[i];
        }
        // actually draw the random number
        const Double rnd(uniform<Double>(0,1));
        size_t ind = 0;
        for(Double cumsum(prb[ind]/sum); 
            rnd > cumsum && (ind+1) < prb.size(); 
            cumsum += (prb[++ind]/sum));
        return ind;
      } // end of multinomial


      /**
       * Generate a draw from a multinomial using a CDF.  This is
       * slightly more efficient since normalization is not required
       * and a binary search can be used.
       */
      template<typename Double>
      inline size_t multinomial_cdf(const std::vector<Double>& cdf) {
        return std::upper_bound(cdf.begin(), cdf.end(),
                                uniform<Double>(0,1)) - cdf.begin();
        
      } // end of multinomial_cdf
      
      //! The real random number generator
      real_rng_type real_rng;
      //! The discrete random number generator
      discrete_rng_type discrete_rng;
      //! The fast discrete random number generator
      fast_discrete_rng_type fast_discrete_rng;
      //! lock used to access local members  
    }; // end of class generator

    /**
     * \ingroup random
     * Seed all generators using the default seed
     */
    void seed();

    /**
     * \ingroup random
     * Seed all generators using an integer
     */
    void seed(size_t seed_value);

    /**
     * \ingroup random
     * Seed all generators using a nondeterministic source
     */
    void nondet_seed();

    /**
     * \ingroup random
     * Seed all generators using the current time in microseconds
     */
    void time_seed();
    

    /**
     * \ingroup random
     * Get the local generator
     */
    generator& get_source();

    /**
     * \ingroup random
     * Generate a random number in the uniform real with range [min,
     * max) or [min, max] if the number type is discrete.
     */
    template<typename NumType>
    inline NumType uniform(const NumType min, const NumType max) { 
      if (min == max) return min;
      return get_source().uniform<NumType>(min, max);
    } // end of uniform
    
    /**
     * \ingroup random
     * Generate a random number in the uniform real with range [min,
     * max) or [min, max] if the number type is discrete.
     */
    template<typename NumType>
    inline NumType fast_uniform(const NumType min, const NumType max) { 
      if (min == max) return min;
      return get_source().fast_uniform<NumType>(min, max);
    } // end of fast_uniform
    
    /**
     * \ingroup random
     * Generate a random number between 0 and 1
     */
    inline double rand01() { return uniform<double>(0, 1); }

    /**
     * \ingroup random
     * Simulates the standard rand function as defined in cstdlib
     */
    inline int rand() { return fast_uniform(0, RAND_MAX); }


    /**
     * \ingroup random
     * Generate a random number from a gamma distribution.
     */
    inline double gamma(const double alpha = double(1)) {
      return get_source().gamma(alpha);
    }



    /**
     * \ingroup random
     * Generate a gaussian random variable with zero mean and unit
     * standard deviation.
     */
    inline double gaussian(const double mean = double(0), 
                           const double stdev = double(1)) {
      return get_source().gaussian(mean, stdev);
    }

    /**
     * \ingroup random
     * Generate a gaussian random variable with zero mean and unit
     * standard deviation.
     */
    inline double normal(const double mean = double(0), 
                         const double stdev = double(1)) {
      return get_source().normal(mean, stdev);
    }

    /**
     * \ingroup random
     * Draw a sample from a bernoulli distribution
     */
    inline bool bernoulli(const double p = double(0.5)) {
      return get_source().bernoulli(p);
    }

    /**
     * \ingroup random
     * Draw a sample form a bernoulli distribution using the faster generator
     */
    inline bool fast_bernoulli(const double p = double(0.5)) {
      return get_source().fast_bernoulli(p);
    }

    /**
     * \ingroup random
     * Generate a draw from a multinomial.  This function
     * automatically normalizes as well.
     */
    template<typename Double>
    inline size_t multinomial(const std::vector<Double>& prb) {
      return get_source().multinomial(prb);
    }


    /**
     * \ingroup random
     * Generate a draw from a cdf;
     */
    template<typename Double>
    inline size_t multinomial_cdf(const std::vector<Double>& cdf) {
      return get_source().multinomial_cdf(cdf);
    }



    /** 
     * \ingroup random
     * Construct a random permutation
     */ 
    template<typename T>
    inline std::vector<T> permutation(const size_t nelems) { 
      return get_source().permutation<T>(nelems); 
    }


    /** 
     * \ingroup random
     * Shuffle a standard vector
     */ 
    template<typename T>
    inline void shuffle(std::vector<T>& vec) { 
      get_source().shuffle(vec); 
    }
   
    /** 
     * \ingroup random
     * Shuffle a range using the begin and end iterators
     */ 
    template<typename Iterator>
    inline void shuffle(Iterator begin, Iterator end) {
      get_source().shuffle(begin, end);
    }

    /**
     * Converts a discrete PDF into a CDF
     */
    void pdf2cdf(std::vector<double>& pdf);


    
  }; // end of random 
}; // end of graphp


#endif

