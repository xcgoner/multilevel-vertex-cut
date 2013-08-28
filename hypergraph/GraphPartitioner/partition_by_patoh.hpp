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


#ifndef GRAPH_PARTITIONER_PARTITION_BY_PATOH
#define GRAPH_PARTITIONER_PARTITION_BY_PATOH

#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random.hpp>
#include <boost/timer.hpp>
#include "basic_graph.hpp"
#include "util.hpp"
#include "partition_strategy.hpp"
#include "patoh.h"

using namespace std;
#ifdef __GNUC__
using namespace __gnu_cxx;
#endif

namespace graphp {

	namespace partition_strategy {
		void PrintInfo(int _k, int *partweights, int cut, int _nconst)
		{
		 double             *avg, *maxi, maxall=-1.0;
		 int                i, j;

		 printf("\n-------------------------------------------------------------------");
		 printf("\n Partitioner: %s", (_nconst>1) ? "Multi-Constraint" : "Single-Constraint");

		 printf("\n %d-way cutsize = %d \n", _k, cut);

		 printf("\nPartWeights are:\n");
		 avg = (double *) malloc(sizeof(double)*_nconst);
		 maxi = (double *) malloc(sizeof(double)*_nconst);
		 for (i=0; i<_nconst; ++i)
			 maxi[i] = avg[i] = 0.0;
		 for (i=0; i<_k; ++i)
			   for (j=0; j<_nconst; ++j)
				 avg[j] += partweights[i*_nconst+j];
		  for (i=0; i<_nconst; ++i)
			 {
			 maxi[i] = 0.0;
			 avg[i] /= (double) _k;
			 }

		 for (i=0; i<_k; ++i)
			 {
			 printf("\n %3d :", i);
			 for (j=0; j<_nconst; ++j)
				 {
				 double im= (double)((double)partweights[i*_nconst+j] - avg[j]) / avg[j];

				 maxi[j] = (maxi[j] > im) ? maxi[j] : im;
				 printf("%10d ", partweights[i*_nconst+j]);
				 }
			 }
		 for (j=0; j<_nconst; ++j)
			 maxall = (maxi[j] > maxall) ? maxi[j] : maxall;
		 printf("\n MaxImbals are (as %%): %.3lf", 100.0*maxall);
		 printf("\n      ");
		 for (i=0; i<_nconst; ++i)
			 printf("%10.1lf ", 100.0*maxi[i]);
		 printf("\n");
		 free(maxi);
		 free(avg);
		}

		void partition_by_patoh (basic_graph& graph, size_t nparts) {
			boost::timer ti;

			// in patoh, cell means vertex and net means hyperedge
			PaToH_Parameters args;
			int _c, _n, _nconst, *cwghts = NULL, *nwghts = NULL, *xpins, *pins, *partvec, cut, *partweights;
			// unweighted
			_c = graph.nedges; _n = graph.nverts;
			// pins = nedges * 2
			_nconst = 1;

			xpins = (int *) malloc((_n + 1) * sizeof(int));
			pins = (int *) malloc(graph.nedges * 2 * sizeof(int));
			size_t vt = 0, et = 0;
			foreach(const basic_graph::verts_map_type::value_type& vp, graph.origin_verts) {
				xpins[vt] = et;

				foreach(const basic_graph::vertex_edge_map_type::value_type& ep, vp.second.edge_list) {
					pins[et] = ep.second;
					et++;
				}

				vt++;
			}
			xpins[vt] = et;
//			cout << "converted" << endl;

			PaToH_Initialize_Parameters(&args, PATOH_CONPART, PATOH_SUGPARAM_DEFAULT);
//			cout << "initialized" << endl;

			args._k = nparts;
			partvec = (int *) malloc(_c*sizeof(int));
			partweights = (int *) malloc(args._k*_nconst*sizeof(int));

			PaToH_Alloc(&args, _c, _n, _nconst, NULL, NULL, xpins, pins);
//			cout << "allocated" << endl;

			PaToH_Part(&args, _c, _n, _nconst, 0, NULL, NULL, xpins, pins, NULL, partvec, partweights, &cut);
//			cout << "parted" << endl;

			cout << "hypergraph " << args._k << "-way cutsize is: " << cut << endl;
//			PrintInfo(args._k, partweights,  cut, _nconst);

			for(int i = 0; i < _c; i++) {
				assign_edge(graph, i, partvec[i]);
			}

			//free(cwghts);      free(nwghts);
			free(xpins);       free(pins);
			free(partweights); free(partvec);

			PaToH_Free();

			cout << "Time elapsed: " << ti.elapsed() << endl;

			report_performance(graph, nparts);
		}
	} // end of namespace partition_strategy

} // end of namespace graphp

#endif
