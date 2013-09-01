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
#include <boost/dynamic_bitset.hpp>
#include <cmath>
#include <queue>
#include <list>
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

		void partition_by_patoh(basic_graph& graph, size_t nparts) {
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
		} // end of partition_by_patoh

		void partition_by_patoh_w(basic_graph& graph, size_t nparts) {
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
			nwghts = (int *) malloc(_n * sizeof(int));
			foreach(const basic_graph::verts_map_type::value_type& vp, graph.origin_verts) {
				xpins[vt] = et;

				foreach(const basic_graph::vertex_edge_map_type::value_type& ep, vp.second.edge_list) {
					pins[et] = ep.second;
					et++;
				}

				nwghts[vt] = vp.second.weight;

				vt++;
			}
			xpins[vt] = et;
//			cout << "converted" << endl;

			PaToH_Initialize_Parameters(&args, PATOH_CONPART, PATOH_SUGPARAM_DEFAULT);
//			cout << "initialized" << endl;

			args._k = nparts;
			partvec = (int *) malloc(_c*sizeof(int));
			for(int i = 0; i < _c; i++) {
				partvec[i] = graph.origin_edges[i].placement;
			}
			partweights = (int *) malloc(args._k*_nconst*sizeof(int));

			PaToH_Alloc(&args, _c, _n, _nconst, NULL, nwghts, xpins, pins);
//			cout << "allocated" << endl;

			PaToH_Part(&args, _c, _n, _nconst, 0, NULL, nwghts, xpins, pins, NULL, partvec, partweights, &cut);
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
		} // end of partition_by_patoh

		void vertex_filter(basic_graph& graph, boost::dynamic_bitset<>& result, size_t factor) {
			map<size_t, vector<size_t>> buckets;
			const size_t c = (size_t) pow(2.0 * graph.origin_edges.size() / graph.origin_verts.size(), 1.2);
			foreach(const basic_graph::verts_map_type::value_type& vp, graph.origin_verts) {
				size_t bucket = vp.second.nbr_list.size() / c;
				if(buckets.count(bucket) == 0) {
					vector<size_t> v;
					buckets.insert(pair<size_t, vector<size_t>>(bucket, v));
				}
				buckets[bucket].push_back(vp.second.vid);
			}

			size_t num_of_vertex = 0;
			const size_t limit = graph.origin_verts.size() * 1 / factor;
			for(map<size_t, vector<size_t>>::reverse_iterator iter = buckets.rbegin(); iter != buckets.rend(); iter++) {
			//for(map<size_t, vector<size_t>>::iterator iter = buckets.begin(); iter != buckets.end(); iter++) {
				foreach(size_t vp, iter->second) {
					result[vp] = true;
				}
				num_of_vertex += iter->second.size();
				if(num_of_vertex >= limit)
					break;
			}
		}

		void greedy_reorder(basic_graph& graph, size_t nparts, bool reverse_order = true) {
			boost::timer ti;

			// filter the vertices
			map<size_t, vector<vertex_id_type>> buckets;
			const size_t c = (size_t) (4.0 * graph.origin_edges.size() / graph.origin_verts.size());
			foreach(const basic_graph::verts_map_type::value_type& vp, graph.origin_verts) {
				size_t bucket = vp.second.nbr_list.size() / c;
				if(buckets.count(bucket) == 0) {
					vector<vertex_id_type> v;
					buckets.insert(pair<size_t, vector<vertex_id_type>>(bucket, v));
				}
				buckets[bucket].push_back(vp.second.vid);
			}

			size_t assign_counter = 0;
			if(reverse_order) {
				for(map<size_t, vector<vertex_id_type>>::reverse_iterator iter = buckets.rbegin(); iter != buckets.rend(); iter++) {
					foreach(vertex_id_type vid, iter->second) {
						foreach(vertex_id_type nbr, graph.origin_verts[vid].nbr_list) {
							edge_id_type eid = graph.origin_verts[vid].edge_list[nbr];
							if(graph.origin_edges[eid].placement == -1 && graph.origin_verts[vid].nbr_list.size() >= graph.origin_verts[nbr].nbr_list.size()) {
								// check if is not assigned
								// greedy assign
								basic_graph::part_t assignment;
								assignment = edge_to_part_greedy(graph.origin_verts[vid], graph.origin_verts[nbr], graph.parts_counter, false);
								assign_edge(graph, eid, assignment);
								assign_counter++;
							}
						}
					}
				}
			}
			else {
				for(map<size_t, vector<vertex_id_type>>::iterator iter = buckets.begin(); iter != buckets.end(); iter++) {
					foreach(vertex_id_type vid, iter->second) {
						foreach(vertex_id_type nbr, graph.origin_verts[vid].nbr_list) {
							edge_id_type eid = graph.origin_verts[vid].edge_list[nbr];
							if(graph.origin_edges[eid].placement == -1 && graph.origin_verts[vid].nbr_list.size() >= graph.origin_verts[nbr].nbr_list.size()) {
								// check if is not assigned
								// greedy assign
								basic_graph::part_t assignment;
								assignment = edge_to_part_greedy(graph.origin_verts[vid], graph.origin_verts[nbr], graph.parts_counter, false);
								assign_edge(graph, eid, assignment);
								assign_counter++;
							}
						}
					}
				}
			}
			cout << "Edges assigned: " << assign_counter << endl;

			foreach(basic_graph::edge_type& e, graph.origin_edges) {
				if(e.placement == -1) {
					basic_graph::part_t assignment;
					assignment = edge_to_part_greedy(graph.origin_verts[e.source], graph.origin_verts[e.target], graph.parts_counter, false);
					assign_edge(graph, e.eid, assignment);
					assign_counter++;
				}
			}
			cout << "Edges assigned: " << assign_counter << endl;

			bal_greedy_refinement(graph, nparts);

			cout << "Time elapsed: " << ti.elapsed() << endl;

			report_performance(graph, nparts);
		} // end of greedy_reorder

		void partition_by_patoh_fast(basic_graph& graph, size_t nparts) {
			boost::timer ti;

			greedy_reorder(graph, nparts, false);
			// filter the vertices
			boost::dynamic_bitset<> v_to_part(graph.max_vid + 1);
			vertex_filter(graph, v_to_part, 200);
			cout << "Vertices to be partitioned by hypergraph: " << v_to_part.count() << endl;

			// convert the subgraph
			basic_graph subgraph(nparts);
			for(size_t idx = v_to_part.find_first(); idx != v_to_part.npos; idx = v_to_part.find_next(idx)) {
				foreach(vertex_id_type nbr, graph.origin_verts[idx].nbr_list) {
					if(v_to_part[nbr]) {
						subgraph.add_edge(idx, nbr);
					}
				}
			}
			vertex_id_type max_vid = subgraph.max_vid + 1;
			for(size_t idx = v_to_part.find_first(); idx != v_to_part.npos; idx = v_to_part.find_next(idx)) {
				foreach(basic_graph::part_t part, graph.origin_verts[idx].mirror_list) {
					subgraph.add_edge(idx, max_vid + part);
				}
			}
			//size_t mean_nverts = graph.nverts / nparts;
			for(size_t part = 0; part < nparts; part++) {
				subgraph.origin_verts[max_vid + part].weight = graph.parts_counter[part];
				foreach(vertex_id_type nbr, subgraph.origin_verts[max_vid + part].nbr_list) {
					assign_edge(subgraph, subgraph.origin_verts[max_vid + part].edge_list[nbr], part);
				}
			}
			cout << "subgraph: verts: " << subgraph.nverts << " edges: " << subgraph.nedges << endl;
			// end of convert

			partition_by_patoh_w(subgraph, nparts);

			graph.clear_partition();
			report_performance(graph, nparts);

			size_t assign_counter = 0;
			// convert back
			foreach(basic_graph::edge_type& e, subgraph.origin_edges) {
				if(e.source < max_vid && e.target < max_vid) {
					assign_edge(graph, graph.origin_verts[e.source].edge_list[e.target], e.placement);
					assign_counter++;
				}
			}
			cout << "Edges assigned: " << assign_counter << endl;
			report_performance(graph, nparts);
			// end of convert back

			//foreach(basic_graph::edge_type& e, graph.origin_edges) {
			//	if(e.placement == -1) {
			//		// check if is sparse
			//		// greedy assign
			//		basic_graph::part_t assignment;
			//		assignment = edge_to_part_greedy(graph.origin_verts[e.source], graph.origin_verts[e.target], graph.parts_counter, false);
			//		assign_edge(graph, e.eid, assignment);
			//		assign_counter++;
			//	}
			//}
			//cout << "Edges assigned: " << assign_counter << endl;
			greedy_reorder(graph, nparts, false);

			cout << "Time elapsed: " << ti.elapsed() << endl;

			report_performance(graph, nparts);
		} // end of partition_by_patoh_fast

	} // end of namespace partition_strategy

} // end of namespace graphp

#endif
