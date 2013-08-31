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

		void vertex_filter(basic_graph& graph, boost::dynamic_bitset<>& result, size_t factor) {
			map<size_t, vector<size_t>> buckets;
			const size_t c = (size_t) pow(2.0 * graph.origin_edges.size() / graph.origin_verts.size(), 1.5);
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

		void partition_by_patoh_fast(basic_graph& graph, size_t nparts) {
			boost::timer ti;

			// filter the vertices
			boost::dynamic_bitset<> vfilter(graph.max_vid + 1);
			vertex_filter(graph, vfilter, 10000);
			//cout << "Vertices to be partitioned by hypergraph: " << vfilter.count() << endl;

			//// count the average degree
			//size_t cutted_vertex_num = 0, boundary_degree = 0;
			//for(hash_map<vertex_id_type, basic_graph::vertex_type>::const_iterator iter = graph.origin_verts.begin(); iter != graph.origin_verts.end(); iter++) {
			//	if(vfilter[iter->first]) {
			//		// is a vertex to be partitioned
			//		cutted_vertex_num++;
			//		boundary_degree += iter->second.nbr_list.size();
			//	}
			//}
			//cout << "Average degree: " << 1.0 * boundary_degree / cutted_vertex_num << " : " << 2.0 * graph.origin_edges.size() / graph.origin_verts.size() << endl;

			// set the subgraph to be partitioned
			boost::dynamic_bitset<> v_to_part(graph.max_vid + 1);
			//for(size_t idx = vfilter.find_first(); idx != vfilter.npos; idx = vfilter.find_next(idx)) {
			//	v_to_part[idx] = true;
			//	foreach(vertex_id_type vid, graph.origin_verts[idx].nbr_list) {
			//		if(vfilter[vid] == false)
			//			v_to_part[vid] = true;
			//	}
			//}
			v_to_part |= vfilter;

			// visit the verts in random order
			vector<vertex_id_type> vertex_order;
			foreach(basic_graph::verts_map_type::value_type& vp, graph.origin_verts) {
				if(v_to_part[vp.first] == false)
					vertex_order.push_back(vp.first);
			}
			random_shuffle(vertex_order.begin(), vertex_order.end());
			// partition the sparse part as a way to cluster / coarsen
			size_t assign_counter = 0;
			foreach(vertex_id_type vid, vertex_order) {
				foreach(vertex_id_type nbr, graph.origin_verts[vid].nbr_list) {
					edge_id_type eid = graph.origin_verts[vid].edge_list[nbr];
					if(v_to_part[vid] == false && v_to_part[nbr] == false && graph.origin_edges[eid].placement == -1) {
						// check if is sparse
						// check if is not assigned
						// greddy assign
						basic_graph::part_t assignment;
						assignment = edge_to_part_greedy(graph.origin_verts[vid], graph.origin_verts[nbr], graph.parts_counter, false, true);
						assign_edge(graph, eid, assignment);
						assign_counter++;
					}
				}
			}
			cout << "Edges assigned: " << assign_counter << endl;
			report_performance(graph, nparts);
			return;

			size_t sub_nedges = 0, sub_nverts = 0, npins = 0;
			typedef map<edge_id_type, edge_id_type> edge_map_type;
			edge_map_type edge_map;
			for(size_t idx = v_to_part.find_first(); idx != v_to_part.npos; idx = v_to_part.find_next(idx)) {
				foreach(vertex_id_type nbr, graph.origin_verts[idx].nbr_list) {
					edge_id_type eid = graph.origin_verts[idx].edge_list[nbr];
					if(edge_map.count(eid) == 0) {
						edge_map.insert(pair<edge_id_type, edge_id_type>(eid, sub_nedges));
						sub_nedges++;
					}
				}
				npins += graph.origin_verts[idx].edge_list.size();
			}
			sub_nverts = v_to_part.count();
			vector<edge_id_type> edge_remap(sub_nedges);
			foreach(edge_map_type::value_type edge_map_entry, edge_map) {
				edge_remap[edge_map_entry.second] = edge_map_entry.first;
			}


			typedef pair<vertex_id_type, vertex_id_type> edge_pair_type;

			// in patoh, cell means vertex and net means hyperedge
			PaToH_Parameters args;
			int _c, _n, _nconst, *cwghts = NULL, *nwghts = NULL, *xpins, *pins, *partvec, cut, *partweights;
			float *targetweights;
			// unweighted
			_c = sub_nedges; _n = sub_nverts;
			cout << "Subgraph size: edges: " << sub_nedges << " verts: " << sub_nverts << endl;

			// pins = nedges * 2
			_nconst = 1;

			xpins = (int *) malloc((_n + 1) * sizeof(int));
			pins = (int *) malloc(npins * sizeof(int));
			size_t vt = 0, et = 0;
			for(size_t idx = v_to_part.find_first(); idx != v_to_part.npos; idx = v_to_part.find_next(idx)) {
				xpins[vt] = et;

				foreach(vertex_id_type nbr, graph.origin_verts[idx].nbr_list) {
					edge_id_type eid = graph.origin_verts[idx].edge_list[nbr];
					pins[et] = edge_map[eid];
					et++;
				}

				vt++;
			}
			xpins[vt] = et;
//			cout << "converted" << endl;

			PaToH_Initialize_Parameters(&args, PATOH_CONPART, PATOH_SUGPARAM_SPEED);
			args.ref_alg = PATOH_REFALG_NONE;
//			cout << "initialized" << endl;

			args._k = nparts;
			partvec = (int *) malloc(_c * sizeof(int));
			for(int i = 0; i < _c; i++) {
				partvec[i] = graph.origin_edges[edge_remap[i]].placement;
			}
			partweights = (int *) malloc(args._k*_nconst * sizeof(int));

			//targetweights = (float *) malloc(args._k * sizeof(float));
			//for(size_t idx = 0; idx < nparts; idx++) {
			//	targetweights[idx] = (float)1.0 * graph.nedges / graph.parts_counter[idx];
			//}

			PaToH_Alloc(&args, _c, _n, _nconst, NULL, nwghts, xpins, pins);
//			cout << "allocated" << endl;

			// use fixed cells
			PaToH_Part(&args, _c, _n, _nconst, 1, NULL, nwghts, xpins, pins, NULL, partvec, partweights, &cut);
//			cout << "parted" << endl;

			cout << "hypergraph " << args._k << "-way cutsize is: " << cut << endl;
//			PrintInfo(args._k, partweights,  cut, _nconst);


			for(int i = 0; i < _c; i++) {
				if(vfilter[graph.origin_edges[edge_remap[i]].source] == true || vfilter[graph.origin_edges[edge_remap[i]].target] == true) {
					assign_edge(graph, edge_remap[i], partvec[i]);
					assign_counter++;
				}
			}
			cout << "Edges assigned: " << assign_counter << endl;

			//free(cwghts);      free(nwghts);
			free(xpins);       free(pins);
			free(partweights); free(partvec);
			//free(targetweights);

			PaToH_Free();

			//// partition the dense part
			//foreach(basic_graph::edge_type& e, graph.origin_edges) {
			//	if(vfilter[e.source] == false && vfilter[e.target] == false) {
			//		// check if is sparse
			//		// greddy assign
			//		basic_graph::part_t assignment;
			//		assignment = edge_to_part_greedy(graph.origin_verts[e.source], graph.origin_verts[e.target], graph.parts_counter, false);
			//		assign_edge(graph, e.eid, assignment);
			//		assign_counter++;
			//	}
			//}
			//cout << "Edges assigned: " << assign_counter << endl;

			cout << "Time elapsed: " << ti.elapsed() << endl;

			report_performance(graph, nparts);
		} // end of partition_by_patoh_fast

		void greedy_reorder(basic_graph& graph, size_t nparts) {
			boost::timer ti;

			// filter the vertices
			boost::dynamic_bitset<> vfilter(graph.max_vid);
			vertex_filter(graph, vfilter, 1000);
			//cout << "Vertices to be partitioned by hypergraph: " << vfilter.count() << endl;

			// set the subgraph to be partitioned
			boost::dynamic_bitset<> v_to_part(graph.max_vid);
			for(size_t idx = vfilter.find_first(); idx != vfilter.npos; idx = vfilter.find_next(idx)) {
				v_to_part[idx] = true;
				foreach(vertex_id_type vid, graph.origin_verts[idx].nbr_list) {
					v_to_part[vid] = true;
				}
			}
			size_t sub_nedges = 0, sub_nverts = 0, npins = 0;
			typedef map<edge_id_type, edge_id_type> edge_map_type;
			edge_map_type edge_map;
			for(size_t idx = v_to_part.find_first(); idx != v_to_part.npos; idx = v_to_part.find_next(idx)) {
				foreach(vertex_id_type nbr, graph.origin_verts[idx].nbr_list) {
					edge_id_type eid = graph.origin_verts[idx].edge_list[nbr];
					if(edge_map.count(eid) == 0) {
						edge_map.insert(pair<edge_id_type, edge_id_type>(eid, sub_nedges));
						sub_nedges++;
					}
				}
				npins += graph.origin_verts[idx].edge_list.size();
			}
			sub_nverts = v_to_part.count();

			size_t assign_counter = 0;

			typedef pair<vertex_id_type, vertex_id_type> edge_pair_type;

			foreach(basic_graph::edge_type& e, graph.origin_edges) {
				if(vfilter[e.source] == false && vfilter[e.target] == false) {
					// check if is sparse
					// greedy assign
					basic_graph::part_t assignment;
					assignment = edge_to_part_greedy(graph.origin_verts[e.source], graph.origin_verts[e.target], graph.parts_counter, false);
					assign_edge(graph, e.eid, assignment);
					assign_counter++;
				}
			}
			cout << "Edges assigned: " << assign_counter << endl;

			foreach(basic_graph::edge_type& e, graph.origin_edges) {
				if(vfilter[e.source] == true || vfilter[e.target] == true) {
					// check if is dense
					// greedy assign
					basic_graph::part_t assignment;
					assignment = edge_to_part_greedy(graph.origin_verts[e.source], graph.origin_verts[e.target], graph.parts_counter, false);
					assign_edge(graph, e.eid, assignment);
					assign_counter++;
				}
			}
			cout << "Edges assigned: " << assign_counter << endl;

			cout << "Time elapsed: " << ti.elapsed() << endl;

			report_performance(graph, nparts);
		} // end of greedy_reorder

	} // end of namespace partition_strategy

} // end of namespace graphp

#endif
