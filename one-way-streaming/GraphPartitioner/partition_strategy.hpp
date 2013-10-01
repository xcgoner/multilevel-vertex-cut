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


#ifndef GRAPH_PARTITIONER_GRAPH_PARTITION_STRATEGY
#define GRAPH_PARTITIONER_GRAPH_PARTITION_STRATEGY

#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random.hpp>
#include <boost/timer.hpp>
#include <queue>
#include <stack>
#include "basic_graph.hpp"
#include "util.hpp"

#include <omp.h>

#define NUM_THREADS 6

using namespace std;
#ifdef __GNUC__
using namespace __gnu_cxx;
#endif

namespace graphp {

	namespace partition_strategy {

		static size_t mix(size_t state) {
			state += (state << 12);
			state ^= (state >> 22);
			state += (state << 4);
			state ^= (state >> 9);
			state += (state << 10);
			state ^= (state >> 2);
			state += (state << 7);
			state ^= (state >> 12);
			return state;
		  }

		static basic_graph::part_t edge_hashing (const pair<vertex_id_type, vertex_id_type>& e, const uint32_t seed = 5) {
			// a bunch of random numbers
			#if (__SIZEOF_PTRDIFF_T__ == 8)
			static const size_t a[8] = {0x6306AA9DFC13C8E7,
				0xA8CD7FBCA2A9FFD4,
				0x40D341EB597ECDDC,
				0x99CFA1168AF8DA7E,
				0x7C55BCC3AF531D42,
				0x1BC49DB0842A21DD,
				0x2181F03B1DEE299F,
				0xD524D92CBFEC63E9};
			#else
			static const size_t a[8] = {0xFC13C8E7,
				0xA2A9FFD4,
				0x597ECDDC,
				0x8AF8DA7E,
				0xAF531D42,
				0x842A21DD,
				0x1DEE299F,
				0xBFEC63E9};
			#endif
			vertex_id_type src = e.first;
			vertex_id_type dst = e.second;
			return (mix(src^a[seed%8]))^(mix(dst^a[(seed+1)%8]));
		}

		static const size_t hashing_seed = 3;

		boost::random::mt19937 gen;
		boost::random::uniform_int_distribution<> edgernd;

		void assign_edge(basic_graph& graph, edge_id_type eid, basic_graph::part_t assignment) {
			// assign edge
			basic_graph::edge_type& e = graph.getEdge(eid);
			e.placement = assignment;
			graph.parts_counter[assignment]++;

			// assign vertex
			basic_graph::vertex_type& source = graph.getVert(e.source);
			basic_graph::vertex_type& target = graph.getVert(e.target);
			source.mirror_list[assignment] = true;
			target.mirror_list[assignment] = true;
		}
		void assign_edge(basic_graph& graph, basic_graph::edge_type& e, basic_graph::part_t assignment) {
			// assign edge
			e.placement = assignment;
			graph.parts_counter[assignment]++;

			// assign vertex
			basic_graph::vertex_type& source = graph.getVert(e.source);
			basic_graph::vertex_type& target = graph.getVert(e.target);
			source.mirror_list[assignment] = true;
			target.mirror_list[assignment] = true;
		}

		void report_performance(const basic_graph& graph, basic_graph::part_t nparts) {
			// count the vertex-cut
			size_t vertex_cut_counter = 0;

			// count the average degree
			size_t cutted_vertex_num = 0, boundary_degree = 0;

			foreach(const basic_graph::vertex_type& v, graph.verts) {
				if(v.isFree())
					continue;
				if(v.mirror_list.count() > 0)
					vertex_cut_counter += (v.mirror_list.count() - 1);
				if(v.mirror_list.count() > 1) {
					// is a boundary vertex
					cutted_vertex_num++;
					boundary_degree += v.mirror_list.count();
				}
			}

			// report
			size_t max_parts = 0;
			for(size_t i = 0; i < nparts; i++) {
				//cout << "Partition " << i << ": " << graph.parts_counter[i] << " edges" << endl;
				if(max_parts < graph.parts_counter[i])
					max_parts = graph.parts_counter[i];
			}
			cout << "Vertex-cut: " << vertex_cut_counter << endl;

			cout << "Normalized replication factor: " << 1.0 * (graph.nverts + vertex_cut_counter) / graph.nverts << endl;

			cout << "Partitioning imbalance: " << 1.0 * max_parts / (graph.nedges / nparts) << endl;

			//cout << "Average degree: " << 1.0 * boundary_degree / cutted_vertex_num << " : " << 2.0 * graph.origin_edges.size() / graph.origin_verts.size() << endl;

			cout << nparts << " "
			<< vertex_cut_counter << " "
			<< 1.0 * (graph.nverts + vertex_cut_counter) / graph.nverts << " "
			<< 1.0 * max_parts / (graph.nedges / nparts) << endl;

		} // end of report performance

		struct report_result {
			size_t nparts;
			size_t vertex_cut_counter;
			double replica_factor;
			double imbalance;
			double runtime;
		};
		void report_performance(const basic_graph& graph, basic_graph::part_t nparts, report_result& result) {
			// count the vertex-cut
			size_t vertex_cut_counter = 0;

			// count the average degree
			size_t cutted_vertex_num = 0, boundary_degree = 0;

			foreach(const basic_graph::vertex_type& v, graph.verts) {
				if(v.isFree())
					continue;
				if(v.mirror_list.count() > 0)
					vertex_cut_counter += (v.mirror_list.count() - 1);
				if(v.mirror_list.count() > 1) {
					// is a boundary vertex
					cutted_vertex_num++;
					boundary_degree += v.mirror_list.count();
				}
			}

			// report
			size_t max_parts = 0;
			for(size_t i = 0; i < nparts; i++) {
				//cout << "Partition " << i << ": " << graph.parts_counter[i] << " edges" << endl;
				if(max_parts < graph.parts_counter[i])
					max_parts = graph.parts_counter[i];
			}
			cout << "Vertex-cut: " << vertex_cut_counter << endl;
			cout << (graph.nverts + vertex_cut_counter) << endl; cout << graph.nverts << endl;
			cout << "Normalized replication factor: " << 1.0 * (graph.nverts + vertex_cut_counter) / graph.nverts << endl;

			cout << "Partitioning imbalance: " << 1.0 * max_parts / (graph.nedges / nparts) << endl;

			//cout << "Average degree: " << 1.0 * boundary_degree / cutted_vertex_num << " : " << 2.0 * graph.origin_edges.size() / graph.origin_verts.size() << endl;

			result.nparts = nparts;
			result.vertex_cut_counter = vertex_cut_counter;
			result.replica_factor = 1.0 * (graph.nverts + vertex_cut_counter) / graph.nverts;
			result.imbalance = 1.0 * max_parts / (graph.nedges / nparts);

			cout << nparts << " "
				<< vertex_cut_counter << " "
				<< result.replica_factor << " "
				<< result.imbalance << endl;

		} // end of report performance
		void report_performance(const basic_graph& graph, basic_graph::part_t nparts, size_t vertex_cut_counter, report_result& result) {
			// report
			size_t max_parts = 0;
			for(size_t i = 0; i < nparts; i++) {
				//cout << "Partition " << i << ": " << graph.parts_counter[i] << " edges" << endl;
				if(max_parts < graph.parts_counter[i])
					max_parts = graph.parts_counter[i];
			}
			cout << "Vertex-cut: " << vertex_cut_counter << endl;
			cout << (graph.nverts + vertex_cut_counter) << endl; cout << graph.nverts << endl;
			cout << "Normalized replication factor: " << 1.0 * (graph.nverts + vertex_cut_counter) / graph.nverts << endl;

			cout << "Partitioning imbalance: " << 1.0 * max_parts / (graph.nedges / nparts) << endl;

			//cout << "Average degree: " << 1.0 * boundary_degree / cutted_vertex_num << " : " << 2.0 * graph.origin_edges.size() / graph.origin_verts.size() << endl;

			result.nparts = nparts;
			result.vertex_cut_counter = vertex_cut_counter;
			result.replica_factor = 1.0 * (graph.nverts + vertex_cut_counter) / graph.nverts;
			result.imbalance = 1.0 * max_parts / (graph.nedges / nparts);

			cout << nparts << " "
				<< vertex_cut_counter << " "
				<< result.replica_factor << " "
				<< result.imbalance << endl;

		} // end of report performance for the total vertex-cut is already summed up

		// edge order

		void random_partition(basic_graph& graph, basic_graph::part_t nparts) {
			typedef pair<vertex_id_type, vertex_id_type> edge_pair_type;
			for(vector<basic_graph::edge_type>::iterator itr = graph.ebegin; itr != graph.eend; ++itr)  {
				basic_graph::edge_type& e = *itr;
				// random assign
				const edge_pair_type edge_pair(min(e.source, e.target), max(e.source, e.target));
				basic_graph::part_t assignment;
				assignment = edge_hashing(edge_pair, hashing_seed) % (nparts);
				//assignment = edgernd(gen) % (nparts)
				assign_edge(graph, e, assignment);
			}
		}

		void chunking_partition(basic_graph& graph, basic_graph::part_t nparts) {
			const size_t C = 16;
			size_t idx = 0;
			for(vector<basic_graph::edge_type>::iterator itr = graph.ebegin; itr != graph.eend; ++itr, ++idx)  {
				basic_graph::edge_type& e = *itr;
				// random assign
				basic_graph::part_t assignment;
				assignment = (idx / C) % nparts;
				assign_edge(graph, e, assignment);
			}
		}

		// weighted / deterministic
		basic_graph::part_t edge_to_part_weighted(basic_graph& graph, 
			const basic_graph::vertex_id_type source,
			const basic_graph::vertex_id_type target,
			const vector<size_t>& part_num_edges,
			const size_t type
			) {
				const size_t nparts = part_num_edges.size();

				const basic_graph::vertex_type& source_v = graph.getVert(source);
				const basic_graph::vertex_type& target_v = graph.getVert(target);

				// compute the score of each part
				basic_graph::part_t best_part = -1;
				double maxscore = 0.0;
				double epsilon = 1.0;
				vector<double> part_score(nparts);

				size_t minedges = *min_element(part_num_edges.begin(), part_num_edges.end());
				size_t maxedges = *max_element(part_num_edges.begin(), part_num_edges.end());

				for(size_t i = 0; i < nparts; ++i) {
					size_t sd = source_v.mirror_list[i];
					size_t td = target_v.mirror_list[i];
					double weight;
					if(type == 0) {
						weight = (maxedges - part_num_edges[i]) / (epsilon + maxedges - minedges);
					}
					else if(type == 1) {
						weight = 1 - 1.0 * part_num_edges[i] / maxedges;
					}
					else {
						weight = 1 - exp(1.0 * part_num_edges[i] - maxedges);
					}
					part_score[i] = weight * ((sd > 0) + (td > 0));
				}

				maxscore = *max_element(part_score.begin(), part_score.end());

				vector<basic_graph::part_t> top_parts;
				for(size_t i = 0; i < nparts; ++i) {
					if(fabs(part_score[i] - maxscore) < 1e-5) {
						top_parts.push_back(i);
					}
				}

				// hash the edge to one of the best parts
				typedef pair<vertex_id_type, vertex_id_type> edge_pair_type;
				const edge_pair_type edge_pair(min(source, target),
					max(source, target));
				best_part = top_parts[edge_hashing(edge_pair) % top_parts.size()];

				return best_part;
		}// end of edge_to_part_weighted
		void weighted_partition(basic_graph& graph, basic_graph::part_t nparts, size_t type = 0) {
			for(vector<basic_graph::edge_type>::iterator itr = graph.ebegin; itr != graph.eend; ++itr)  {
				basic_graph::edge_type& e = *itr;
				// greedy assign by specific type
				basic_graph::part_t assignment;
				assignment = edge_to_part_weighted(graph, e.source, e.target, graph.parts_counter, type);
				assign_edge(graph, e, assignment);
			}
		}

		// powergraph
		basic_graph::part_t edge_to_part_powergraph(basic_graph& graph, 
			const basic_graph::vertex_id_type source,
			const basic_graph::vertex_id_type target,
			const vector<size_t>& part_num_edges
			) {
				const size_t nparts = part_num_edges.size();

				const basic_graph::vertex_type& source_v = graph.getVert(source);
				const basic_graph::vertex_type& target_v = graph.getVert(target);

				// compute the score of each part
				basic_graph::part_t best_part = -1;
				double maxscore = 0.0;
				double epsilon = 1.0;
				vector<double> part_score(nparts);
				size_t minedges = *min_element(part_num_edges.begin(), part_num_edges.end());
				size_t maxedges = *max_element(part_num_edges.begin(), part_num_edges.end());

				for(size_t i = 0; i < nparts; ++i) {
					size_t sd = source_v.mirror_list[i];
					size_t td = target_v.mirror_list[i];
					double bal = (maxedges - part_num_edges[i]) / (epsilon + maxedges - minedges);
					part_score[i] = bal + ((sd > 0) + (td > 0));
				}

				maxscore = *max_element(part_score.begin(), part_score.end());

				vector<basic_graph::part_t> top_parts;
				for(size_t i = 0; i < nparts; ++i) {
					if(fabs(part_score[i] - maxscore) < 1e-5) {
						top_parts.push_back(i);
					}
				}

				// hash the edge to one of the best parts
				typedef pair<vertex_id_type, vertex_id_type> edge_pair_type;
				const edge_pair_type edge_pair(min(source, target),
					max(source, target));
				best_part = top_parts[edge_hashing(edge_pair) % top_parts.size()];

				return best_part;
		}
		void powergraph_partition(basic_graph& graph, basic_graph::part_t nparts) {
			for(vector<basic_graph::edge_type>::iterator itr = graph.ebegin; itr != graph.eend; ++itr)  {
				basic_graph::edge_type& e = *itr;
				// greedy assign
				basic_graph::part_t assignment;
				assignment = edge_to_part_powergraph(graph, e.source, e.target, graph.parts_counter);
				assign_edge(graph, e, assignment);
				//cout << e.eid << " " << e.source << " " << e.target << " " << e.weight << " " << e.placement << endl;
			}
		}

		basic_graph::part_t edge_to_part_degree(basic_graph& graph, 
			const basic_graph::vertex_id_type source,
			const basic_graph::vertex_id_type target,
			const size_t source_degree,
			const size_t target_degree,
			const vector<size_t>& part_num_edges
			) {
				const size_t nparts = part_num_edges.size();

				const basic_graph::vertex_type& source_v = graph.getVert(source);
				const basic_graph::vertex_type& target_v = graph.getVert(target);

				// compute the score of each part
				basic_graph::part_t best_part = -1;
				double maxscore = 0.0;
				double epsilon = 1.0;
				vector<double> part_score(nparts);
				size_t minedges = *min_element(part_num_edges.begin(), part_num_edges.end());
				size_t maxedges = *max_element(part_num_edges.begin(), part_num_edges.end());

				// greedy for degree
				// nbr_list is not used in streaming partitioning
				//double sum = source_v.nbr_list.size() + target_v.nbr_list.size();
				//double s = target_v.nbr_list.size() / sum + 1;
				//double t = source_v.nbr_list.size() / sum + 1;
				// use degree in streaming partitioning
				double sum = source_degree + target_degree;
				double s = target_degree / sum + 1;
				double t = source_degree / sum + 1;

				for(size_t i = 0; i < nparts; ++i) {
					size_t sd = source_v.mirror_list[i];
					size_t td = target_v.mirror_list[i];
					double bal = (maxedges - part_num_edges[i]) / (epsilon + maxedges - minedges);
					part_score[i] = bal + ((sd > 0) * s + (td > 0) * t);
				}

				maxscore = *max_element(part_score.begin(), part_score.end());

				vector<basic_graph::part_t> top_parts;
				for(size_t i = 0; i < nparts; ++i) {
					if(fabs(part_score[i] - maxscore) < 1e-5) {
						top_parts.push_back(i);
					}
				}

				// hash the edge to one of the best parts
				typedef pair<vertex_id_type, vertex_id_type> edge_pair_type;
				const edge_pair_type edge_pair(min(source, target),
					max(source, target));
				best_part = top_parts[edge_hashing(edge_pair) % top_parts.size()];

				return best_part;
		}
		void degree_partition(basic_graph& graph, basic_graph::part_t nparts) {
			for(vector<basic_graph::edge_type>::iterator itr = graph.ebegin; itr != graph.eend; ++itr)  {
				basic_graph::edge_type& e = *itr;
				// greedy assign
				basic_graph::part_t assignment;
				graph.getVert(e.source).degree++;
				graph.getVert(e.target).degree++;
				assignment = edge_to_part_degree(graph, e.source, e.target, graph.getVert(e.source).degree, graph.getVert(e.target).degree, graph.parts_counter);
				assign_edge(graph, e, assignment);
			}
		}

		// vertex order

		void vertex_reorder(basic_graph& graph, vector<basic_graph::vertex_id_type>& vertex_order, size_t type) {
			vertex_order.clear();
			if(type == 0) {
				// random
				for(size_t vid = graph.vmap.find_first(); vid != graph.vmap.npos; vid = graph.vmap.find_next(vid)) {
					vertex_order.push_back(vid);
				}
				random_shuffle(vertex_order.begin(), vertex_order.end());
			}// end of random order
			else if(type == 1) {
				// bfs
				queue<basic_graph::vertex_id_type> vq;
				size_t vcounter = graph.vmap.count();
				boost::dynamic_bitset<> vmap(graph.vmap);
				while(vcounter > 0) {
					// find a vertex not visited
					size_t vid = vmap.find_first();
					// push into the queue
					vq.push(vid);
					while(!vq.empty()) {
						// visit the top
						vid = vq.front();
						vq.pop();
						// if already visited, continue
						if(vmap[vid] == false)
							continue;
						// if not visited
						vcounter--;
						vmap[vid] = false;
						vertex_order.push_back(vid);
						const basic_graph::vertex_type& v = graph.getVert(vid);
						// get all the neighbours
						for(size_t ebegin = v.edge_begin; ebegin < v.edge_end; ebegin++) {
							const basic_graph::edge_type& e = graph.getEdge(ebegin);
							if(e.source != vid) {
								cerr << "There must be something wrong about vertex reorder ..." << endl;
								exit(0);
							}
							// push all neighbours that are not visited
							if(vmap[e.target] == true) {
								vq.push(e.target);
							}
						}
					}
				}

				// just for test
				// deprecate later ...
				//foreach(basic_graph::vertex_id_type vid, vertex_order) {
				//	cout << vid << ", ";
				//}
				//cout << endl;
			}// end of bfs order
			else if(type == 2) {
				// dfs
				stack<basic_graph::vertex_id_type> vs;
				size_t vcounter = graph.vmap.count();
				boost::dynamic_bitset<> vmap(graph.vmap);
				while(vcounter > 0) {
					// find a vertex not visited
					size_t vid = vmap.find_first();
					// push into the stack
					vs.push(vid);
					while(!vs.empty()) {
						// visit the top
						vid = vs.top();
						vs.pop();
						// if already visited, continue
						if(vmap[vid] == false)
							continue;
						// if not visited
						vcounter--;
						vmap[vid] = false;
						vertex_order.push_back(vid);
						const basic_graph::vertex_type& v = graph.getVert(vid);
						// get all the neighbours
						for(int ebegin = v.edge_end - 1; ebegin >= v.edge_begin && ebegin != -1; ebegin--) {
							const basic_graph::edge_type& e = graph.getEdge(ebegin);
							if(e.source != vid) {
								cerr << "There must be something wrong about vertex reorder ..." << endl;
								exit(0);
							}
							// push all neighbours that are not visited
							if(vmap[e.target] == true) {
								vs.push(e.target);
							}
						}
					}
				}
				// just for test
				// deprecate later ...
				//foreach(basic_graph::vertex_id_type vid, vertex_order) {
				//	cout << vid << ", ";
				//}
				//cout << endl;
			}// end of dfs order
		}

		void v_random_partition(basic_graph& graph, basic_graph::part_t nparts, const vector<basic_graph::vertex_id_type> vertex_order) {
			foreach(basic_graph::vertex_id_type vid, vertex_order) {
				basic_graph::vertex_type& v = graph.getVert(vid);
				for(size_t eidx = v.edge_begin; eidx < v.edge_end; eidx++) {
					basic_graph::edge_type& e = graph.getEdge(eidx);
					// assign edges
					basic_graph::part_t assignment;
					assignment = edgernd(gen) % (nparts);
					assign_edge(graph, e, assignment);
				}
			}
		}

		void v_chunking_partition(basic_graph& graph, basic_graph::part_t nparts, const vector<basic_graph::vertex_id_type> vertex_order) {
			const size_t C = 16;
			size_t idx = 0;
			foreach(basic_graph::vertex_id_type vid, vertex_order) {
				basic_graph::vertex_type& v = graph.getVert(vid);
				for(size_t eidx = v.edge_begin; eidx < v.edge_end; eidx++) {
					basic_graph::edge_type& e = graph.getEdge(eidx);
					// assign edges
					basic_graph::part_t assignment;
					assignment = (idx / C) % nparts;
					assign_edge(graph, e, assignment);
					idx++;
				}
			}
		}

		void v_powergraph_partition(basic_graph& graph, basic_graph::part_t nparts, const vector<basic_graph::vertex_id_type> vertex_order) {
			foreach(basic_graph::vertex_id_type vid, vertex_order) {
				basic_graph::vertex_type& v = graph.getVert(vid);
				for(size_t eidx = v.edge_begin; eidx < v.edge_end; eidx++) {
					basic_graph::edge_type& e = graph.getEdge(eidx);
					// assign edges
					basic_graph::part_t assignment;
					assignment = edge_to_part_powergraph(graph, e.source, e.target, graph.parts_counter);
					assign_edge(graph, e, assignment);
				}
			}
		}

		void v_degree_partition(basic_graph& graph, basic_graph::part_t nparts, const vector<basic_graph::vertex_id_type> vertex_order) {
			foreach(basic_graph::vertex_id_type vid, vertex_order) {
				basic_graph::vertex_type& v = graph.getVert(vid);
				v.degree += (v.edge_end - v.edge_begin);
				for(size_t eidx = v.edge_begin; eidx < v.edge_end; eidx++) {
					basic_graph::edge_type& e = graph.getEdge(eidx);
					graph.getVert(e.target).degree++;
					// assign edges
					basic_graph::part_t assignment;
					assignment = edge_to_part_degree(graph, e.source, e.target, graph.getVert(e.source).degree, graph.getVert(e.target).degree, graph.parts_counter);
					assign_edge(graph, e, assignment);
				}
			}
		}

		void v_weighted_partition(basic_graph& graph, basic_graph::part_t nparts, const vector<basic_graph::vertex_id_type> vertex_order, size_t type = 0) {
			foreach(basic_graph::vertex_id_type vid, vertex_order) {
				basic_graph::vertex_type& v = graph.getVert(vid);
				for(size_t eidx = v.edge_begin; eidx < v.edge_end; eidx++) {
					basic_graph::edge_type& e = graph.getEdge(eidx);
					// assign edges
					basic_graph::part_t assignment;
					assignment = edge_to_part_weighted(graph, e.source, e.target, graph.parts_counter, type);
					assign_edge(graph, e, assignment);
				}
			}
		}

		// batch way
		void vbatch_random_partition(basic_graph& graph, basic_graph::part_t nparts, const vector<basic_graph::vertex_id_type> vertex_order) {
			size_t avg_degree = graph.getVert(vertex_order[0]).edge_end - graph.getVert(vertex_order[0]).edge_begin;
			basic_graph::part_t assignment;
			foreach(basic_graph::vertex_id_type vid, vertex_order) {
				basic_graph::vertex_type& v = graph.getVert(vid);
				v.degree = v.edge_begin - v.edge_end;
				if(v.degree > avg_degree) {
					for(size_t eidx = v.edge_begin; eidx < v.edge_end; eidx++) {
						basic_graph::edge_type& e = graph.getEdge(eidx);
						// assign edges
						assignment = edgernd(gen) % (nparts);
						assign_edge(graph, e, assignment);
					}
				}
				else {
					assignment = edgernd(gen) % (nparts);
					for(size_t eidx = v.edge_begin; eidx < v.edge_end; eidx++) {
						basic_graph::edge_type& e = graph.getEdge(eidx);
						// assign edges
						assign_edge(graph, e, assignment);
					}
				}
				avg_degree += v.degree;
			}
		}

		void vbatch_balanced_partition(basic_graph& graph, basic_graph::part_t nparts, const vector<basic_graph::vertex_id_type> vertex_order) {
			size_t avg_degree = graph.getVert(vertex_order[0]).edge_end - graph.getVert(vertex_order[0]).edge_begin;
			basic_graph::part_t assignment;
			foreach(basic_graph::vertex_id_type vid, vertex_order) {
				basic_graph::vertex_type& v = graph.getVert(vid);
				v.degree = v.edge_begin - v.edge_end;
				if(v.degree > avg_degree) {
					for(size_t eidx = v.edge_begin; eidx < v.edge_end; eidx++) {
						basic_graph::edge_type& e = graph.getEdge(eidx);
						// assign edges
						assignment = (min_element(graph.parts_counter.begin(), graph.parts_counter.end()) - graph.parts_counter.begin());
						assign_edge(graph, e, assignment);
					}
				}
				else {
					assignment = (min_element(graph.parts_counter.begin(), graph.parts_counter.end()) - graph.parts_counter.begin());
					for(size_t eidx = v.edge_begin; eidx < v.edge_end; eidx++) {
						basic_graph::edge_type& e = graph.getEdge(eidx);
						// assign edges
						assign_edge(graph, e, assignment);
					}
				}
				avg_degree += v.degree;
			}
		}

		// weighted / deterministic in batch
		basic_graph::part_t edge_to_part_weighted(basic_graph& graph, 
			const vector<basic_graph::vertex_id_type> sources,
			const vector<basic_graph::vertex_id_type> targets,
			const vector<size_t>& part_num_edges,
			const size_t type
			) {
				const size_t nparts = part_num_edges.size();

				// compute the score of each part
				basic_graph::part_t best_part = -1;
				double maxscore = 0.0;
				double epsilon = 1.0;
				vector<double> part_score(nparts);

				size_t minedges = *min_element(part_num_edges.begin(), part_num_edges.end());
				size_t maxedges = *max_element(part_num_edges.begin(), part_num_edges.end());

				for(size_t i = 0; i < nparts; ++i) {
					size_t d = 0;
					for(size_t j = 0; j < sources.size(); j++) {
						const basic_graph::vertex_type& source_v = graph.getVert(sources[j]);
						const basic_graph::vertex_type& target_v = graph.getVert(targets[j]);
						d += source_v.mirror_list[i];
						d += target_v.mirror_list[i];
					}
					double weight;
					if(type == 0) {
						weight = (maxedges - part_num_edges[i]) / (epsilon + maxedges - minedges);
					}
					else if(type == 1) {
						weight = 1 - 1.0 * part_num_edges[i] / maxedges;
					}
					else {
						weight = 1 - exp(1.0 * part_num_edges[i] - maxedges);
					}
					part_score[i] = weight * d;
				}

				maxscore = *max_element(part_score.begin(), part_score.end());

				vector<basic_graph::part_t> top_parts;
				for(size_t i = 0; i < nparts; ++i) {
					if(fabs(part_score[i] - maxscore) < 1e-5) {
						top_parts.push_back(i);
					}
				}

				// hash the edge to one of the best parts
				best_part = top_parts[edgernd(gen) % top_parts.size()];

				return best_part;
		}// end of edge_to_part_weighted in batch
		void vbatch_weighted_partition(basic_graph& graph, basic_graph::part_t nparts, const vector<basic_graph::vertex_id_type> vertex_order, size_t type = 0) {
			size_t avg_degree = graph.getVert(vertex_order[0]).edge_end - graph.getVert(vertex_order[0]).edge_begin;
			basic_graph::part_t assignment;
			foreach(basic_graph::vertex_id_type vid, vertex_order) {
				basic_graph::vertex_type& v = graph.getVert(vid);
				v.degree = v.edge_begin - v.edge_end;
				if(v.degree > avg_degree) {
					for(size_t eidx = v.edge_begin; eidx < v.edge_end; eidx++) {
						basic_graph::edge_type& e = graph.getEdge(eidx);
						// assign edges
						assignment = edge_to_part_weighted(graph, e.source, e.target, graph.parts_counter, type);
						assign_edge(graph, e, assignment);
					}
				}
				else {
					vector<basic_graph::vertex_id_type> sources;
					vector<basic_graph::vertex_id_type> targets;
					for(size_t eidx = v.edge_begin; eidx < v.edge_end; eidx++) {
						basic_graph::edge_type& e = graph.getEdge(eidx);
						// construct the batch vector
						sources.push_back(e.source);
						targets.push_back(e.target);
					}
					assignment = edge_to_part_weighted(graph, sources, targets, graph.parts_counter, type);
					for(size_t eidx = v.edge_begin; eidx < v.edge_end; eidx++) {
						basic_graph::edge_type& e = graph.getEdge(eidx);
						// assign edges
						assign_edge(graph, e, assignment);
					}
				}
				avg_degree += v.degree;
			}
		}

		void run_partition(basic_graph& graph, vector<basic_graph::part_t>& nparts, vector<string>& strategies, vector<string>& orders, string type) {
			vector<basic_graph::vertex_id_type> vertex_order;

			vector<report_result> result_table(orders.size() * strategies.size() * nparts.size());
			//remember to clear the degree before each partitioning ...
			for(size_t k = 0; k < orders.size(); k++) {
				string order = orders[k];
				if(order == "random" && type == "edge")
					vertex_reorder(graph, vertex_order, 0);
				else if(order == "bfs")
					vertex_reorder(graph, vertex_order, 1);
				else if(order == "dfs")
					vertex_reorder(graph, vertex_order, 2);

				if(order == "random" && type == "edge") {
					random_shuffle(graph.ebegin, graph.eend);
				}

				for(size_t j = 0; j < strategies.size(); j++) {
					// select the streaming type
					if(type == "edge") {
						// select the strategy
						void (*partition_func)(basic_graph& graph, basic_graph::part_t nparts);
						string strategy = strategies[j];
						if(strategy == "random")
							partition_func = random_partition;
						if(strategy == "chunking")
							partition_func = chunking_partition;
						else if(strategy == "powergraph")
							partition_func = powergraph_partition;
						else if(strategy == "degree")
							partition_func = degree_partition;

						for(size_t i = 0; i < nparts.size(); i++) {
							// initialize
							graph.initialize(nparts[i]);

							cout << endl << strategy << endl;

							boost::timer ti;
							double runtime;

							if(strategy == "weighted0") {
								weighted_partition(graph, nparts[i], 0);
							}
							else if(strategy == "weighted1") {
								weighted_partition(graph, nparts[i], 1);
							}
							else if(strategy == "weighted2") {
								weighted_partition(graph, nparts[i], 2);
							}
							else {
								partition_func(graph, nparts[i]);
							}

							runtime = ti.elapsed();
							cout << "Time elapsed: " << runtime << endl;

							size_t rid = k * strategies.size() * nparts.size() + j * nparts.size() + i;
							report_performance(graph, nparts[i], result_table[rid]);
							result_table[rid].runtime = runtime;
						}
					}// end of edge streaming
					else {
						// select the strategy
						void (*partition_func)(basic_graph& graph, basic_graph::part_t nparts, const vector<basic_graph::vertex_id_type> vertex_order);
						string strategy = strategies[j];
						if(strategy == "random")
							partition_func = v_random_partition;
						if(strategy == "chunking")
							partition_func = v_chunking_partition;
						else if(strategy == "powergraph")
							partition_func = v_powergraph_partition;
						else if(strategy == "degree")
							partition_func = v_degree_partition;

						for(size_t i = 0; i < nparts.size(); i++) {
							// initialize
							graph.initialize(nparts[i]);

							cout << endl << strategy << endl;

							boost::timer ti;
							double runtime;

							if(strategy == "weighted0") {
								v_weighted_partition(graph, nparts[i], vertex_order, 0);
							}
							else if(strategy == "weighted1") {
								v_weighted_partition(graph, nparts[i], vertex_order, 1);
							}
							else if(strategy == "weighted2") {
								v_weighted_partition(graph, nparts[i], vertex_order, 2);
							}
							else {
								partition_func(graph, nparts[i], vertex_order);
							}

							runtime = ti.elapsed();
							cout << "Time elapsed: " << runtime << endl;

							size_t rid = k * strategies.size() * nparts.size() + j * nparts.size() + i;
							report_performance(graph, nparts[i], result_table[rid]);
							result_table[rid].runtime = runtime;
						}
					}// end of vertex order
				}
			}

			cout << endl;
			// report the table
			foreach(const report_result& result, result_table) {
				cout << result.nparts << "\t" 
					<< result.vertex_cut_counter << "\t" 
					<< result.replica_factor << "\t" 
					<< result.imbalance << "\t" 
					<< result.runtime 
					<< endl;
			}
		} // run partition in one thread

	} // end of namespace partition_strategy

} // end of namespace graphp

#endif
