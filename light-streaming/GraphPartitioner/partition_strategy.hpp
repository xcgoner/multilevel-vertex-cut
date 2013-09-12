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
#include "basic_graph.hpp"
#include "util.hpp"

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

		void report_performance(const basic_graph& graph, basic_graph::part_t nparts) {
			// count the vertex-cut
			size_t vertex_cut_counter = 0;

			// count the average degree
			size_t cutted_vertex_num = 0, boundary_degree = 0;

			foreach(const basic_graph::vertex_type& v, graph.verts) {
				if(v.vid == -1)
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
				cout << "Partition " << i << ": " << graph.parts_counter[i] << " edges" << endl;
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

		}

		void random_partition(basic_graph& graph, basic_graph::part_t nparts) {
			boost::timer ti;

			typedef pair<vertex_id_type, vertex_id_type> edge_pair_type;
			foreach(basic_graph::edge_type& e, graph.edges_storage) {
				// random assign
				const edge_pair_type edge_pair(min(e.source, e.target), max(e.source, e.target));
				basic_graph::part_t assignment;
				assignment = edge_hashing(edge_pair, hashing_seed) % (nparts);
				//assignment = edgernd(gen) % (nparts)
				assign_edge(graph, e.eid, assignment);
			}

			cout << "Time elapsed: " << ti.elapsed() << endl;

			report_performance(graph, nparts);
		}

		basic_graph::part_t edge_to_part_greedy(const basic_graph::vertex_type& source_v,
			const basic_graph::vertex_type& target_v,
			const vector<size_t>& part_num_edges,
			bool usehash = false,
			bool unbalanced = false
			) {
				const size_t nparts = part_num_edges.size();

				// compute the score of each part
				basic_graph::part_t best_part = -1;
				double maxscore = 0.0;
				double epsilon = 1.0;
				vector<double> part_score(nparts);
				size_t minedges = *min_element(part_num_edges.begin(), part_num_edges.end());
				size_t maxedges = *max_element(part_num_edges.begin(), part_num_edges.end());

				if(unbalanced) {
					for(size_t i = 0; i < nparts; ++i) {
						size_t sd = source_v.mirror_list[i] + (usehash && (source_v.vid % nparts == i));
						size_t td = target_v.mirror_list[i] + (usehash && (source_v.vid % nparts == i));
						part_score[i] = ((sd > 0) + (td > 0));
					}
				}
				else {
					for(size_t i = 0; i < nparts; ++i) {
						size_t sd = source_v.mirror_list[i] + (usehash && (source_v.vid % nparts == i));
						size_t td = target_v.mirror_list[i] + (usehash && (source_v.vid % nparts == i));
						double bal = (maxedges - part_num_edges[i]) / (epsilon + maxedges - minedges);
						part_score[i] = bal + ((sd > 0) + (td > 0));
					}
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
				const edge_pair_type edge_pair(min(source_v.vid, target_v.vid),
					max(source_v.vid, target_v.vid));
				best_part = top_parts[edge_hashing(edge_pair) % top_parts.size()];

				return best_part;
		}

		basic_graph::part_t edge_to_part_greedy(const basic_graph::vertex_type& source_v,
			const basic_graph::vertex_type& target_v,
			const vector<basic_graph::part_t>& candidates,
			const vector<size_t>& part_num_edges,
			bool usehash = false
			) {
				const size_t nparts = part_num_edges.size();

				// compute the score of each part
				basic_graph::part_t best_part = -1;
				double maxscore = 0.0;
				double epsilon = 1.0;
				vector<double> part_score(candidates.size());
				size_t minedges = *min_element(part_num_edges.begin(), part_num_edges.end());
				size_t maxedges = *max_element(part_num_edges.begin(), part_num_edges.end());

				for(size_t j = 0; j < candidates.size(); ++j) {
					basic_graph::part_t i = candidates[j];
					size_t sd = source_v.mirror_list[i] + (usehash && (source_v.vid % nparts == i));
					size_t td = target_v.mirror_list[i] + (usehash && (source_v.vid % nparts == i));
					double bal = (maxedges - part_num_edges[i]) / (epsilon + maxedges - minedges);
					part_score[j] = bal + ((sd > 0) + (td > 0));
				}

				maxscore = *max_element(part_score.begin(), part_score.end());

				vector<basic_graph::part_t> top_parts;
				for (size_t j = 0; j < candidates.size(); ++j) {
					if(fabs(part_score[j] - maxscore) < 1e-5) {
						top_parts.push_back(candidates[j]);
					}
				}

				// hash the edge to one of the best parts
				typedef pair<vertex_id_type, vertex_id_type> edge_pair_type;
				const edge_pair_type edge_pair(min(source_v.vid, target_v.vid),
					max(source_v.vid, target_v.vid));
				best_part = top_parts[edge_hashing(edge_pair) % top_parts.size()];

				return best_part;
		}

		void greedy_partition(basic_graph& graph, basic_graph::part_t nparts) {
			boost::timer ti;

			typedef pair<vertex_id_type, vertex_id_type> edge_pair_type;
			foreach(basic_graph::edge_type& e, graph.edges_storage) {
				// greedy assign
				basic_graph::part_t assignment;
				assignment = edge_to_part_greedy(graph.getVert(e.source), graph.getVert(e.target), graph.parts_counter, false);
				assign_edge(graph, e.eid, assignment);
				//cout << e.eid << " " << e.source << " " << e.target << " " << e.weight << " " << e.placement << endl;
			}

			cout << "Time elapsed: " << ti.elapsed() << endl;

			report_performance(graph, nparts);
		}

		basic_graph::part_t edge_to_part_greedy2(const basic_graph::vertex_type& source_v,
			const basic_graph::vertex_type& target_v,
			const vector<size_t>& part_num_edges,
			bool usehash = false,
			bool unbalanced = false
			) {
				const size_t nparts = part_num_edges.size();

				// compute the score of each part
				basic_graph::part_t best_part = -1;
				double maxscore = 0.0;
				double epsilon = 1.0;
				vector<double> part_score(nparts);
				size_t minedges = *min_element(part_num_edges.begin(), part_num_edges.end());
				size_t maxedges = *max_element(part_num_edges.begin(), part_num_edges.end());

				// greedy for degree
				double sum = source_v.nbr_list.size() + target_v.nbr_list.size();
				double s = target_v.nbr_list.size() / sum + 1;
				double t = source_v.nbr_list.size() / sum + 1;

				if(unbalanced) {
					for(size_t i = 0; i < nparts; ++i) {
						size_t sd = source_v.mirror_list[i] + (usehash && (source_v.vid % nparts == i));
						size_t td = target_v.mirror_list[i] + (usehash && (source_v.vid % nparts == i));
						part_score[i] = ((sd > 0) * s + (td > 0) * t);
					}
				}
				else {
					for(size_t i = 0; i < nparts; ++i) {
						size_t sd = source_v.mirror_list[i] + (usehash && (source_v.vid % nparts == i));
						size_t td = target_v.mirror_list[i] + (usehash && (source_v.vid % nparts == i));
						double bal = (maxedges - part_num_edges[i]) / (epsilon + maxedges - minedges);
						part_score[i] = bal + ((sd > 0) * s + (td > 0) * t);
					}
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
				const edge_pair_type edge_pair(min(source_v.vid, target_v.vid),
					max(source_v.vid, target_v.vid));
				best_part = top_parts[edge_hashing(edge_pair) % top_parts.size()];

				return best_part;
		}

		void greedy_partition2(basic_graph& graph, basic_graph::part_t nparts) {
			boost::timer ti;

			typedef pair<vertex_id_type, vertex_id_type> edge_pair_type;
			foreach(basic_graph::edge_type& e, graph.edges_storage) {
				// random assign
				basic_graph::part_t assignment;
				assignment = edge_to_part_greedy(graph.getVert(e.source), graph.getVert(e.target), graph.parts_counter, false);
				assign_edge(graph, e.eid, assignment);
			}

			cout << "Time elapsed: " << ti.elapsed() << endl;

			report_performance(graph, nparts);
		}

	} // end of namespace partition_strategy

} // end of namespace graphp

#endif
