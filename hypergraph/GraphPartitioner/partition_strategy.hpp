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
			basic_graph::edge_type& e = graph.origin_edges[eid];
			e.placement = assignment;
			graph.parts_counter[assignment]++;

			// assign vertex
			basic_graph::vertex_type& source = graph.origin_verts[e.source];
			basic_graph::vertex_type& target = graph.origin_verts[e.target];
			source.mirror_list.insert(assignment);
			target.mirror_list.insert(assignment);

		}

		void report_performance(const basic_graph& graph, basic_graph::part_t nparts) {
			// count the vertex-cut
			size_t vertex_cut_counter = 0;

			// count the average degree
			size_t cutted_vertex_num = 0, boundary_degree = 0;

			for(hash_map<vertex_id_type, basic_graph::vertex_type>::const_iterator iter = graph.origin_verts.begin(); iter != graph.origin_verts.end(); iter++) {
				vertex_cut_counter += (iter->second.mirror_list.size() - 1);
				if(iter->second.mirror_list.size() > 1) {
					// is a boundary vertex
					cutted_vertex_num++;
					boundary_degree += iter->second.nbr_list.size();
				}
			}

			// report
			int max_parts = 0;
			for(size_t i = 0; i < nparts; i++) {
//				cout << "Partition " << i << ": " << graph.parts_counter[i] << " edges" << endl;
				if(max_parts < graph.parts_counter[i])
					max_parts = graph.parts_counter[i];
			}
			cout << "Vertex-cut: " << vertex_cut_counter << endl;

			cout << "Normalized replication factor: " << 1.0 * (graph.nverts + vertex_cut_counter) / graph.nverts << endl;

			cout << "Partitioning imbalance: " << 1.0 * max_parts / (graph.nedges / nparts) << endl;

			cout << "Average degree: " << 1.0 * boundary_degree / cutted_vertex_num << " : " << 2.0 * graph.origin_edges.size() / graph.origin_verts.size() << endl;

			cout << nparts << " "
			<< vertex_cut_counter << " "
			<< 1.0 * (graph.nverts + vertex_cut_counter) / graph.nverts << " "
			<< 1.0 * max_parts / (graph.nedges / nparts) << endl;

		}

		void random_partition(basic_graph& graph, basic_graph::part_t nparts) {
			boost::timer ti;

			typedef pair<vertex_id_type, vertex_id_type> edge_pair_type;
			foreach(basic_graph::edge_type& e, graph.origin_edges) {
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
			bool userecent = false
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
					size_t sd = source_v.mirror_list.count(i) + (usehash && (source_v.vid % nparts == i));
					size_t td = target_v.mirror_list.count(i) + (usehash && (source_v.vid % nparts == i));
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
				const edge_pair_type edge_pair(min(source_v.vid, target_v.vid),
					max(source_v.vid, target_v.vid));
				best_part = top_parts[edge_hashing(edge_pair) % top_parts.size()];

				//if(userecent) {
				//	source_v.mirror_list.clear();
				//	target_v.mirror_list.clear();
				//}
				return best_part;
		}

		basic_graph::part_t edge_to_part_greedy(const basic_graph::vertex_type& source_v,
			const basic_graph::vertex_type& target_v,
			const vector<basic_graph::part_t>& candidates,
			const vector<size_t>& part_num_edges,
			bool usehash = false,
			bool userecent = false
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
					size_t sd = source_v.mirror_list.count(i) + (usehash && (source_v.vid % nparts == i));
					size_t td = target_v.mirror_list.count(i) + (usehash && (source_v.vid % nparts == i));
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

				//if(userecent) {
				//	source_v.mirror_list.clear();
				//	target_v.mirror_list.clear();
				//}
				return best_part;
		}

		void greedy_partition(basic_graph& graph, basic_graph::part_t nparts) {
			boost::timer ti;

			typedef pair<vertex_id_type, vertex_id_type> edge_pair_type;
			foreach(basic_graph::edge_type& e, graph.origin_edges) {
				// random assign
				basic_graph::part_t assignment;
				assignment = edge_to_part_greedy(graph.origin_verts[e.source], graph.origin_verts[e.target], graph.parts_counter, false);
				assign_edge(graph, e.eid, assignment);
			}

			cout << "Time elapsed: " << ti.elapsed() << endl;

			report_performance(graph, nparts);
		}

		void move_edge(basic_graph& graph, edge_id_type eid, basic_graph::part_t a, basic_graph::part_t b) {
			// get the vertices and edges by their IDs
			basic_graph::edge_type& e = graph.origin_edges[eid];
			basic_graph::vertex_type& source = graph.origin_verts[e.source];
			basic_graph::vertex_type& target = graph.origin_verts[e.target];

			// erase edge from a
			//e.placement = -1;
			graph.parts_counter[a]--;
			// remove vertex from a
			bool flag = false;
			foreach(vertex_id_type nbr, source.nbr_list) {
				if(source.edge_list[nbr] != eid && graph.origin_edges[source.edge_list[nbr]].placement == a) {
					// if there exists an edge which is still belong to a, then the mirror can not be erased
					flag = true;
				}
			}
			if(flag == false) {
				// if such edge do not exist, then erase
				source.mirror_list.erase(a);
			}
			flag = false;
			foreach(vertex_id_type nbr, target.nbr_list) {
				if(target.edge_list[nbr] != eid && graph.origin_edges[target.edge_list[nbr]].placement == a) {
					// if there exists an edge which is still belong to a, then the mirror can not be erased
					flag = true;
				}
			}
			if(flag == false) {
				// if such edge do not exist, then erase
				target.mirror_list.erase(a);
			}

			// assign edge to b
			e.placement = b;
			graph.parts_counter[b]++;

			// assign vertex
			source.mirror_list.insert(b);
			target.mirror_list.insert(b);

		}

		void move_gain(basic_graph& graph, edge_id_type eid, basic_graph::part_t a, basic_graph::part_t b) {

		}

		size_t cost_greedy_refinement(basic_graph& graph, basic_graph::part_t nparts) {
			boost::timer ti;

			size_t wmin = 0.9 * graph.nedges / nparts;
			size_t wmax = 1.03 * graph.nedges / nparts;

			// visit the edges in random order
			vector<vertex_id_type> vertex_order(graph.nverts);
			size_t i = 0;
			foreach(basic_graph::verts_map_type::value_type& vp, graph.origin_verts) {
				vertex_order[i] = vp.first;
				i++;
			}
			random_shuffle(vertex_order.begin(), vertex_order.end());

			size_t refine_counter = 0;

			// refine by moving mirrors
			foreach(const vertex_id_type vid, vertex_order) {
				// if the vertex is internal, do nothing
				if(graph.origin_verts[vid].mirror_list.size() <= 1)
					continue;

				// if the vertex is on the boundary, check if it could be refined

				// visit the mirrors in random order
				vector<basic_graph::part_t> parts_order(graph.origin_verts[vid].mirror_list.size());
				size_t j = 0;
				foreach(const basic_graph::part_t pt, graph.origin_verts[vid].mirror_list) {
					parts_order[j] = pt;
					j++;
				}
				random_shuffle(parts_order.begin(), parts_order.end());

				set<basic_graph::part_t> exclude_parts;

				foreach(const basic_graph::part_t pt, parts_order) {
					// check if this mirror could be refined

					// the local neighbour vertices in this partiton
					vector<vertex_id_type> local_nbr_list;
					foreach(const vertex_id_type& nbr, graph.origin_verts[vid].nbr_list) {
						if(graph.origin_edges[graph.origin_verts[vid].edge_list[nbr]].placement == pt) {
							local_nbr_list.push_back(nbr);
						}
					}

					// ???
					if(local_nbr_list.size() == 0) {
						cerr << "WTF" << endl;
						continue;
					}

					// check balance constraint
					if(graph.parts_counter[pt] - local_nbr_list.size() < wmin)
						continue;

					size_t local_nbr_counter = 0;

					foreach(const vertex_id_type local_nbr, local_nbr_list) {
						bool movable = false;
						foreach(const basic_graph::part_t other_part, parts_order) {
							// check the partition different to this one and the partition is not excluded
							if(other_part == pt || exclude_parts.count(other_part) > 0)
								continue;

							// check if the local_nbr also have a mirror in other_part
							if(graph.origin_verts[local_nbr].mirror_list.count(other_part) > 0) {
								movable = true;
								break;
							}
						}
						if(movable)
							local_nbr_counter++;
					}
					// check if all the edges are movable
					if(local_nbr_counter == local_nbr_list.size()) {
						// then move the vid vertex and all the neighbour edges to other mirror parttitions
						refine_counter++;

						// move the local_nbr in random order
						random_shuffle(local_nbr_list.begin(), local_nbr_list.end());

						foreach(const vertex_id_type local_nbr, local_nbr_list) {
							basic_graph::part_t assignment;

							vector<basic_graph::part_t> move_dst;
							foreach(const basic_graph::part_t other_part, parts_order) {
								// check the partition different to this one
								if(other_part == pt || exclude_parts.count(other_part) > 0)
									continue;

								// check if the local_nbr also have a mirror in other_part
								if(graph.origin_verts[local_nbr].mirror_list.count(other_part) > 0) {
									move_dst.push_back(other_part);
								}
							}

							if(move_dst.size() == 1)
								assignment = move_dst[0];
							else {
								const basic_graph::edge_type& e = graph.origin_edges[graph.origin_verts[vid].edge_list[local_nbr]];
								// use the candidate version of edge_to_part_greedy
								assignment = edge_to_part_greedy(graph.origin_verts[e.source], graph.origin_verts[e.target], move_dst, graph.parts_counter, true);
							}
							move_edge(graph, graph.origin_verts[vid].edge_list[local_nbr], pt, assignment);
						}
						exclude_parts.insert(pt);
					} // end check movable
				} // end check all the mirrors
			}

			// report the result
			cout << refine_counter << " mirrors are moved" << endl;

			cout << "Time elapsed: " << ti.elapsed() << endl;

			report_performance(graph, nparts);

			return refine_counter;
		}

		void bal_greedy_refinement(basic_graph& graph, basic_graph::part_t nparts) {
			boost::timer ti;

			size_t wmin = 0.9 * graph.nedges / nparts;
			size_t wmax = 1.03 * graph.nedges / nparts;

			// visit the edges in random order
			vector<edge_id_type> edge_order(graph.nedges);
			for(edge_id_type i = 0; i < graph.nedges; i++) {
				edge_order[i] = i;
			}
			random_shuffle(edge_order.begin(), edge_order.end());

			size_t move_counter = 0;

			for(size_t i = 0; i < edge_order.size(); i++) {
				edge_id_type eid = edge_order[i];

				const basic_graph::vertex_type& source = graph.origin_verts[graph.origin_edges[eid].source];
				const basic_graph::vertex_type& target = graph.origin_verts[graph.origin_edges[eid].target];

				// check if the edge is a boundary edge
				if(source.mirror_list.size() > 1 && target.mirror_list.size() > 1) {
					vector<basic_graph::part_t> intersection_parts;
					vector<basic_graph::part_t> candidates;
					set_intersection(
						target.mirror_list.begin(), target.mirror_list.end(), source.mirror_list.begin(),
						source.mirror_list.end(),
						inserter(intersection_parts, intersection_parts.begin())
						);

					// could be moved without increasing the cost
					if(intersection_parts.size() > 0) {
						const basic_graph::part_t origin_part = graph.origin_edges[eid].placement;
						foreach(basic_graph::part_t candidate, intersection_parts) {
							if(graph.parts_counter[origin_part] > graph.parts_counter[candidate])
								candidates.push_back(candidate);
						}

						// check if there exists movement that can improve the balance
						if(candidates.size() > 0) {
							basic_graph::part_t assignment;
							if(candidates.size() == 1)
								assignment = candidates[0];
							else {
								// use the candidate version of edge_to_part_greedy
								assignment = edge_to_part_greedy(source, target, candidates, graph.parts_counter, true);
							}
							move_edge(graph, eid, origin_part, assignment);
							move_counter++;
						}
					}
				}
			}

			// report the result
			cout << move_counter << " edges are moved" << endl;

			cout << "Time elapsed: " << ti.elapsed() << endl;

			report_performance(graph, nparts);
		}

		void itr_cost_greedy_refinement(basic_graph& graph, basic_graph::part_t nparts) {
			size_t refine_counter;
			do
			{
				refine_counter = cost_greedy_refinement(graph, nparts);
				bal_greedy_refinement(graph, nparts);
			} while (refine_counter > 0);
		}

	} // end of namespace partition_strategy

} // end of namespace graphp

#endif
