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
#include "sharding_constraint.hpp"

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

		/** \brief Returns the hashed value of a vertex. */
		inline static part_t hash_vertex (const vertex_id_type vid) { 
			return mix(vid);
		}

		inline static part_t hash_edge (const pair<vertex_id_type, vertex_id_type>& e, const uint32_t seed = 5) {
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

		void assign_edge(basic_graph& graph, edge_id_type eid, part_t assignment) {
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
		void assign_edge(basic_graph& graph, basic_graph::edge_type& e, part_t assignment) {
			// assign edge
			e.placement = assignment;
			graph.parts_counter[assignment]++;

			// assign vertex
			basic_graph::vertex_type& source = graph.getVert(e.source);
			basic_graph::vertex_type& target = graph.getVert(e.target);
			source.mirror_list[assignment] = true;
			target.mirror_list[assignment] = true;
		}

		void report_performance(const basic_graph& graph, part_t nparts) {
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
		void report_performance(const basic_graph& graph, part_t nparts, report_result& result) {
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
		void report_performance(const basic_graph& graph, part_t nparts, size_t vertex_cut_counter, report_result& result) {
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

		void random_partition(basic_graph& graph, part_t nparts) {
			typedef pair<vertex_id_type, vertex_id_type> edge_pair_type;
			//size_t edge_counter = 0;
			for(vector<basic_graph::edge_type>::iterator itr = graph.ebegin; itr != graph.eend; ++itr)  {
				basic_graph::edge_type& e = *itr;

				// random assign
				const edge_pair_type edge_pair(min(e.source, e.target), max(e.source, e.target));
				part_t assignment;
				assignment = hash_edge(edge_pair, hashing_seed) % (nparts);
				//edge_counter++;
				//assignment = edgernd(gen) % (nparts);
				assign_edge(graph, e, assignment);
			}
			//cout << edge_counter << endl;
		}

		void random_partition_constrained(basic_graph& graph, part_t nparts) {
			sharding_constraint* constraint;
			constraint = new sharding_constraint(nparts, "grid");
			typedef pair<vertex_id_type, vertex_id_type> edge_pair_type;
			for(vector<basic_graph::edge_type>::iterator itr = graph.ebegin; itr != graph.eend; ++itr)  {
				basic_graph::edge_type& e = *itr;

				// random assign
				const edge_pair_type edge_pair(min(e.source, e.target), max(e.source, e.target));
				part_t assignment;
				const vector<part_t>& candidates = constraint->get_joint_neighbors(hash_vertex(e.source) % nparts,
					hash_vertex(e.target) % nparts);
				assignment = candidates[hash_edge(edge_pair) % (candidates.size())];
				//assignment = edgernd(gen) % (nparts);
				assign_edge(graph, e, assignment);
			}
			delete constraint;
		}

		part_t edge_to_part_greedy(basic_graph& graph, 
			const basic_graph::vertex_id_type source,
			const basic_graph::vertex_id_type target,
			const vector<size_t>& part_num_edges,
			bool usehash = false
			) {
				const size_t nparts = part_num_edges.size();

				const basic_graph::vertex_type& source_v = graph.getVert(source);
				const basic_graph::vertex_type& target_v = graph.getVert(target);

				// compute the score of each part
				part_t best_part = -1;
				double maxscore = 0.0;
				double epsilon = 1.0;
				vector<double> part_score(nparts);
				size_t minedges = *min_element(part_num_edges.begin(), part_num_edges.end());
				size_t maxedges = *max_element(part_num_edges.begin(), part_num_edges.end());

				for(size_t i = 0; i < nparts; ++i) {
					size_t sd = source_v.mirror_list[i] + (usehash && (source % nparts == i));
					size_t td = target_v.mirror_list[i] + (usehash && (target % nparts == i));
					double bal = (maxedges - part_num_edges[i]) / (epsilon + maxedges - minedges);
					part_score[i] = bal + ((sd > 0) + (td > 0));
				}

				maxscore = *max_element(part_score.begin(), part_score.end());

				vector<part_t> top_parts;
				for(size_t i = 0; i < nparts; ++i) {
					if(fabs(part_score[i] - maxscore) < 1e-5) {
						top_parts.push_back(i);
					}
				}

				// hash the edge to one of the best parts
				typedef pair<vertex_id_type, vertex_id_type> edge_pair_type;
				const edge_pair_type edge_pair(min(source, target),
					max(source, target));
				best_part = top_parts[hash_edge(edge_pair) % top_parts.size()];
				//best_part = top_parts[edgernd(gen) % (top_parts.size())];

				return best_part;
		}

		part_t edge_to_part_greedy(basic_graph& graph, 
			const basic_graph::vertex_id_type source,
			const basic_graph::vertex_id_type target,
			const vector<part_t>& candidates,
			const vector<size_t>& part_num_edges,
			bool usehash = false
			) {
				const size_t nparts = part_num_edges.size();

				const basic_graph::vertex_type& source_v = graph.getVert(source);
				const basic_graph::vertex_type& target_v = graph.getVert(target);

				// compute the score of each part
				part_t best_part = -1;
				double maxscore = 0.0;
				double epsilon = 1.0;
				vector<double> part_score(candidates.size());
				size_t minedges = *min_element(part_num_edges.begin(), part_num_edges.end());
				size_t maxedges = *max_element(part_num_edges.begin(), part_num_edges.end());

				for(size_t j = 0; j < candidates.size(); ++j) {
					part_t i = candidates[j];
					size_t sd = source_v.mirror_list[i] + (usehash && (source % nparts == i));
					size_t td = target_v.mirror_list[i] + (usehash && (target % nparts == i));
					double bal = (maxedges - part_num_edges[i]) / (epsilon + maxedges - minedges);
					part_score[j] = bal + ((sd > 0) + (td > 0));
				}

				maxscore = *max_element(part_score.begin(), part_score.end());

				vector<part_t> top_parts;
				for (size_t j = 0; j < candidates.size(); ++j) {
					if(fabs(part_score[j] - maxscore) < 1e-5) {
						top_parts.push_back(candidates[j]);
					}
				}

				// hash the edge to one of the best parts
				typedef pair<vertex_id_type, vertex_id_type> edge_pair_type;
				const edge_pair_type edge_pair(min(source, target),
					max(source, target));
				best_part = top_parts[hash_edge(edge_pair) % top_parts.size()];
				//best_part = top_parts[edgernd(gen) % (top_parts.size())];

				return best_part;
		}

		void greedy_partition(basic_graph& graph, part_t nparts) {
			for(vector<basic_graph::edge_type>::iterator itr = graph.ebegin; itr != graph.eend; ++itr)  {
				basic_graph::edge_type& e = *itr;

				// greedy assign
				part_t assignment;
				assignment = edge_to_part_greedy(graph, e.source, e.target, graph.parts_counter, false);
				assign_edge(graph, e, assignment);
				//cout << e.eid << " " << e.source << " " << e.target << " " << e.weight << " " << e.placement << endl;
			}
		}

		void greedy_partition_constrainted(basic_graph& graph, part_t nparts) {
			sharding_constraint* constraint;
			boost::hash<vertex_id_type> hashvid;
			constraint = new sharding_constraint(nparts, "grid"); 
			for(vector<basic_graph::edge_type>::iterator itr = graph.ebegin; itr != graph.eend; ++itr)  {
				basic_graph::edge_type& e = *itr;

				// greedy assign
				part_t assignment;
				const vector<part_t>& candidates = 
					constraint->get_joint_neighbors(hashvid(e.source) % nparts, hashvid(e.target) % nparts);
				assignment = edge_to_part_greedy(graph, e.source, e.target, candidates, graph.parts_counter, false);
				assign_edge(graph, e, assignment);
				//cout << e.eid << " " << e.source << " " << e.target << " " << e.weight << " " << e.placement << endl;
			}
			delete constraint;
		}

		part_t edge_to_part_greedy2(basic_graph& graph, 
			const basic_graph::vertex_id_type source,
			const basic_graph::vertex_id_type target,
			const vector<size_t>& part_num_edges,
			bool usehash = false
			) {
				const size_t nparts = part_num_edges.size();

				const basic_graph::vertex_type& source_v = graph.getVert(source);
				const basic_graph::vertex_type& target_v = graph.getVert(target);

				// compute the score of each part
				part_t best_part = -1;
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
				double sum = source_v.degree + target_v.degree;
				double s = target_v.degree / sum + 1;
				double t = source_v.degree / sum + 1;

				for(size_t i = 0; i < nparts; ++i) {
					size_t sd = source_v.mirror_list[i] + (usehash && (source % nparts == i));
					size_t td = target_v.mirror_list[i] + (usehash && (target % nparts == i));
					double bal = (maxedges - part_num_edges[i]) / (epsilon + maxedges - minedges);
					part_score[i] = bal + ((sd > 0) * s + (td > 0) * t);
				}

				maxscore = *max_element(part_score.begin(), part_score.end());

				vector<part_t> top_parts;
				for(size_t i = 0; i < nparts; ++i) {
					if(fabs(part_score[i] - maxscore) < 1e-5) {
						top_parts.push_back(i);
					}
				}

				// hash the edge to one of the best parts
				typedef pair<vertex_id_type, vertex_id_type> edge_pair_type;
				const edge_pair_type edge_pair(min(source, target),
					max(source, target));
				best_part = top_parts[hash_edge(edge_pair) % top_parts.size()];
				//best_part = top_parts[edgernd(gen) % (top_parts.size())];

				return best_part;
		}

		part_t edge_to_part_greedy2(basic_graph& graph, 
			const basic_graph::vertex_id_type source,
			const basic_graph::vertex_id_type target,
			const vector<part_t>& candidates,
			const vector<size_t>& part_num_edges,
			bool usehash = false
			) {
				const size_t nparts = part_num_edges.size();

				const basic_graph::vertex_type& source_v = graph.getVert(source);
				const basic_graph::vertex_type& target_v = graph.getVert(target);

				// compute the score of each part
				part_t best_part = -1;
				double maxscore = 0.0;
				double epsilon = 1.0;
				vector<double> part_score(candidates.size());
				size_t minedges = *min_element(part_num_edges.begin(), part_num_edges.end());
				size_t maxedges = *max_element(part_num_edges.begin(), part_num_edges.end());

				// greedy for degree
				// nbr_list is not used in streaming partitioning
				//double sum = source_v.nbr_list.size() + target_v.nbr_list.size();
				//double s = target_v.nbr_list.size() / sum + 1;
				//double t = source_v.nbr_list.size() / sum + 1;
				// use degree in streaming partitioning
				double sum = source_v.degree + target_v.degree;
				double s = target_v.degree / sum + 1;
				double t = source_v.degree / sum + 1;

				for(size_t j = 0; j < candidates.size(); ++j) {
					part_t i = candidates[j];
					size_t sd = source_v.mirror_list[i] + (usehash && (source % nparts == i));
					size_t td = target_v.mirror_list[i] + (usehash && (target % nparts == i));
					double bal = (maxedges - part_num_edges[i]) / (epsilon + maxedges - minedges);
					part_score[j] = bal + ((sd > 0) * s + (td > 0) * t);
				}

				maxscore = *max_element(part_score.begin(), part_score.end());

				vector<part_t> top_parts;
				for(size_t j = 0; j < candidates.size(); ++j) {
					if(fabs(part_score[j] - maxscore) < 1e-5) {
						top_parts.push_back(candidates[j]);
					}
				}

				// hash the edge to one of the best parts
				typedef pair<vertex_id_type, vertex_id_type> edge_pair_type;
				const edge_pair_type edge_pair(min(source, target),
					max(source, target));
				best_part = top_parts[hash_edge(edge_pair) % top_parts.size()];
				//best_part = top_parts[edgernd(gen) % (top_parts.size())];

				return best_part;
		}

		void greedy_partition2(basic_graph& graph, part_t nparts) {
			for(vector<basic_graph::edge_type>::iterator itr = graph.ebegin; itr != graph.eend; ++itr)  {
				basic_graph::edge_type& e = *itr;

				// greedy assign
				part_t assignment;
				assignment = edge_to_part_greedy2(graph, e.source, e.target, graph.parts_counter, false);
				assign_edge(graph, e, assignment);
			}
		}

		void greedy_partition2_constrainted(basic_graph& graph, part_t nparts) {
			sharding_constraint* constraint;
			boost::hash<vertex_id_type> hashvid;
			constraint = new sharding_constraint(nparts, "grid"); 
			for(vector<basic_graph::edge_type>::iterator itr = graph.ebegin; itr != graph.eend; ++itr)  {
				basic_graph::edge_type& e = *itr;

				// greedy assign
				part_t assignment;
				const vector<part_t>& candidates = 
					constraint->get_joint_neighbors(hashvid(e.source) % nparts, hashvid(e.target) % nparts);
				assignment = edge_to_part_greedy2(graph, e.source, e.target, candidates, graph.parts_counter, false);
				assign_edge(graph, e, assignment);
			}
			delete constraint;
		}

		part_t edge_to_part_degree(basic_graph& graph,
			const basic_graph::vertex_id_type source,
			const basic_graph::vertex_id_type target,
			const vector<size_t>& part_num_edges
			) {
				const size_t nparts = part_num_edges.size();

				const basic_graph::vertex_type& source_v = graph.getVert(source);
				const basic_graph::vertex_type& target_v = graph.getVert(target);
				const size_t source_degree = source_v.degree;
				const size_t target_degree = target_v.degree;

				// compute the score of each part
				part_t best_part = -1;
				double maxscore = 0.0;
				double epsilon = 0.001;
				vector<double> part_score(nparts);
				size_t minedges = *min_element(part_num_edges.begin(), part_num_edges.end());
				size_t maxedges = *max_element(part_num_edges.begin(), part_num_edges.end());

				// not to be zero
				double e = 0.001;
				double sum = source_degree + target_degree + e;
				double s = target_degree / sum + 1;
				double t = source_degree / sum + 1;

				for(size_t i = 0; i < nparts; ++i) {
					size_t sd = source_v.mirror_list[i];
					size_t td = target_v.mirror_list[i];
					double bal = (maxedges - part_num_edges[i]) / (epsilon + maxedges - minedges);
					part_score[i] = bal + (sd > 0) + (sd > 0 && s > t) + (td > 0) + (td > 0 && s < t);
				}

				maxscore = *max_element(part_score.begin(), part_score.end());

				vector<part_t> top_parts;
				for(size_t i = 0; i < nparts; ++i) {
					if(fabs(part_score[i] - maxscore) < 1e-3) {
						top_parts.push_back(i);
					}
				}

				// hash the edge to one of the best parts
				typedef pair<vertex_id_type, vertex_id_type> edge_pair_type;
				const edge_pair_type edge_pair(min(source, target),
					max(source, target));
				best_part = top_parts[hash_edge(edge_pair) % top_parts.size()];

				return best_part;
		}
		part_t edge_to_part_degree(basic_graph& graph,
			const basic_graph::vertex_id_type source,
			const basic_graph::vertex_id_type target,
			const vector<part_t>& candidates,
			const vector<size_t>& part_num_edges
			) {
				const size_t nparts = part_num_edges.size();

				const basic_graph::vertex_type& source_v = graph.getVert(source);
				const basic_graph::vertex_type& target_v = graph.getVert(target);
				const size_t source_degree = source_v.degree;
				const size_t target_degree = target_v.degree;

				// compute the score of each part
				part_t best_part = -1;
				double maxscore = 0.0;
				double epsilon = 0.001;
				vector<double> part_score(candidates.size());
				size_t minedges = *min_element(part_num_edges.begin(), part_num_edges.end());
				size_t maxedges = *max_element(part_num_edges.begin(), part_num_edges.end());

				// not to be zero
				double e = 0.001;
				double sum = source_degree + target_degree + e;
				double s = target_degree / sum + 1;
				double t = source_degree / sum + 1;

				for(size_t j = 0; j < candidates.size(); ++j) {
					part_t i = candidates[j];
					size_t sd = source_v.mirror_list[i];
					size_t td = target_v.mirror_list[i];
					double bal = (maxedges - part_num_edges[i]) / (epsilon + maxedges - minedges);
					part_score[j] = bal + (sd > 0) + (sd > 0 && s > t) + (td > 0) + (td > 0 && s < t);
				}

				maxscore = *max_element(part_score.begin(), part_score.end());

				vector<part_t> top_parts;
				for(size_t j = 0; j < candidates.size(); ++j) {
					if(fabs(part_score[j] - maxscore) < 1e-5) {
						top_parts.push_back(candidates[j]);
					}
				}

				// hash the edge to one of the best parts
				typedef pair<vertex_id_type, vertex_id_type> edge_pair_type;
				const edge_pair_type edge_pair(min(source, target),
					max(source, target));
				best_part = top_parts[hash_edge(edge_pair) % top_parts.size()];

				return best_part;
		}

		void degree_partition(basic_graph& graph, part_t nparts) {
			for(vector<basic_graph::edge_type>::iterator itr = graph.ebegin; itr != graph.eend; ++itr)  {
				basic_graph::edge_type& e = *itr;

				// greedy assign
				part_t assignment;
				assignment = edge_to_part_degree(graph, e.source, e.target, graph.parts_counter);
				assign_edge(graph, e, assignment);
			}
		}

		void degree_partition_constrainted(basic_graph& graph, part_t nparts) {
			sharding_constraint* constraint;
			boost::hash<vertex_id_type> hashvid;
			constraint = new sharding_constraint(nparts, "grid"); 
			for(vector<basic_graph::edge_type>::iterator itr = graph.ebegin; itr != graph.eend; ++itr)  {
				basic_graph::edge_type& e = *itr;

				// greedy assign
				part_t assignment;
				const vector<part_t>& candidates = 
					constraint->get_joint_neighbors(hashvid(e.source) % nparts, hashvid(e.target) % nparts);
				assignment = edge_to_part_degree(graph, e.source, e.target, candidates, graph.parts_counter);
				assign_edge(graph, e, assignment);
			}
			delete constraint;
		}

		void hash_partition(basic_graph& graph, part_t nparts) {
			//const size_t threshold = nparts;
			for(vector<basic_graph::edge_type>::iterator itr = graph.ebegin; itr != graph.eend; ++itr)  {
				basic_graph::edge_type& e = *itr;

				// greedy assign
				part_t assignment;
				//const basic_graph::vertex_type& source_v = graph.getVert(e.source);
				//const basic_graph::vertex_type& target_v = graph.getVert(e.target);

				// randomly assign to the source vertex ...
				//if(source_v.degree < threshold && target_v.degree < threshold)
					assignment = hash_vertex(min(e.source, e.target)) % nparts;
				//else
				//	assignment = (source_v.degree < target_v.degree ? (hash_vertex(e.source) % nparts) : (hash_vertex(e.target) % nparts));
				assign_edge(graph, e, assignment);
			}
		}

		// deprecated for a while ...
		void run_partition(basic_graph& graph, vector<part_t>& nparts, vector<string>& strategies) {
		      vector<report_result> result_table(strategies.size() * nparts.size());
			  cout << "single thread" << endl;
		      for(size_t j = 0; j < strategies.size(); j++) {
		              // select the strategy
		              void (*partition_func)(basic_graph& graph, part_t nparts);
		              string strategy = strategies[j];
		              if(strategy == "random")
		                      partition_func = random_partition;
		              else if(strategy == "randomc")
		                      partition_func = random_partition_constrained;
		              else if(strategy == "greedy")
		                      partition_func = greedy_partition;
		              else if(strategy == "greedyc")
		                      partition_func = greedy_partition_constrainted;
		              else if(strategy == "degree")
		                      partition_func = greedy_partition2;
		              else if(strategy == "degreec")
		                      partition_func = greedy_partition2_constrainted;

		              for(size_t i = 0; i < nparts.size(); i++) {
		                      // initialize
		                      graph.initialize(nparts[i]);

		                      cout << endl << strategy << endl;

		                      boost::timer ti;
		                      double runtime;

		                      partition_func(graph, nparts[i]);

							  // debug
							  for(vector<basic_graph::edge_type>::iterator itr = graph.ebegin; itr != graph.eend; ++itr)  {
								  basic_graph::edge_type& e = *itr;
							  	cout << e.source << ":" << e.target << ":" << e.placement << endl;
							  }

		                      runtime = ti.elapsed();
		                      cout << "Time elapsed: " << runtime << endl;

		                      report_performance(graph, nparts[i], result_table[j * nparts.size() + i]);
		                      result_table[j * nparts.size() + i].runtime = runtime;
		              }
		      }

		      cout << endl;
		      // report the table
		      foreach(const report_result& result, result_table) {
		              cout << result.nparts << " "
		                      << result.vertex_cut_counter << " "
		                      << result.replica_factor << " "
		                      << result.imbalance << " "
		                      << result.runtime
		                      << endl;
		      }
		} // run partition in one thread

		
		void run_partition(basic_graph& graph, vector<part_t>& nparts, vector<size_t>& nthreads, vector<string>& prestrategies, vector<string>& strategies) {
			cout << endl;

			vector<report_result> result_table(prestrategies.size() * strategies.size() * nparts.size());

			//vector<basic_graph::vertex_id_type> vmap(graph.max_vid + 1);
			//for(boost::unordered_map<vertex_id_type, vertex_id_type>::iterator itr = graph.vid_to_lvid.begin(); itr != graph.vid_to_lvid.end(); ++itr) {
			//      vmap[itr->first] = itr->second;
			//}

			srand(time(0));

			//omp_set_num_threads(NUM_THREADS);

			typedef pair<vertex_id_type, vertex_id_type> edge_pair_type;

			for(size_t i = 0; i < nparts.size(); i++) {
				omp_set_num_threads(nthreads[i]);
				// construct the subgraphs for partitioning
				vector<size_t> thread_p(nthreads[i]);
				foreach(size_t& p, thread_p) {
					p = 0;
				}

				size_t edge_counter = 0;

				// threshold
				const size_t threshold = nparts[i];

				//// debug
				//for(boost::unordered_map<vertex_id_type, vertex_id_type>::iterator itr = graph.vid_to_lvid.begin(); itr != graph.vid_to_lvid.end(); itr++) {
				//	cout << itr->first << ":" << graph.getVert(itr->first).degree << " ";
				//}
				//cout << endl;
				//cout << endl;
				//for(boost::unordered_map<vertex_id_type, vertex_id_type>::iterator itr = graph.vid_to_lvid.begin(); itr != graph.vid_to_lvid.end(); itr++) {
				//	if(graph.getVert(itr->first).degree >= threshold)
				//		cout << itr->first << ":" << graph.getVert(itr->first).degree << " ";
				//}
				//cout << endl;
				//cout << endl;

				for(vector<basic_graph::edge_type>::iterator itr = graph.edges.begin(); itr != graph.edges.end(); itr++)  {
					part_t assignment;
					basic_graph::edge_type& e = *itr;
					const basic_graph::vertex_type& source_v = graph.getVert(e.source);
					const basic_graph::vertex_type& target_v = graph.getVert(e.target);

					// randomly assign to the source vertex ...
					if(source_v.degree < threshold && target_v.degree < threshold)
						assignment = hash_vertex(min(e.source, e.target)) % nthreads[i];
					else
						assignment = (source_v.degree < target_v.degree ? (hash_vertex(e.source) % nthreads[i]) : (hash_vertex(e.target) % nthreads[i])) + nthreads[i];
					//assignment = source_v.degree < target_v.degree ? hash_vertex(e.source) % nthreads[i] : hash_vertex(e.target) % nthreads[i];
					
					e.placement = assignment;
					if(assignment < nthreads[i])
						thread_p[assignment]++;
					else
						thread_p[assignment - nthreads[i]]++;
					edge_counter++;
				}

				// warning
				if(edge_counter != graph.nedges)
					cerr << "edge_counter != graph.nedges" << endl;

				vector<size_t> pp(nthreads[i]);
				pp[0] = 0;
				for(size_t idx_p = 1; idx_p < nthreads[i]; idx_p++) {
					pp[idx_p] = thread_p[idx_p - 1] + pp[idx_p - 1];
				}

				// warning
				edge_counter = 0;
				vector<size_t> lh(nthreads[i]);
				foreach(size_t& elem, lh) {
					elem = 0;
				}
				for(vector<basic_graph::edge_type>::iterator itr = graph.edges.begin(); itr != graph.edges.end(); ++itr)  {
				      basic_graph::edge_type& e = *itr;
				      size_t t = e.placement;
					  if(t < nthreads[i]) {
						  graph.edges_p[pp[t]] = e;
						  graph.edges_p[pp[t]].placement = nthreads[i]+1;
						  edge_counter++;
						  pp[t]++;
						  lh[t]++;
					  }
				}
				for(vector<basic_graph::edge_type>::iterator itr = graph.edges.begin(); itr != graph.edges.end(); ++itr)  {
					basic_graph::edge_type& e = *itr;
					size_t t = e.placement;
					if(t >= nthreads[i]) {
						t = t - nthreads[i];
						graph.edges_p[pp[t]] = e;
						graph.edges_p[pp[t]].placement = nthreads[i]+1;
						edge_counter++;
						pp[t]++;
					}
				}
				if(edge_counter != graph.nedges)
					cerr << "edge_counter != graph.nedges" << endl;
				vector<basic_graph::edge_type>().swap(graph.edges);
				graph.edges.clear();

				// debug
				//for(vector<basic_graph::edge_type>::iterator itr = graph.edges.begin(); itr != graph.edges.end(); ++itr)  {
				//	basic_graph::edge_type& e = *itr;
				//	cout << e.source << ":" << e.target << ":" << e.placement << endl;
				//}
				//cout << endl;
				//for(vector<basic_graph::edge_type>::iterator itr = graph.edges_p.begin(); itr != graph.edges_p.end(); ++itr)  {
				//	basic_graph::edge_type& e = *itr;
				//	cout << e.source << ":" << e.target << endl;
				//}
				//system("Pause");
				//exit(0);


				graph.ebegin = graph.edges_p.begin();
				graph.eend = graph.edges_p.end();
				pp[0] = 0;
				for(size_t idx_p = 1; idx_p < nthreads[i]; idx_p++) {
					pp[idx_p] = thread_p[idx_p - 1] + pp[idx_p - 1];
				}
				for(size_t idx_p = 0; idx_p < nthreads[i]; idx_p++) {
					lh[idx_p] = pp[idx_p] + lh[idx_p];
				}
				for(size_t idx_p = 1; idx_p < nthreads[i]; idx_p++) {
					thread_p[idx_p] = thread_p[idx_p] + thread_p[idx_p - 1];
				}

				// debug
				//foreach(size_t observation, pp) {
				//	cout << observation << "\t";
				//}
				//cout << endl;
				//foreach(size_t observation, lh) {
				//	cout << observation << "\t";
				//}
				//cout << endl;
				//foreach(size_t observation, thread_p) {
				//	cout << observation << "\t";
				//}
				//cout << endl;

				// random inner shuffle
				//for(size_t idx_p = 0; idx_p < nthreads[i]; idx_p++) {
				//      random_shuffle(graph.ebegin + pp[idx_p], graph.ebegin + thread_p[idx_p]);
				//}
				for(size_t prei = 0; prei < prestrategies.size(); prei++) {
					void (*prepartition_func)(basic_graph& graph, part_t nparts);
					string prestrategy = prestrategies[prei];
					if(prestrategy == "random")
						prepartition_func = random_partition;
					else if(prestrategy == "randomc")
						prepartition_func = random_partition_constrained;
					else if(prestrategy == "greedy")
						prepartition_func = greedy_partition;
					else if(prestrategy == "greedyc")
						prepartition_func = greedy_partition_constrainted;
					else if(prestrategy == "degree")
						prepartition_func = greedy_partition2;
					else if(prestrategy == "degreec")
						prepartition_func = greedy_partition2_constrainted;
					else if(prestrategy == "degree2")
						prepartition_func = degree_partition;
					else if(prestrategy == "degree2c")
						prepartition_func = degree_partition_constrainted;
					else if(prestrategy == "hash")
						prepartition_func = hash_partition;
					else 
						prepartition_func = random_partition;

					for(size_t j = 0; j < strategies.size(); j++) {
						// select the strategy
						void (*partition_func)(basic_graph& graph, part_t nparts);
						string strategy = strategies[j];
						if(strategy == "random")
							partition_func = random_partition;
						else if(strategy == "randomc")
							partition_func = random_partition_constrained;
						else if(strategy == "greedy")
							partition_func = greedy_partition;
						else if(strategy == "greedyc")
							partition_func = greedy_partition_constrainted;
						else if(strategy == "degree")
							partition_func = greedy_partition2;
						else if(strategy == "degreec")
							partition_func = greedy_partition2_constrainted;
						else if(strategy == "degree2")
							partition_func = degree_partition;
						else if(strategy == "degree2c")
							partition_func = degree_partition_constrainted;
						else 
							partition_func = random_partition;

						vector<basic_graph> subgraphs(nthreads[i]);

						cout << strategy << endl;
						size_t nt = NUM_THREADS;
						//cout << "using " << nt << " threads..." << endl;

						//// debug
						//for(vector<basic_graph::edge_type>::iterator itr = graph.ebegin; itr != graph.eend; ++itr) {
						//	basic_graph::edge_type& e = *itr;
						//	cout << e.source << ":" << e.target << ":" << e.placement << " ";
						//}
						//cout << endl;
						//cout << endl;

						graph.initialize(nparts[i]);
						vector<omp_lock_t> vlocks(graph.nverts);
						vector<omp_lock_t> plocks(nparts[i]);
						#pragma omp parallel for
						for(size_t tid = 0; tid < graph.nverts; tid++) {
							omp_init_lock(&(vlocks[tid]));
						}
						#pragma omp parallel for
						for(size_t tid = 0; tid < nparts[i]; tid++) {
							omp_init_lock(&(plocks[tid]));
						}
						cout << "Locks initialized ..." << endl;

						// initialize each subgraph
						#pragma omp parallel for
						for(size_t tid = 0; tid < nthreads[i]; tid++) {
							//cout << "Thread: " << omp_get_thread_num() << ", Tid: " << tid << endl;
							size_t begin = pp[tid];
							size_t end = lh[tid];
							subgraphs[tid].ebegin = graph.ebegin + begin;
							subgraphs[tid].eend = graph.ebegin + end;

							// do not let finalize to save edges
							subgraphs[tid].nparts = nparts[i];
							subgraphs[tid].max_vid = graph.max_vid;
							subgraphs[tid].finalize(false);
							subgraphs[tid].initialize(nparts[i]);
							for(boost::unordered_map<vertex_id_type, vertex_id_type>::iterator itr = subgraphs[tid].vid_to_lvid.begin(); itr != subgraphs[tid].vid_to_lvid.end(); ++itr) {
								subgraphs[tid].getVert(itr->first).degree = graph.getVert(itr->first).degree;
							}
						}
						//cout << "initialized ..." << endl;
						
						#pragma omp parallel for
						for(size_t tid = 0; tid < nthreads[i]; tid++) {
							//prepartition_func(subgraphs[tid], nparts[i]);

							//// debug
							//cout << begin << "," << end << endl;
							//for(vector<basic_graph::edge_type>::iterator itr = subgraphs[tid].ebegin; itr != subgraphs[tid].eend; ++itr) {
							//	basic_graph::edge_type& e = *itr;
							//	cout << e.source << ":" << e.target << ":" << e.placement << " ";
							//}
							//cout << endl;
							for(vector<basic_graph::edge_type>::iterator itr = subgraphs[tid].ebegin; itr != subgraphs[tid].eend; ++itr)  {
								//cout << tid << endl;
								basic_graph::edge_type& e = *itr;
								// global
								basic_graph::vertex_type& gsource = graph.getVert(e.source);
								basic_graph::vertex_type& gtarget = graph.getVert(e.target);
								// local
								basic_graph::vertex_type& lsource = subgraphs[tid].getVert(e.source);
								basic_graph::vertex_type& ltarget = subgraphs[tid].getVert(e.target);
								// id
								size_t source_id = graph.vid_to_lvid[e.source];
								//size_t sid = e.source;
								//cout << "sid: " << sid << endl;
								size_t target_id = graph.vid_to_lvid[e.target];
								//size_t tid = e.target;
								//cout << "tid: " << tid << endl;
								//cout << "length: " << vlocks.size() << endl;

								// greedy assign
								part_t assignment;
								omp_set_lock(&(vlocks[source_id]));
								lsource.mirror_list = gsource.mirror_list;
								omp_unset_lock(&(vlocks[source_id]));
								omp_set_lock(&(vlocks[target_id]));
								ltarget.mirror_list = gtarget.mirror_list;
								omp_unset_lock(&(vlocks[target_id]));

								//assignment = edge_to_part_degree(subgraphs[tid], e.source, e.target, subgraphs[tid].parts_counter);
								assignment = edge_to_part_greedy(subgraphs[tid], e.source, e.target, subgraphs[tid].parts_counter);
								e.placement = assignment;
								

								omp_set_lock(&(plocks[assignment]));
								graph.parts_counter[assignment]++;
								omp_unset_lock(&(plocks[assignment]));
								subgraphs[tid].parts_counter[assignment]++;

								if(!(lsource.mirror_list[assignment])) {
									omp_set_lock(&(vlocks[source_id]));
									gsource.mirror_list[assignment] = true;
									omp_unset_lock(&(vlocks[source_id]));
									lsource.mirror_list[assignment] = true;
								}
								if(!(ltarget.mirror_list[assignment])) {
									omp_set_lock(&(vlocks[target_id]));
									gtarget.mirror_list[assignment] = true;
									omp_unset_lock(&(vlocks[target_id]));
									ltarget.mirror_list[assignment] = true;
								}
							}

							// clear memory
							vector<basic_graph::vertex_type>().swap(subgraphs[tid].verts);
							boost::unordered_map<basic_graph::vertex_id_type, basic_graph::vertex_id_type>().swap(subgraphs[tid].vid_to_lvid);
						}

						//// debug
						//for(vector<basic_graph::edge_type>::iterator itr = graph.ebegin; itr != graph.eend; ++itr) {
						//	basic_graph::edge_type& e = *itr;
						//	cout << e.source << ":" << e.target << ":" << e.placement << " ";
						//}
						//cout << endl;
						//cout << endl;
						cout << "stage 1 finished ..." << endl;

						// do assignment in single thread
						// note: use edges_p
						//for(vector<basic_graph::edge_type>::iterator itr = graph.edges_p.begin(); itr != graph.edges_p.end(); ++itr)  {
						//	basic_graph::edge_type& e = *itr;

						//	// assign vertex
						//	basic_graph::vertex_type& source = graph.getVert(e.source);
						//	basic_graph::vertex_type& target = graph.getVert(e.target);
						//	if(source.degree < threshold && target.degree < threshold) {
						//		source.mirror_list[e.placement] = true;
						//		target.mirror_list[e.placement] = true;
						//		// assign edge
						//		graph.parts_counter[e.placement]++;
						//	}
						//}
						//cout << "synchronization finished ..." << endl;
						//exit(0);

						#pragma omp parallel for
						for(size_t tid = 0; tid < nthreads[i]; tid++) {
							size_t begin = lh[tid];
							size_t end = thread_p[tid];
							subgraphs[tid].ebegin = graph.ebegin + begin;
							subgraphs[tid].eend = graph.ebegin + end;

							// do not let finalize to save edges
							subgraphs[tid].nparts = nparts[i];
							subgraphs[tid].max_vid = graph.max_vid;
							subgraphs[tid].finalize(false);
							subgraphs[tid].initialize(nparts[i]);
							for(boost::unordered_map<vertex_id_type, vertex_id_type>::iterator itr = subgraphs[tid].vid_to_lvid.begin(); itr != subgraphs[tid].vid_to_lvid.end(); ++itr) {
								subgraphs[tid].getVert(itr->first).degree = graph.getVert(itr->first).degree;
								subgraphs[tid].getVert(itr->first).mirror_list = graph.getVert(itr->first).mirror_list;
							}
							subgraphs[tid].parts_counter = graph.parts_counter;
							partition_func(subgraphs[tid], nparts[i]);
							//// debug
							//cout << begin << "," << end << endl;
							//for(vector<basic_graph::edge_type>::iterator itr = subgraphs[tid].ebegin; itr != subgraphs[tid].eend; ++itr) {
							//	basic_graph::edge_type& e = *itr;
							//	cout << e.source << ":" << e.target << ":" << e.placement << " ";
							//}
							//cout << endl;

							// clear memory
							vector<basic_graph::vertex_type>().swap(subgraphs[tid].verts);
							boost::unordered_map<basic_graph::vertex_id_type, basic_graph::vertex_id_type>().swap(subgraphs[tid].vid_to_lvid);
						}

						//// debug
						//for(vector<basic_graph::edge_type>::iterator itr = graph.ebegin; itr != graph.eend; ++itr) {
						//	basic_graph::edge_type& e = *itr;
						//	cout << e.source << ":" << e.target << ":" << e.placement << " ";
						//}
						//cout << endl;
						//cout << endl;

						//boost::timer ti;
						double runtime = 0;
						//runtime = omp_get_wtime();

						// do assignment in single thread
						// note: use edges_p
						for(vector<basic_graph::edge_type>::iterator itr = graph.edges_p.begin(); itr != graph.edges_p.end(); ++itr)  {
							basic_graph::edge_type& e = *itr;

							// assign vertex
							basic_graph::vertex_type& source = graph.getVert(e.source);
							basic_graph::vertex_type& target = graph.getVert(e.target);
							if(source.degree < threshold && target.degree < threshold)
								continue;
							source.mirror_list[e.placement] = true;
							target.mirror_list[e.placement] = true;

							// assign edge
							graph.parts_counter[e.placement]++;
						}

						report_performance(graph, nparts[i], result_table[i*prestrategies.size()*strategies.size() + prei*strategies.size() + j]);
						result_table[i*prestrategies.size()*strategies.size() + prei*strategies.size() + j].runtime = 0;

#pragma omp parallel for
						for(size_t tid = 0; tid < graph.nverts; tid++) {
							omp_destroy_lock(&(vlocks[tid]));
						}
#pragma omp parallel for
						for(size_t tid = 0; tid < nparts[i]; tid++) {
							omp_destroy_lock(&(plocks[tid]));
						}

					}
				}
			}

			cout << endl;
			// report the table
			foreach(const report_result& result, result_table) {
				cout << result.nparts << " "
					<< result.vertex_cut_counter << " "
					<< result.replica_factor << " "
					<< result.imbalance << " "
					<< result.runtime
					<< endl;
			}
		}// run partition in multi-threads


	} // end of namespace partition_strategy

} // end of namespace graphp

#endif