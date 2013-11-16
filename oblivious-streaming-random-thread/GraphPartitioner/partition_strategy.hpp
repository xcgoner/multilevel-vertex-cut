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
			for(vector<basic_graph::edge_type>::iterator itr = graph.ebegin; itr != graph.eend; ++itr)  {
				basic_graph::edge_type& e = *itr;
				// random assign
				const edge_pair_type edge_pair(min(e.source, e.target), max(e.source, e.target));
				part_t assignment;
				assignment = hash_edge(edge_pair, hashing_seed) % (nparts);
				//assignment = edgernd(gen) % (nparts);
				assign_edge(graph, e, assignment);
			}
		}

		void random_partition_constrained(basic_graph& graph, part_t nparts/*, const std::string& method*/) {
			sharding_constraint* constraint;
			constraint = new sharding_constraint(nparts, /*method*/ "grid");
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

				//part_t assignment1, assignment2;
				//// assign first
				//assignment1 = edge_to_part_greedy2(graph, e.source, e.target, graph.parts_counter, false);
				//// assign second
				//const vector<part_t>& candidates = 
				//	constraint->get_joint_neighbors(hashvid(e.source) % nparts, hashvid(e.target) % nparts);
				//assignment2 = edge_to_part_greedy2(graph, e.source, e.target, candidates, graph.parts_counter, false);
				//const basic_graph::vertex_type& source_v = graph.getVert(e.source);
				//const basic_graph::vertex_type& target_v = graph.getVert(e.target);
				//size_t score1 = (size_t)(source_v.mirror_list[assignment1]) + (size_t)(target_v.mirror_list[assignment1]);
				//size_t score2 = (size_t)(source_v.mirror_list[assignment2]) + (size_t)(target_v.mirror_list[assignment2]);
				//if(score1 > score2)
				//	assign_edge(graph, e, assignment1);
				//else
				//	assign_edge(graph, e, assignment2);

				part_t assignment;
				const vector<part_t>& candidates = 
					constraint->get_joint_neighbors(hashvid(e.source) % nparts, hashvid(e.target) % nparts);
				assignment = edge_to_part_greedy2(graph, e.source, e.target, candidates, graph.parts_counter, false);
				assign_edge(graph, e, assignment);
			}
			delete constraint;
		}

		void greedy_partition2_constrainted1(basic_graph& graph, part_t nparts) {
			sharding_constraint* constraint;
			boost::hash<vertex_id_type> hashvid;
			constraint = new sharding_constraint(nparts, "grid"); 
			for(vector<basic_graph::edge_type>::iterator itr = graph.ebegin; itr != graph.eend; ++itr)  {
				basic_graph::edge_type& e = *itr;
				// greedy assign

				part_t assignment1, assignment2;
				// assign first
				assignment1 = edge_to_part_greedy2(graph, e.source, e.target, graph.parts_counter, false);
				// assign second
				const vector<part_t>& candidates = 
					constraint->get_joint_neighbors(hashvid(e.source) % nparts, hashvid(e.target) % nparts);
				assignment2 = edge_to_part_greedy2(graph, e.source, e.target, candidates, graph.parts_counter, false);
				const basic_graph::vertex_type& source_v = graph.getVert(e.source);
				const basic_graph::vertex_type& target_v = graph.getVert(e.target);
				size_t score1 = (size_t)(source_v.mirror_list[assignment1]) + (size_t)(target_v.mirror_list[assignment1]);
				size_t score2 = (size_t)(source_v.mirror_list[assignment2]) + (size_t)(target_v.mirror_list[assignment2]);
				if(score1 > score2)
					assign_edge(graph, e, assignment1);
				else
					assign_edge(graph, e, assignment2);
			}
			delete constraint;
		}

		void run_partition(basic_graph& graph, vector<part_t>& nparts, vector<string>& strategies) {
			vector<report_result> result_table(strategies.size() * nparts.size());
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
				else if(strategy == "degreec1")
					partition_func = greedy_partition2_constrainted1;

				for(size_t i = 0; i < nparts.size(); i++) {
					// initialize
					graph.initialize(nparts[i]);

					cout << endl << strategy << endl;

					boost::timer ti;
					double runtime;

					partition_func(graph, nparts[i]);

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

		void run_partition(basic_graph& graph, vector<part_t>& nparts, vector<size_t>& nthreads, vector<string>& strategies) {
			cout << endl;

			vector<report_result> result_table(strategies.size() * nparts.size());

			//vector<basic_graph::vertex_id_type> vmap(graph.max_vid + 1);
			//for(boost::unordered_map<vertex_id_type, vertex_id_type>::iterator itr = graph.vid_to_lvid.begin(); itr != graph.vid_to_lvid.end(); ++itr) {
			//	vmap[itr->first] = itr->second;
			//}

			srand(time(0));

			omp_set_num_threads(NUM_THREADS);
			typedef pair<vertex_id_type, vertex_id_type> edge_pair_type;
			cout << "nedges: " << graph.nedges << endl;
			const size_t file_block_size = 36 * graph.nedges / 5100000;
			//const size_t file_block_size = 2;
			for(size_t i = 0; i < nparts.size(); i++) {
				// construct the subgraphs for partitioning
				vector<size_t> thread_p(nthreads[i]);
				foreach(size_t& p, thread_p) {
					p = 0;
				}

				size_t edge_counter = 0;
				// random chunk
				vector<size_t> eindex;
				for(vector<basic_graph::edge_type>::iterator itr = graph.edges.begin(); itr != graph.edges.end();)  {
					part_t assignment;
					//assignment = edge_counter % (nthreads[i]);
					assignment = edgernd(gen) % (nthreads[i]);
					//assignment = rand() % (nthreads[i]);
					eindex.push_back(edge_counter);
					for(size_t idx = 0; idx < file_block_size && itr != graph.edges.end(); idx++, itr++) {
						basic_graph::edge_type& e = *itr;
						// random assign
						//const edge_pair_type edge_pair(min(e.source, e.target), max(e.source, e.target));
						//assignment = edgernd(gen) % (nthreads[i]);
						e.placement = assignment;
						thread_p[assignment]++;
						edge_counter++;
					}
				}
				// push the end
				eindex.push_back(edge_counter);

				// warning
				if(edge_counter != graph.nedges)
					cerr << "edge_counter != graph.nedges" << endl;

				// shuffle the chunks
				vector<size_t> idx_eindex;
				for(size_t idx = 0; idx < eindex.size(); idx++) {
					idx_eindex.push_back(idx);
				}
				random_shuffle(idx_eindex.begin(), idx_eindex.end());

				vector<size_t> pp(nthreads[i]);
				pp[0] = 0;
				for(size_t idx_p = 1; idx_p < nthreads[i]; idx_p++) {
					pp[idx_p] = thread_p[idx_p - 1] + pp[idx_p - 1];
				}

				// place the chunks
				// alternative0
				edge_counter = 0;
				foreach(size_t ei, idx_eindex) {
					if(ei == eindex.size() - 1)
						continue;
					for(size_t idx = eindex[ei]; idx < eindex[ei+1]; idx++) {
						basic_graph::edge_type& e = graph.edges[idx];
						size_t t = e.placement;
						graph.edges_p[pp[t]] = e;
						pp[t]++;
						edge_counter++;
					}
				}
				// warning
				if(edge_counter != graph.nedges)
					cerr << "edge_counter != graph.nedges" << endl;
				// alternative1
				//for(vector<basic_graph::edge_type>::iterator itr = graph.edges.begin(); itr != graph.edges.end(); ++itr)  {
				//	basic_graph::edge_type& e = *itr;
				//	size_t t = e.placement;
				//	graph.edges_p[pp[t]] = e;
				//	pp[t]++;
				//}


				graph.ebegin = graph.edges_p.begin();
				graph.eend = graph.edges_p.end();
				pp[0] = 0;
				for(size_t idx_p = 1; idx_p < nthreads[i]; idx_p++) {
					pp[idx_p] = thread_p[idx_p - 1] + pp[idx_p - 1];
				}
				for(size_t idx_p = 1; idx_p < nthreads[i]; idx_p++) {
					thread_p[idx_p] = thread_p[idx_p] + thread_p[idx_p - 1];
				}

				// random inner shuffle
				//for(size_t idx_p = 0; idx_p < nthreads[i]; idx_p++) {
				//	random_shuffle(graph.ebegin + pp[idx_p], graph.ebegin + thread_p[idx_p]);
				//}

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

					vector<basic_graph> subgraphs(nthreads[i]);

					cout << strategy << endl;
					size_t nt = NUM_THREADS;
					cout << "using " << nt << " threads..." << endl;

					// initialize each subgraph
					for(size_t ptid = 0; ptid <= nthreads[i] / nt; ptid++) {
						size_t tbegin = nt * ptid;
						size_t tend = nt * (ptid + 1);
						if(tbegin >= nthreads[i])
							break;
						if(tend >= nthreads[i])
							tend = nthreads[i];
						cout << "threads " << tbegin << " to " << tend - 1 << endl;
						size_t tl = tend - tbegin;
						#pragma omp parallel for
						for(size_t tt = 0; tt < tl; tt++) {
							size_t tid = tbegin + tt;
							size_t begin = pp[tid];
							size_t end = thread_p[tid];
							if(tid == nthreads[i] - 1)
								end = graph.nedges;
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
							partition_func(subgraphs[tid], nparts[i]);

							// clear memory
							vector<basic_graph::vertex_type>().swap(subgraphs[tid].verts);
							boost::unordered_map<basic_graph::vertex_id_type, basic_graph::vertex_id_type>().swap(subgraphs[tid].vid_to_lvid);
						}
					}

					//boost::timer ti;
					double runtime = 0;
					//runtime = omp_get_wtime();

					//#pragma omp parallel for
					//for(size_t tid = 0; tid < nthreads[i]; tid++) {
					//	partition_func(subgraphs[tid], nparts[i]);
					//}

					//runtime = ti.elapsed();
					//runtime = omp_get_wtime() - runtime;
					//cout << "Time elapsed: " << runtime << endl;

					// assign back to the origin graph
					graph.initialize(nparts[i]);

					// do assignment in single thread
					// note: use edges_p
					for(vector<basic_graph::edge_type>::iterator itr = graph.edges_p.begin(); itr != graph.edges_p.end(); ++itr)  {
						basic_graph::edge_type& e = *itr;
						//assign_edge(graph, e, e.placement);

						// assign edge
						graph.parts_counter[e.placement]++;

						// assign vertex
						//basic_graph::vertex_type& source = graph.verts[vmap[e.source]];
						//basic_graph::vertex_type& target = graph.verts[vmap[e.target]];
						basic_graph::vertex_type& source = graph.getVert(e.source);
						basic_graph::vertex_type& target = graph.getVert(e.target);
						source.mirror_list[e.placement] = true;
						target.mirror_list[e.placement] = true;
					}

					// do assignment in multi-threads
					//vector<size_t> vertex_cut_counters(nthreads[i]);
					//#pragma omp parallel for
					//for(size_t tid = 0; tid < nthreads[i]; tid++) {
					//	vertex_cut_counters[tid] = 0;
					//	foreach(basic_graph::vertex_type& v, subgraphs[tid].verts) {
					//		if(v.mirror_list.count() > 0)
					//			vertex_cut_counters[tid] += (v.mirror_list.count() - 1);
					//	}
					//}
					//size_t vertex_cut_counter = 0;
					//for(size_t tid = 0; tid < nthreads[i]; tid++)
					//	vertex_cut_counter += vertex_cut_counters[tid];
					//omp_set_num_threads(nparts[i]);
					//#pragma omp parallel for
					//for(size_t pid = 0; pid < nparts[i]; pid++) {
					//	for(size_t tid = 0; tid < nthreads[i]; tid++) {
					//		graph.parts_counter[pid] += subgraphs[tid].parts_counter[pid];
					//	}
					//}

					report_performance(graph, nparts[i], result_table[j * nparts.size() + i]);
					//report_performance(graph, nparts[i], vertex_cut_counter, result_table[j * nparts.size() + i]);
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
		}// run partition in multi-threads

	} // end of namespace partition_strategy

} // end of namespace graphp

#endif
