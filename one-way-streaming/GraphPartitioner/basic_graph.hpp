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
 */
#ifndef GRAPH_PARTITIONER_BASIC_GRAPH_HPP
#define GRAPH_PARTITIONER_BASIC_GRAPH_HPP

#include <string>
#include <sstream>
#include <iostream>
#include <vector>
#include <map>
#include <hash_map>
#include <set>
#include <deque>
#include <fstream>
#include <sstream>

#include "builtin_parsers.hpp"
#include "graph_basic_types.hpp"
#include "util.hpp"
#include "fs_util.hpp"
#include "memory_info.hpp"
#include <boost/program_options.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/filesystem.hpp>
#include <boost/timer.hpp>
#include <boost/dynamic_bitset.hpp>
#include <boost/unordered_map.hpp>

#include <omp.h>

namespace graphp_options = boost::program_options;

using namespace std;
#ifdef __GNUC__ 
using namespace __gnu_cxx; 
#endif

namespace graphp {

	
	class basic_graph {

	public:

		typedef graphp::basic_graph graph_type;

		typedef boost::function<bool(basic_graph&, const string&, const string&)> line_parser_type;

		typedef graphp::vertex_id_type vertex_id_type;
		typedef graphp::edge_id_type edge_id_type;

		typedef unsigned short part_t;

		// list of vertices
		typedef vector<vertex_id_type> vertex_list_type;

		// list of edges
		typedef vector<edge_id_type> edge_list_type;

		size_t nverts, nedges, nparts;

		vertex_id_type max_vid;

		vector<size_t> parts_counter;

		struct vertex_type {
			// neighbour list
			// not used in streaming partitioning
			//vertex_list_type nbr_list;

			// key: neighbour target vid, value: edge eid
			// not used in streaming partitioning
			//edge_list_type edge_list;

			// use degree in streaming partitioning
			size_t degree;

			size_t edge_begin;
			size_t edge_end;

			boost::dynamic_bitset<> mirror_list;

			vertex_type() :
			edge_begin(-1), edge_end(-1), degree(0) { }

			bool isFree() const {
				return (mirror_list.size() == 0);
			}

			friend class basic_graph;
		};

		struct edge_type {
			vertex_id_type source;
			vertex_id_type target;
			part_t placement;
			edge_type() :
				placement(-1) { }
			edge_type(const vertex_id_type& source, const vertex_id_type& target) :
				source(source), target(target), placement(-1) { }

			friend class basic_graph;
		};

		typedef map<vertex_id_type, vertex_type> verts_map_type;
		vector<vertex_type> verts;
		boost::unordered_map<vertex_id_type, vertex_id_type> vid_to_lvid;

		boost::dynamic_bitset<> vmap;

		vector<edge_type> edges;
		vector<edge_type>::iterator ebegin, eend;
		vector<edge_type> edges_p;

		deque<edge_type> edges_storage;

		//deque<edge_type>& edges = edges_storage;

		// constructor
		basic_graph() : nverts(0), nedges(0), max_vid(0), nparts(1) {
		}
		// do not assign 0 to nparts, something bad will occur ...
		basic_graph(size_t nparts) : nverts(0), nedges(0), max_vid(0), nparts(nparts) {
			parts_counter.resize(nparts);
			foreach(size_t& num_edges, parts_counter) {
				num_edges = 0;
			}
		}

		bool add_vertex(const vertex_id_type& vid, const size_t& weight = 1) {
			if(verts[vid].isFree()) {
				verts[vid].degree = 0;
				verts[vid].mirror_list.resize(nparts);
				nverts++;
				return true;
			}
			return false;
		}

		void add_edge_to_storage(const vertex_id_type& source, const vertex_id_type& target, const size_t& weight = 1, const part_t& placement = -1) {
			edge_type e(source, target);
			if(source > max_vid)
				max_vid = source;
			if(target > max_vid)
				max_vid = target;
			if(placement != -1)
				e.placement = placement;
			edges_storage.push_back(e);
			nedges++;
		}

		void clear_partition_counter() {
			foreach(size_t& num_edges, parts_counter) {
				num_edges = 0;
			}
		}
		void clear_partition() {
			// actually this step is useless ...
			//for(vector<edge_type>::iterator itr = ebegin; itr != eend; ++itr) {
			//	itr->placement = -1;
			//}
		}
		void clear_mirrors() {
			foreach(vertex_type& v, verts) {
				v.mirror_list.resize(nparts);
				v.mirror_list.reset();
			}
		}

		void initialize(part_t np) {
			nparts = np;
			parts_counter.resize(np);
			clear_partition_counter();
			clear_partition();
			clear_mirrors();
			foreach(vertex_type& v, verts) {
				v.degree = 0;
			}
		}

		vertex_type& getVert(vertex_id_type vid) {
			return verts[vid_to_lvid[vid]];
		}
		edge_type& getEdge(edge_id_type eid) {
			//return edges[eid];
			return *(ebegin + eid);
		}

		void finalize(bool saveEdges = true) {
			cout << "finalizing..." << endl;

			//edges.reserve(edges_storage.size() + 1);
			//verts.resize(max_vid + 1);

			boost::timer ti;

			if(saveEdges) {
				edges.resize(edges_storage.size());
				size_t edges_idx = 0;
				for(deque<edge_type>::iterator itr = edges_storage.begin(); itr != edges_storage.end(); ++itr) {
					edges[edges_idx++] = (*itr);
				}
				// release the memory
				edges_storage.clear();
				// access the edges in random order
				//srand(time(0));
				//random_shuffle(edges.begin(), edges.end());
				//cout << "edges are random shuffled ..." << endl;

				ebegin = edges.begin();
				eend = edges.end();

				edges_p.resize(edges.size());
			}

			// access the edges in random order
			//random_shuffle(edges.begin(), edges.end());

			vmap.resize(max_vid + 1);
			for(vector<edge_type>::iterator itr = ebegin; itr != eend; ++itr) {
				vmap[itr->source] = true;
				vmap[itr->target] = true;
			}
			nverts = vmap.count();
			verts.resize(nverts);
			for(size_t vid = vmap.find_first(), idx = 0; vid != vmap.npos; vid = vmap.find_next(vid), idx++) {
				vid_to_lvid.insert(pair<vertex_id_type, vertex_id_type>(vid, idx));
				verts[idx].degree = 0;
				verts[idx].mirror_list.resize(nparts);
			}

			if(saveEdges) {
				size_t edgecount = 0;
				size_t current_source_vid = -1;
				for(size_t idx = 0; idx < edges.size(); idx++) {
					const edge_type& e = edges[idx];
					if(e.source != current_source_vid) {
						if(current_source_vid != -1) {
							getVert(edges[idx - 1].source).edge_end = idx;
						}
						if(getVert(e.source).edge_begin != -1) {
							cerr << "The graph file is not in a edge-batch order !!!" << endl;
							exit(0);
						}
						getVert(e.source).edge_begin = idx;
						current_source_vid = e.source;
					}
					// add edge
					//edges[edgecount] = itr;

					// add vertex
					// treat every single edge as an undirected one
					//add_vertex(itr->source);
					// not used in streaming partitioning
					//verts[itr->source].nbr_list.push_back(itr->target);
					//verts[itr->source].edge_list.push_back(edgecount);
					//verts[itr->source].degree++;
					// use degree in streaming partitioning
					//deprecate in online setting: degree is not known ahead
					//getVert(e.source).degree++;
					//add_vertex(itr->target);
					// not used in streaming partitioning
					//verts[itr->target].nbr_list.push_back(itr->source);
					//verts[itr->target].edge_list.push_back(edgecount);
					//verts[itr->target].degree++;
					// use degree in streaming partitioning
					//deprecate in online setting: degree is not known ahead
					//getVert(e.target).degree++;

					edgecount++;
					if(saveEdges)
						if(ti.elapsed() > 5.0) {
							cout << edgecount << " edges saved" << endl;
							ti.restart();
						}
				}
				getVert(edges[edges.size() - 1].source).edge_end = edges.size();
			}

			cout << "Nodes: " << nverts << " Edges: " << nedges <<endl;
			//memory_info::print_usage();

			cout << "finalized" << endl;
		}

		//// some utilities
		//vertex_list_type vertex_intersection(const vertex_list_type& list1, const vertex_list_type& list2) {
		//	vertex_list_type result;
		//	set_intersection(list1.begin(), list1.end(), list2.begin(), list2.end(), inserter(result, result.begin()));
		//	return result;
		//}

		//vertex_list_type vertex_union(const vertex_list_type& list1, const vertex_list_type& list2) {
		//	vertex_list_type result;
		//	set_union(list1.begin(), list1.end(), list2.begin(), list2.end(), inserter(result, result.begin()));
		//	return result;
		//}

		void load_format(const string& path, const string& format) {
			line_parser_type line_parser;
			if (format == "snap") {
				line_parser = builtin_parsers::snap_parser<basic_graph>;
				load_graph(path, line_parser);
			} else if (format == "adj") {
				line_parser = builtin_parsers::adj_parser<basic_graph>;
				load_graph(path, line_parser);
			} else if (format == "tsv") {
				line_parser = builtin_parsers::tsv_parser<basic_graph>;
				load_graph(path, line_parser);
			//} else if (format == "graphjrl") {
			//	line_parser = builtin_parsers::graphjrl_parser<basic_graph>;
			//	load_graph(path, line_parser);
			//} else if (format == "bintsv4") {
			//	load_direct(path,&graph_type::load_bintsv4_from_stream);
			//} else if (format == "bin") {
			//	load_binary(path);
			} else {
				cerr << "Unrecognized Format \"" << format << "\"!" << endl;
				return;
			}
		} // end of load

		void load_graph(string prefix, 
			line_parser_type line_parser) {
				string directory_name; string original_path(prefix);
				boost::filesystem::path path(prefix);
				string search_prefix;
				if (boost::filesystem::is_directory(path)) {
					// if this is a directory
					// force a "/" at the end of the path
					// make sure to check that the path is non-empty. (you do not
					// want to make the empty path "" the root path "/" )

					directory_name = path.generic_string();
				}
				else {
					directory_name = path.parent_path().generic_string();
					search_prefix = path.filename().generic_string();
					directory_name = (directory_name.empty() ? "." : directory_name);
				}
				vector<string> graph_files;
				fs_util::list_files_with_prefix(directory_name, search_prefix, graph_files);
				if (graph_files.size() == 0) {
					cerr << "No files found matching " << original_path << endl;
				}
				for(size_t i = 0; i < graph_files.size(); ++i) {
					cerr << "Loading graph from file: " << graph_files[i] << endl;
					// open the stream
					ifstream in_file(graph_files[i].c_str(), 
						ios_base::in | ios_base::binary);
					// attach gzip if the file is gzip
					boost::iostreams::filtering_stream<boost::iostreams::input> fin;  
					// Using gzip filter
					fin.push(in_file);
					const bool success = load_from_stream(graph_files[i], fin, line_parser);
					if(!success) { 
						cerr << "\n\tError parsing file: " << graph_files[i] << endl;
					}
					fin.pop();
				}
		} // end of load graph

		template<typename Fstream>
		bool load_from_stream(string filename, Fstream& fin, 
			line_parser_type& line_parser) {
				size_t linecount = 0;
				boost::timer ti;
				ti.restart();
				while(fin.good() && !fin.eof()) {
					string line;
					getline(fin, line);
					if(line.empty()) continue;
					if(fin.fail()) break;
					const bool success = line_parser(*this, filename, line);
					if (!success) {
						cerr
							<< "Error parsing line " << linecount << " in "
							<< filename << ": " << endl
							<< "\t\"" << line << "\"" << endl;  
						return false;
					}
					++linecount;      
					if (ti.elapsed() > 5.0) {
						cout << linecount << " Lines read" << endl;
						ti.restart();
					}
				}
				return true;
		} // end of load from stream

	}; // class graph_type
} // namespace graphp

#endif