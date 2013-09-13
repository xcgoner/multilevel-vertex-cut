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
			vertex_list_type nbr_list;

			// key: neighbour target vid, value: edge eid
			edge_list_type edge_list;

			boost::dynamic_bitset<> mirror_list;

			vertex_type() { }

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

		deque<edge_type> edges_storage;

		//deque<edge_type>& edges = edges_storage;

		// constructor
		basic_graph(size_t nparts) : nverts(0), nedges(0), max_vid(0), nparts(nparts) {
			parts_counter.resize(nparts);
			foreach(size_t& num_edges, parts_counter) {
				num_edges = 0;
			}
		}

		bool add_vertex(const vertex_id_type& vid, const size_t& weight = 1) {
			if(verts[vid].isFree()) {
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

		//void add_edge(const vertex_id_type& source, const vertex_id_type& target, const size_t& weight = 1, const part_t& placement = -1) {
		//	// check if the edge already exists
		//	add_vertex(source);
		//	add_vertex(target);
		//	// just check one of the two conditions should be ok...
		//	if(origin_verts[source].nbr_list.count(target) > 0 || origin_verts[target].nbr_list.count(source) > 0)
		//		return ;

		//	edge_type e(nedges, source, target, weight);
		//	if(placement != -1)
		//		e.placement = placement;
		//	origin_edges.push_back(e);
		//	
		//	// undirected
		//	origin_verts[source].edge_list.insert(pair<vertex_id_type, edge_id_type>(target, e.eid));
		//	origin_verts[source].nbr_list.insert(target);
		//	
		//	origin_verts[target].edge_list.insert(pair<vertex_id_type, edge_id_type>(source, e.eid));
		//	origin_verts[target].nbr_list.insert(source);
		//	nedges++;
		//}
		//void add_edge(edge_type& e) {
		//	vertex_id_type source = e.source, target = e.target;

		//	// check if the edge already exists
		//	add_vertex(source);
		//	add_vertex(target);
		//	// just check one of the two conditions should be ok...
		//	if(origin_verts[source].nbr_list.count(target) > 0 || origin_verts[target].nbr_list.count(source) > 0)
		//		return ;

		//	e.eid = nedges;
		//	origin_edges.push_back(e);

		//	// undirected
		//	origin_verts[source].edge_list.insert(pair<vertex_id_type, edge_id_type>(target, e.eid));
		//	origin_verts[source].nbr_list.insert(target);
		//	origin_verts[source].degree++;

		//	origin_verts[target].edge_list.insert(pair<vertex_id_type, edge_id_type>(source, e.eid));
		//	origin_verts[target].nbr_list.insert(source);
		//	origin_verts[target].degree++;
		//	nedges++;
		//}

		void clear_partition_counter() {
			foreach(size_t& num_edges, parts_counter) {
				num_edges = 0;
			}
		}
		void clear_partition() {
			foreach(edge_type& e, edges_storage) {
				e.placement = -1;
			}
		}
		void clear_mirrors() {
			foreach(vertex_type& v, verts) {
				v.mirror_list.reset();
			}
		}

		void finalize() {
			cout << "finalizing..." << endl;

			//edges.reserve(edges_storage.size() + 1);
			verts.resize(max_vid + 1);

			boost::timer ti;
			size_t edgecount = 0;
			for(deque<edge_type>::iterator itr = edges_storage.begin(); itr != edges_storage.end(); ++itr) {
				// add edge
				//edges[edgecount] = itr;

				// add vertex
				// treat every single edge as an undirected one
				add_vertex(itr->source);
				verts[itr->source].nbr_list.push_back(itr->target);
				verts[itr->source].edge_list.push_back(edgecount);
				add_vertex(itr->target);
				verts[itr->target].nbr_list.push_back(itr->source);
				verts[itr->target].edge_list.push_back(edgecount);

				edgecount++;
				if(ti.elapsed() > 5.0) {
					cout << edgecount << " edges saved" << endl;
					ti.restart();
				}
			}

			// release the memory
			//edges_storage.clear();

			cout << "Nodes: " << nverts << " Edges: " << nedges <<endl;
			memory_info::print_usage();

			cout << "finalized" << endl;
		}

		vertex_type& getVert(vertex_id_type vid) {
			return verts[vid];
		}
		edge_type& getEdge(edge_id_type eid) {
			return edges_storage[eid];
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