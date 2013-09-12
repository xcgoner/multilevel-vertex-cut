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
#include <list>
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

		typedef size_t part_t;

		// list of vertices
		typedef set<vertex_id_type> vertex_list_type;

		// list of edges
		typedef set<edge_id_type> edge_list_type;

		size_t nverts, nedges, nparts;

		vertex_id_type max_vid;

		vector<size_t> parts_counter;

		typedef map<vertex_id_type, edge_id_type> vertex_edge_map_type;
		struct vertex_type {
			vertex_id_type vid;
			size_t degree;
			size_t weight;

			// neighbour list
			vertex_list_type nbr_list;

			// key: neighbour target vid, value: edge eid
			vertex_edge_map_type edge_list;

			set<part_t> mirror_list;

			vertex_type() :
				vid(-1), degree(0), weight(1) { }
			vertex_type(const vertex_id_type& vid) :
				vid(vid), degree(0), weight(1) { }
			vertex_type(const vertex_id_type& vid, const size_t& weight) :
				vid(vid), degree(0), weight(weight) { }

			bool operator==(vertex_type& v) const {
				return vid == v.vid;
			}



			friend class basic_graph;
		};

		struct edge_type {
			edge_id_type eid;
			vertex_id_type source;
			vertex_id_type target;
			size_t weight;
			part_t placement;
			edge_type() :
				eid(-1), weight(1), placement(-1) { }
			edge_type(const edge_id_type& eid) :
				eid(eid), weight(1), placement(-1) { }
			edge_type(const edge_id_type& eid, const size_t& weight) :
				eid(eid), weight(weight), placement(-1) { }
			edge_type(const edge_id_type& eid, const vertex_id_type& source, const vertex_id_type& target, const size_t& weight) :
				eid(eid), source(source), target(target), weight(weight), placement(-1) { }
			bool operator==(edge_type& e) const {
				return eid == e.eid;
			}

			friend class basic_graph;
		};

		typedef map<vertex_id_type, vertex_type> verts_map_type;
		verts_map_type origin_verts;
		vector<edge_type> origin_edges;

		list<edge_type> edges_storage;

		// constructor
		basic_graph(size_t nparts) : nverts(0), nedges(0), max_vid(0), nparts(nparts) {
			parts_counter.resize(nparts);
			foreach(size_t& num_edges, parts_counter) {
				num_edges = 0;
			}
		}

		bool add_vertex(const vertex_id_type& vid, const size_t& weight = 1) {
			if(origin_verts.count(vid) == 0) {
				if(vid > max_vid)
					max_vid = vid;
				vertex_type v(vid, weight);
				origin_verts.insert(pair<vertex_id_type, vertex_type>(vid, v));
				nverts++;
				return true;
			}
			return false;
		}

		void add_edge_to_storage(const vertex_id_type& source, const vertex_id_type& target, const size_t& weight = 1, const part_t& placement = -1) {
			edge_type e(-1, source, target, weight);
			if(placement != -1)
				e.placement = placement;
			edges_storage.push_back(e);
		}

		void add_edge(const vertex_id_type& source, const vertex_id_type& target, const size_t& weight = 1, const part_t& placement = -1) {
			// check if the edge already exists
			add_vertex(source);
			add_vertex(target);
			// just check one of the two conditions should be ok...
			if(origin_verts[source].nbr_list.count(target) > 0 || origin_verts[target].nbr_list.count(source) > 0)
				return ;

			edge_type e(nedges, source, target, weight);
			if(placement != -1)
				e.placement = placement;
			origin_edges.push_back(e);
			
			// undirected
			origin_verts[source].edge_list.insert(pair<vertex_id_type, edge_id_type>(target, e.eid));
			origin_verts[source].nbr_list.insert(target);
			origin_verts[source].degree++;
			
			origin_verts[target].edge_list.insert(pair<vertex_id_type, edge_id_type>(source, e.eid));
			origin_verts[target].nbr_list.insert(source);
			origin_verts[target].degree++;
			nedges++;
		}
		void add_edge(edge_type& e) {
			vertex_id_type source = e.source, target = e.target;

			// check if the edge already exists
			add_vertex(source);
			add_vertex(target);
			// just check one of the two conditions should be ok...
			if(origin_verts[source].nbr_list.count(target) > 0 || origin_verts[target].nbr_list.count(source) > 0)
				return ;

			e.eid = nedges;
			origin_edges.push_back(e);

			// undirected
			origin_verts[source].edge_list.insert(pair<vertex_id_type, edge_id_type>(target, e.eid));
			origin_verts[source].nbr_list.insert(target);
			origin_verts[source].degree++;

			origin_verts[target].edge_list.insert(pair<vertex_id_type, edge_id_type>(source, e.eid));
			origin_verts[target].nbr_list.insert(source);
			origin_verts[target].degree++;
			nedges++;
		}

		void clear_partition_counter() {
			foreach(size_t& num_edges, parts_counter) {
				num_edges = 0;
			}
		}
		void clear_partition() {
			foreach(edge_type& e, origin_edges) {
				e.placement = -1;
			}
		}
		void clear_mirrors() {
			foreach(verts_map_type::value_type& vp, origin_verts) {
				vp.second.mirror_list.clear();
			}
		}

		void finalize() {
			cout << "finalizing..." << endl;

			nedges = 0;
			origin_edges.reserve(edges_storage.size() + 1);
			foreach(edge_type& e, edges_storage) {
				add_edge(e);
				//cout << nedges << " "<< e.eid << " " << e.source << " " << e.target << " " << e.weight << " " << e.placement << endl;
			}

			// release the memory
			edges_storage.clear();

			cout << "Nodes: " << origin_verts.size() << " Edges: " << origin_edges.size() <<endl;
			memory_info::print_usage();

			cout << "finalized" << endl;
		}

		// some utilities
		vertex_list_type vertex_intersection(const vertex_list_type& list1, const vertex_list_type& list2) {
			vertex_list_type result;
			set_intersection(list1.begin(), list1.end(), list2.begin(), list2.end(), inserter(result, result.begin()));
			return result;
		}

		vertex_list_type vertex_union(const vertex_list_type& list1, const vertex_list_type& list2) {
			vertex_list_type result;
			set_union(list1.begin(), list1.end(), list2.begin(), list2.end(), inserter(result, result.begin()));
			return result;
		}

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