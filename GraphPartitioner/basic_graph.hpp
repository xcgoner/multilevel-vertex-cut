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

		typedef size_t part_t;

		// list of vertices
		typedef vector<vertex_id_type> vertex_list_type;

		// list of edges
		typedef vector<edge_id_type> edge_list_type;

		size_t nverts, nedges, nparts;

		vector<size_t> parts_counter;

		typedef boost::dynamic_bitset<> mirror_list_type;

		struct vertex_info {
			vector<size_t> weight;
			vector<vertex_list_type> nbr_list;
			vector<edge_list_type> edge_list;
			vector<mirror_list_type> mirror_list;
		};

		struct edge_info {
			vector<vertex_id_type> source;
			vector<vertex_id_type> target;
			vector<size_t> weight;
			vector<part_t> placement;
		};

		vertex_info verts;
		edge_info edges;

		// constructor
		basic_graph(const size_t nparts) : nverts(0), nedges(0), nparts(nparts) {
			parts_counter.resize(nparts);
			foreach(size_t& part_counter, parts_counter) {
				part_counter = 0;
			}
		}

		void add_vertex(const vertex_id_type& vid, const size_t& weight = 1) {
			if(vid >= verts.weight.size()) {
				verts.weight.resize(vid + 1);
				verts.nbr_list.resize(vid + 1);
				verts.edge_list.resize(vid + 1);
				verts.mirror_list.resize(vid + 1);
			}
			verts.weight[vid] = weight;
		}

		void add_edge(const vertex_id_type& source, const vertex_id_type& target, const size_t& weight = 1) {
			// check if the edge already exists
			add_vertex(source);
			add_vertex(target);
			// just check one of the two conditions should be ok...
			bool existence = false;
			for(size_t i = 0; i < verts.nbr_list[source].size(); i++) {
				if(verts.nbr_list[source][i] == target)
					existence = true;
			}
			if(existence)
				return;

			edges.source.push_back(source);
			edges.target.push_back(target);
			edges.weight.push_back(weight);
			edges.placement.push_back(-1);
			
			// undirected
			verts.edge_list[source].push_back(nedges);
			verts.nbr_list[source].push_back(target);

			verts.edge_list[target].push_back(nedges);
			verts.nbr_list[target].push_back(source);

			nedges++;
		}

		void finalize() {
			foreach(boost::dynamic_bitset<>& mirror_list, verts.mirror_list) {
				mirror_list.resize(nparts);
			}

			nverts = verts.weight.size();

			cout << "Nodes: " << nverts << " Edges: " << nedges <<endl;
			memory_info::print_usage();
		}

		// some utilities
		void vertex_intersection(const vertex_list_type& list1, const vertex_list_type& list2, vertex_list_type& result) {
			foreach(vertex_id_type vid1, list1) {
				foreach(vertex_id_type vid2, list2) {
					if(vid1 == vid2)
						result.push_back(vid1);
				}
			}
		}

		//void list_type vertex_union(const vertex_list_type& list1, const vertex_list_type& list2) {
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