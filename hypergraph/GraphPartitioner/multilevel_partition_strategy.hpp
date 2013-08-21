#ifndef GRAPH_PARTITIONER_GRAPH_MULTILEVEL_PARTITION_STRATEGY
#define GRAPH_PARTITIONER_GRAPH_MULTILEVEL_PARTITION_STRATEGY

#include <map>

#include <boost/dynamic_bitset.hpp>
#include <boost/timer.hpp>
#include "basic_graph.hpp"
#include "util.hpp"
#include "partition_strategy.hpp"

using namespace std;
#ifdef __GNUC__ 
using namespace __gnu_cxx; 
#endif 

namespace graphp {

	namespace partition_strategy {

		class multilevel_partition {

			basic_graph tmp_graph;

			typedef map<vertex_id_type, vertex_id_type> vmap_type;
			typedef map<edge_id_type, edge_id_type> emap_type;

			// maps used to projection
			vector<vmap_type> vmaps;
			vector<emap_type> emaps;

			// the # of levels
			size_t nlevels;

			// internal helper funcs

			void coarsen(const basic_graph& graph, size_t level) {
				// set true if vertex is matched
				boost::dynamic_bitset<> vlocks(graph.nverts, false);

				// visit the edges in random order
				vector<edge_id_type> edge_order(graph.nedges);
				for(edge_id_type i = 0; i < graph.nedges; i++) {
					edge_order[i] = i;
				}
				random_shuffle(edge_order.begin(), edge_order.end());
			}

			void partition(basic_graph& graph, basic_graph::part_t nparts) {
				boost::timer ti;

				cout << "Time elapsed: " << ti.elapsed() << endl;

				report_performance(graph, nparts);
			}
		};

	} // end of namespace partition_strategy

} // end of namespace graphp

#endif