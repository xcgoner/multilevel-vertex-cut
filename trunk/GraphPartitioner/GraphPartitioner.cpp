// GraphPartitioner.cpp : 定义控制台应用程序的入口点。
//

#include <iostream>
#include "basic_graph.hpp"
#include "partition_strategy.hpp"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace std;

int main(int argc, char* argv[])
{

	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "A vertex-cut based graph partitioner...")
		("file", po::value<string>(), "Set file path...")
		("format", po::value<string>(), "Set file format...")
		("nparts", po::value<size_t>(), "Set the number of partitions...")
		("strategy", po::value<string>(), "Set file partitioning strategy...")
	;

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if(vm.count("help")) {
        cout << desc << "\n";
		system("Pause");
        return 0;
    }

	size_t nparts = 2;
	if(vm.count("nparts") > 0) {
		nparts = vm["nparts"].as<size_t>();
	}

	graphp::basic_graph graph(nparts);

	if(vm.count("file") > 0 && vm.count("format") > 0) {
		graph.load_format(vm["file"].as<string>(), vm["format"].as<string>());
	}

	graph.finalize();

	if(vm.count("strategy") == 0 || vm["strategy"].as<string>() == "random")
		graphp::partition_strategy::random_partition(graph, nparts);
	else if(vm["strategy"].as<string>() == "oblivious")
		graphp::partition_strategy::greedy_partition(graph, nparts);

	//for(size_t i = 0; i < graph.origin_edges.size(); i++) {
	//	cout << graph.origin_edges[i].source << "->" << graph.origin_edges[i].target << " : " << graph.origin_edges[i].placement << endl;
	//}

	// refine
	graphp::partition_strategy::itr_cost_greedy_refinement(graph, nparts);

#ifdef WIN32
	system("Pause");
#endif

	return 0;
}

