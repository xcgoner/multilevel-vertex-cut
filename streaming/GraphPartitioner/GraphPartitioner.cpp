// GraphPartitioner.cpp : �������̨Ӧ�ó������ڵ�?
//

#include <iostream>
#include "basic_graph.hpp"
#include "partition_by_patoh.hpp"

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

	return 0;

	if(vm.count("strategy") == 0 || vm["strategy"].as<string>() == "random" || vm["strategy"].as<string>() == "random_refine")
		graphp::partition_strategy::random_partition(graph, nparts);
	else if(vm["strategy"].as<string>() == "oblivious" || vm["strategy"].as<string>() == "oblivious_refine")
		graphp::partition_strategy::greedy_partition(graph, nparts);
	else if(vm["strategy"].as<string>() == "degree" || vm["strategy"].as<string>() == "oblivious_refine")
		graphp::partition_strategy::greedy_partition2(graph, nparts);
	else if(vm["strategy"].as<string>() == "obliviousreorder")
		graphp::partition_strategy::greedy_reorder(graph, nparts);
	else if(vm["strategy"].as<string>() == "hypergraph")
		graphp::partition_strategy::partition_by_patoh(graph, nparts);
	else if(vm["strategy"].as<string>() == "fasthypergraph")
		graphp::partition_strategy::partition_by_patoh_fast(graph, nparts);
	else if(vm["strategy"].as<string>() == "oblivioush")
		graphp::partition_strategy::oblivious_hypergraph(graph, nparts);

	// refine
	if(vm["strategy"].as<string>() == "oblivious_refine" || vm["strategy"].as<string>() == "random_refine")
		graphp::partition_strategy::itr_cost_greedy_refinement(graph, nparts);

#ifdef WIN32
	system("Pause");
#endif

	return 0;
}

