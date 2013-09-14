// GraphPartitioner.cpp : �������̨Ӧ�ó������ڵ�?
//

#include <iostream>
#include "basic_graph.hpp"
#include "partition_strategy.hpp"
#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>

namespace po = boost::program_options;

using namespace std;

int main(int argc, char* argv[])
{

	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "A vertex-cut based graph partitioner...")
		("file", po::value<string>(), "Set file path...")
		("format", po::value<string>(), "Set file format...")
		("nparts", po::value<string>(), "Set the number of partitions...")
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

	vector<graphp::basic_graph::part_t> nparts;
	nparts.push_back(2);
	if(vm.count("nparts") > 0) {
		nparts.clear();
		typedef boost::tokenizer<boost::char_separator<char>> tokenizers;
		boost::char_separator<char> sep(",");
		tokenizers tok(vm["nparts"].as<string>(), sep);
		for(tokenizers::iterator beg=tok.begin(); beg!=tok.end(); ++beg){
			//cout << *beg << endl;
			nparts.push_back(boost::lexical_cast<size_t>(*beg));
		}
	}

	vector<string> strategies;
	strategies.push_back("random");
	if(vm.count("strategy") > 0) {
		strategies.clear();
		typedef boost::tokenizer<boost::char_separator<char>> tokenizers;
		boost::char_separator<char> sep(",");
		tokenizers tok(vm["strategy"].as<string>(), sep);
		for(tokenizers::iterator beg=tok.begin(); beg!=tok.end(); ++beg){
			//cout << *beg << endl;
			strategies.push_back(*beg);
		}
	}

	graphp::basic_graph graph(nparts[0]);

	if(vm.count("file") > 0 && vm.count("format") > 0) {
		graph.load_format(vm["file"].as<string>(), vm["format"].as<string>());
	}

	graph.finalize();

	//if(vm.count("strategy") == 0 || vm["strategy"].as<string>() == "random")
	//	graphp::partition_strategy::random_partition(graph, nparts);
	//else if(vm["strategy"].as<string>() == "greedy")
	//	graphp::partition_strategy::greedy_partition(graph, nparts);
	//else if(vm["strategy"].as<string>() == "degree")
	//	graphp::partition_strategy::greedy_partition2(graph, nparts);
	

	graphp::partition_strategy::run_partition(graph, nparts, strategies);

#ifdef WIN32
	system("Pause");
#endif

	return 0;
}

