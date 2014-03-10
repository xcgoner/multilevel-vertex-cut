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
		("order", po::value<string>(), "Set the order of stream...")
		("strategy", po::value<string>(), "Set the file partitioning strategy...")
		("type", po::value<string>(), "Set the streaming type...")
		("powerlaw", po::value<size_t>(), "Generate a synthetic powerlaw graph...")
		("alpha", po::value<double>(), "Set the paramater of powerlaw...")
		("beta", po::value<double>(), "Set the paramater of powerlaw...")
		("indegree", po::value<string>(), "Set the paramater of powerlaw...")
		("reverse", po::value<string>(), "Set the paramater of powerlaw...")
		("rearrange", po::value<string>(), "Rearrange the edges by their source...")
		("histprefix", po::value<string>(), "The prefix of histgram files...")
		("saveprefix", po::value<string>(), "The prefix of output files...")
		("times", po::value<size_t>(), "For random average...")
	;

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if(vm.count("help")) {
        cout << desc << "\n";
		system("Pause");
        return 0;
    }

	vector<graphp::part_t> nparts;
	nparts.push_back(2);
	if(vm.count("nparts") > 0) {
		nparts.clear();
		typedef boost::tokenizer< boost::char_separator<char> > tokenizers;
		boost::char_separator<char> sep(",");
		tokenizers tok(vm["nparts"].as<string>(), sep);
		for(tokenizers::iterator beg=tok.begin(); beg!=tok.end(); ++beg){
			//cout << *beg << endl;
			nparts.push_back(boost::lexical_cast<size_t>(*beg));
		}
	}

	vector<string> orders;
	if(vm.count("order") > 0) {
		orders.clear();
		typedef boost::tokenizer< boost::char_separator<char> > tokenizers;
		boost::char_separator<char> sep(",");
		tokenizers tok(vm["order"].as<string>(), sep);
		for(tokenizers::iterator beg=tok.begin(); beg!=tok.end(); ++beg){
			//cout << *beg << endl;
			orders.push_back(*beg);
		}
	}

	vector<string> strategies;
	strategies.push_back("random");
	if(vm.count("strategy") > 0) {
		strategies.clear();
		typedef boost::tokenizer< boost::char_separator<char> > tokenizers;
		boost::char_separator<char> sep(",");
		tokenizers tok(vm["strategy"].as<string>(), sep);
		for(tokenizers::iterator beg=tok.begin(); beg!=tok.end(); ++beg){
			//cout << *beg << endl;
			strategies.push_back(*beg);
		}
	}

	graphp::basic_graph graph;

	if(vm.count("indegree") > 0 && vm["indegree"].as<string>() == "true")
		graph.isInDegree = true;
	if(vm.count("reverse") > 0 && vm["reverse"].as<string>() == "true")
		graph.isReverse = true;
	if(vm.count("rearrange") > 0 && vm["rearrange"].as<string>() == "true")
		graph.rearrange = true;

	if(vm.count("powerlaw") > 0) {
		// true for in/out-degree
		if(vm.count("beta") == 0)
			graph.load_synthetic_powerlaw(vm["powerlaw"].as<size_t>(), false, vm["alpha"].as<double>());
		else
			graph.load_synthetic_powerlawio(vm["powerlaw"].as<size_t>(), vm["alpha"].as<double>(), vm["beta"].as<double>());
	}
	else if(vm.count("file") > 0 && vm.count("format") > 0) {
		graph.load_format(vm["file"].as<string>(), vm["format"].as<string>());
	}
	
	graph.finalize();

	if(vm.count("histprefix") > 0)
		graphp::partition_strategy::degreeHistgram(graph, vm["histprefix"].as<string>());
	else if(vm.count("saveprefix") > 0)
		graphp::partition_strategy::convertSnap(graph, vm["saveprefix"].as<string>());
	else {
		if(vm.count("times") > 0)
			graphp::partition_strategy::run_partition(graph, nparts, strategies, orders, vm["type"].as<string>(), vm["times"].as<size_t>());
		else
			graphp::partition_strategy::run_partition(graph, nparts, strategies, orders, vm["type"].as<string>());
	}

#ifdef WIN32
	system("Pause");
#endif

	return 0;
}

