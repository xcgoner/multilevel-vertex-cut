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
		("nthreads", po::value<string>(), "Set the number of threads...")
		("strategy", po::value<string>(), "Set file partitioning strategy...")
		("prestrategy", po::value<string>(), "Set file pre-partitioning strategy...")
		("reverse", po::value<string>(), "Set the paramater of powerlaw...")
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

	vector<size_t> nthreads;
	if(vm.count("nthreads") > 0) {
		nthreads.clear();
		typedef boost::tokenizer< boost::char_separator<char> > tokenizers;
		boost::char_separator<char> sep(",");
		tokenizers tok(vm["nthreads"].as<string>(), sep);
		for(tokenizers::iterator beg=tok.begin(); beg!=tok.end(); ++beg){
			//cout << *beg << endl;
			nthreads.push_back(boost::lexical_cast<size_t>(*beg));
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

	vector<string> prestrategies;
	prestrategies.push_back("random");
	if(vm.count("prestrategy") > 0) {
		prestrategies.clear();
		typedef boost::tokenizer< boost::char_separator<char> > tokenizers;
		boost::char_separator<char> sep(",");
		tokenizers tok(vm["prestrategy"].as<string>(), sep);
		for(tokenizers::iterator beg=tok.begin(); beg!=tok.end(); ++beg){
			//cout << *beg << endl;
			prestrategies.push_back(*beg);
		}
	}

	graphp::basic_graph graph;

	if(vm.count("reverse") > 0 && vm["reverse"].as<string>() == "true")
		graph.isReverse = true;

	if(vm.count("file") > 0 && vm.count("format") > 0) {
		graph.load_format(vm["file"].as<string>(), vm["format"].as<string>());
	}
	
	graph.finalize();
	if(vm.count("nthreads") > 0) {
		if(nthreads.size() != nparts.size()) {
			cerr << "The length of paramater nthreads should be the same as that of nparts..." << endl;
			return 0;
		}
		graphp::partition_strategy::run_partition(graph, nparts, nthreads, prestrategies, strategies);
	}

#ifdef WIN32
	system("Pause");
#endif

	return 0;
}

