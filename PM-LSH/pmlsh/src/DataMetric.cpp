#include "DataMetric.h"
#include <algorithm>
#include <fstream>
#include "Timer.h"



void DataMetric::loadData(const string & data_file_,int dim_)
{
	this->dimension = dim_;
	this->dataset.clear();
	FILE *infile = fopen(data_file_.c_str(), "rb");
	if (infile == NULL) {
		std::cout << "File not found" << std::endl;
		return;
	}
	
	int rowCt = 0;
	int dimen;
	while (true) {
		if (fread(&dimen, sizeof(int), 1, infile) == 0) {
			break;
		}
		if (dimen!=dim_){
			cout<<"dimension mismatch"<<endl;
			exit(1);
		}
		std::vector<DATATYPE> v(dimen);
		if(fread(v.data(), sizeof(DATATYPE), dimen, infile) == 0) {
			std::cout << "Error when reading" << std::endl;
		};
		
		dataset.emplace_back(v);

		rowCt++;
	}

	if (fclose(infile)) {
		std::cout << "Could not close data file" << std::endl;
	}
}

void DataMetric::lowerDimFromAnother(const DataMetric& another, E2LSH* e2lsh, Config& config)
{
	dataset.resize(another.size());
	for (int i = 0; i < dataset.size(); ++i) {
		dataset[i] = e2lsh->getHashVal(another[i],config);
	}
	this->dimension = dataset[0].size();
}



