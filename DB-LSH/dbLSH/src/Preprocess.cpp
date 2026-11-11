#include "Preprocess.h"
#include "basis.h"
#include <fstream>
#include <assert.h>
#include <random>
#include <iostream>
#include <fstream>
#include <map>
#include <ctime>
#include <sstream>
#include <numeric>
#include<algorithm>
using namespace std; 


#define E 2.718281746
#define PI 3.1415926

#define min(a,b)            (((a) < (b)) ? (a) : (b))

Preprocess::Preprocess(int _k, const std::string& path, const std::string& ben_file_)
{
	lsh::timer timer;
	k = _k;
	std::cout << "LOADING DATA..." << std::endl;
	timer.restart();
	load_data(path);
	std::cout << "LOADING TIME: " << timer.elapsed() << "s." << std::endl << std::endl;

	data_file = path;
	ben_file = ben_file_;
	if (data.N > 500) {
		ben_create();
	}
}

Preprocess::Preprocess(const std::string& path, const std::string& ben_file_, float beta_)
{

	hasT = true;
	beta = beta_;
	lsh::timer timer;
	std::cout << "LOADING DATA..." << std::endl;
	timer.restart();
	load_data(path);
	std::cout << "LOADING TIME: " << timer.elapsed() << "s." << std::endl << std::endl;

	data_file = path;
	ben_file = ben_file_;
	if (data.N > 1000) {
		ben_create();
	}
}


vector<vector<float>> readFVecsFromExternal(string filepath, int maxRow=-1) {
  FILE *infile = fopen(filepath.c_str(), "rb");
  vector<vector<float>> data= {};
  if (infile == NULL) {
    std::cout << "File not found" << std::endl;
    return data;
  }
  
  int rowCt = 0;
  int dimen;
  while (true) {
    if (fread(&dimen, sizeof(int), 1, infile) == 0) {
      break;
    }
    std::vector<float> v(dimen);
    if(fread(v.data(), sizeof(float), dimen, infile) == 0) {
      std::cout << "Error when reading" << std::endl;
    };
    
	data.push_back(v);

    rowCt++;
    
    if (maxRow != -1 && rowCt >= maxRow) {
      break;
    }
  }
  // std::cout<<"Row count test: "<<rowCt<<std::endl;

  if (fclose(infile)) {
    std::cout << "Could not close data file" << std::endl;
  }
  return data;
}

void Preprocess::load_data(const std::string& path)
{
	vector<vector<float>> base = readFVecsFromExternal(path+"base.fvecs");
	data.N = base.size();
	data.dim = base[0].size();
	data.val = new float* [data.N];
	for (int i = 0; i < data.N; ++i) {
		data.val[i] = new float[data.dim];
		//in.seekg(sizeof(float), std::ios::cur);
		for (int j=0;j<base[i].size();j++) {
			data.val[i][j]=base[i][j];
		}
	}
	vector<vector<float>> query = readFVecsFromExternal(path+"query.fvecs");
	data.numQuery = query.size();
	data.query = new float* [data.numQuery];
	for (int i = 0; i < data.numQuery; ++i) {
		data.query[i] = new float[data.dim];
		for (int j=0;j<query[i].size();j++) {
			data.query[i][j]=query[i][j];
		}
	}


	std::cout << "Load from new file: " << path << "\n";
	std::cout << "N=    " << data.N << "\n";
	std::cout << "dim=  " << data.dim << "\n\n";
	std::cout << "Qnum=  " << data.numQuery << "\n\n";
	std::cout << "size of base=  " << sizeof(&(data.val)) << "\n\n";
	std::cout << "size of query=  " << sizeof(&(data.query)) << "\n\n";
}

//void Preprocess::load_data(const std::string& path)
//{
//	std::string file = path + "_all";
//	std::ifstream is_new(file.c_str(), std::ios::binary);
//	unsigned header[3] = { 0 };
//	if (!is_new) {
//		std::ifstream is(path.c_str(), std::ios::binary);
//		if (!is) {
//			std::cout << "Fail to open the file:\n" << path << "!\n";
//			exit(-10086);
//		}
//
//		assert(sizeof header == 3 * 4);
//		is.read((char*)header, sizeof(header));
//		assert(header[1] != 0);
//		float* array_ = new float[header[2] * header[1]];
//		is.read((char*)&array_[0], sizeof(float) * header[2] * header[1]);
//		data.N = header[1];
//		//data.N = 100;
//		data.dim = header[2];
//		//Eigen::Map<Eigen::MatrixXf> data_(array_, header[1], header[2]);
//		//data.val = data_;
//		is.close();
//
//		data.val = new float* [data.N];
//		for (int i = 0; i < data.N; ++i) {
//			data.val[i] = new float[data.dim];
//			for (int j = 0; j < data.dim; ++j) {
//				data.val[i][j] = array_[j * data.N + i];
//			}
//		}
//		delete[] array_;
//
//		/*******************Restore the data by other approach******************/
//
//		int MaxQueryNum = std::min(200, (int)data.N - 1);
//		data.query = data.val;
//		data.val = &(data.query[200]);
//		data.N -= MaxQueryNum;
//
//		std::ofstream out(file.c_str(), std::ios::binary);
//		//out.write((char*)header, sizeof(header));
//		for (int i = 0; i < data.N; ++i) {
//			out.write((char*)&i, sizeof(int));
//			out.write((char*)data.val[i], sizeof(float) * header[2]);
//		}
//
//		out.close();
//
//		file.append("_query");
//		std::ofstream outq(file.c_str(), std::ios::binary);
//		//out.write((char*)header, sizeof(header));
//		for (int i = 0; i < MaxQueryNum; ++i) {
//			outq.write((char*)&i, sizeof(int));
//			outq.write((char*)data.query[i], sizeof(float) * header[2]);
//		}
//
//		outq.close();
//
//		exit(10010);
//	}
//	else {
//		file = path + "_new";
//		std::ifstream in(file.c_str(), std::ios::binary);
//		assert(sizeof header == 3 * 4);
//		in.read((char*)header, sizeof(header));
//		assert(header[1] != 0);
//		//float* array_ = new float[header[2] * header[1]];
//		//is.read((char*)&array_[0], sizeof(float) * header[2] * header[1]);
//		data.N = header[1];
//		//data.N = 100;
//		data.dim = header[2];
//		//Eigen::Map<Eigen::MatrixXf> data_(array_, header[1], header[2]);
//		//data.val = data_;
//		//is.close();
//
//		data.val = new float* [data.N];
//		for (int i = 0; i < data.N; ++i) {
//			data.val[i] = new float[data.dim];
//			//in.seekg(sizeof(float), std::ios::cur);
//			in.read((char*)data.val[i], sizeof(float) * header[2]);
//		}
//
//		int MaxQueryNum = std::min(200, (int)data.N - 1);
//		data.query = data.val;
//		data.val = &(data.query[200]);
//		data.N -= MaxQueryNum;
//
//		std::cout << "Load from new file:\n" << file << "****, " << header[2] << "!\n";
//
//		//for (int i = 0; i < data.N; ++i)
//
//		in.close();
//
//	}
//}

struct Tuple
{
	unsigned id;
	float dist;
};

bool comp(const Tuple& a, const Tuple& b)
{
	return a.dist < b.dist;
}

void Preprocess::ben_make()
{
	int MaxQueryNum = data.numQuery;
	benchmark.N = MaxQueryNum, benchmark.num = k;
	cout<<"Benchmark k updated to: "<<k<<endl;
	benchmark.indice = new int* [benchmark.N];
	benchmark.dist = new float* [benchmark.N];
	for (unsigned j = 0; j < benchmark.N; j++) {
		benchmark.indice[j] = new int[benchmark.num];
		benchmark.dist[j] = new float[benchmark.num];
	}

	lsh::progress_display pd(benchmark.N);
	for (unsigned j = 0; j < benchmark.N; j++)
	{
		std::vector<Tuple> dists(data.N);

		for (unsigned i = 0; i < data.N; i++)
		{
			dists[i].id = i;
			dists[i].dist = cal_dist(data.val[i], data.query[j], data.dim);
		}

		sort(dists.begin(), dists.end(), comp);
		for (unsigned i = 0; i < benchmark.num; i++)
		{
			benchmark.indice[j][i] = (int)dists[i].id;
			benchmark.dist[j][i] = dists[i].dist;
		}
		++pd;
	}

	//clear_2d_array(Coe_Mat, data.N);
}



void Preprocess::ben_save()
{
	std::ofstream out(ben_file.c_str(), std::ios::binary);
	out.write((char*)&benchmark.N, sizeof(unsigned));
	out.write((char*)&benchmark.num, sizeof(unsigned));

	for (unsigned j = 0; j < benchmark.N; j++) {
		out.write((char*)&benchmark.indice[j][0], sizeof(int) * benchmark.num);
	}

	for (unsigned j = 0; j < benchmark.N; j++) {
		out.write((char*)&benchmark.dist[j][0], sizeof(float) * benchmark.num);
	}

	out.close();
}

void Preprocess::ben_load()
{
	std::ifstream in(ben_file.c_str(), std::ios::binary);
	in.read((char*)&benchmark.N, sizeof(unsigned));
	in.read((char*)&benchmark.num, sizeof(unsigned));

	benchmark.indice = new int* [benchmark.N];
	benchmark.dist = new float* [benchmark.N];
	for (unsigned j = 0; j < benchmark.N; j++) {
		benchmark.indice[j] = new int[benchmark.num];
		in.read((char*)&benchmark.indice[j][0], sizeof(int) * benchmark.num);
	}

	for (unsigned j = 0; j < benchmark.N; j++) {
		benchmark.dist[j] = new float[benchmark.num];
		in.read((char*)&benchmark.dist[j][0], sizeof(float) * benchmark.num);
	}
	in.close();
}

void Preprocess::ben_create()
{
	unsigned a_test = data.N + 1;
	lsh::timer timer;
	std::ifstream in(ben_file.c_str(), std::ios::binary);
	in.read((char*)&a_test, sizeof(unsigned));
	in.close();
	if (a_test > 0 && a_test < data.N)//�ж��Ƿ��ܸ�дa_test
	{
		std::cout << "LOADING BENMARK..." << std::endl;
		timer.restart();
		ben_load();
		std::cout << "LOADING TIME: " << timer.elapsed() << "s." << std::endl << std::endl;

	}
	else
	{
		std::cout << "MAKING BENMARK..." << std::endl;
		timer.restart();
		ben_make();
		std::cout << "MAKING TIME: " << timer.elapsed() << "s." << std::endl << std::endl;

		std::cout << "SAVING BENMARK..." << std::endl;
		timer.restart();
		ben_save();
		std::cout << "SAVING TIME: " << timer.elapsed() << "s." << std::endl << std::endl;

	}
}

Preprocess::~Preprocess()
{
	int MaxQueryNum = data.numQuery;

	clear_2d_array(data.query, data.N + data.numQuery);
	//clear_2d_array(Dists, MaxQueryNum);
	clear_2d_array(benchmark.indice, benchmark.N);
	clear_2d_array(benchmark.dist, benchmark.N);
	delete[] SquareLen;
}


Parameter::Parameter(Preprocess& prep, unsigned L_, unsigned K_, float rmin_)
{
	N = prep.data.N;
	dim = prep.data.dim;
	L = L_;
	K = K_;
	MaxSize = 5;
	R_min = rmin_;
}

Parameter::~Parameter()
{
}