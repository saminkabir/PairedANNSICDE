#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <getopt.h>
#include <sys/sysinfo.h>
#include <sys/stat.h>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Dense>
#include "VAQ.hpp"
#include "utils/TimingUtils.hpp"
#include "utils/Experiment.hpp"
#include "utils/IO.hpp"
using namespace std;

template<typename T>
void save_to_fvecs(const char* filename, T* &dataset, int n, int d)
{

	FILE * ofp = fopen(filename, "w");
  // cout<<"Reach 2"<<endl;
	for(int i=0;i<n;i++)
	{
		fwrite(&d,sizeof(int),1,ofp);
		fwrite(&dataset[i*d],sizeof(T),d,ofp);
	}

  // cout<<"Reach 3"<<endl;
	fclose(ofp);	

}


template<typename T>
void save_to_fvecsV1(const char* filename, const std::vector<int> &labels, int n, int d)
{

  T* dataset = new T[n*d];

  // cout<<"Reach 0"<<labels.size()<<endl;
  for(int i=0; i<n*d; i++) {
    dataset[i] = labels[i];
  }

  // cout<<"Reach 1"<<endl;
  save_to_fvecs<T>(filename, dataset, n, d);
}

double myrand(std::string datasetName) {
  // if(datasetName.find("MNIST")!=-1) {
  //   return 0;
  // }
  return (1 + rand() % 1000)/10000.00;
	
}

RowMatrixXf fillWithRandomNumber(RowMatrixXf &X, std::string datasetName) {
  for (int rowIdx=0; rowIdx<X.rows(); rowIdx++) {
    for (int colIdx=0; colIdx<X.cols(); colIdx++) {
      X(rowIdx, colIdx) = myrand(datasetName);
    }
  }
  return X;
}


int main(int argc, char **argv) {
  
  std::vector<ArgsParse::opt> long_options {
    {"removeProjection", 'i', "0"},
    {"test-id", 's', ""},
    {"dataset", 's', ""},
    {"queries", 's', ""},
    {"file-format-ori", 's', "fvecs"},
    {"save", 's', ""},
    {"save-enc", 's', ""},
    {"groundtruth", 's', ""},
    {"groundtruth-format", 's', "ascii"},
    {"result", 's', ""},
    {"timeseries-size", 'i', "1"},
    {"dataset-size", 'i', "0"},
    {"queries-size", 'i', "0"},
    {"k", 'i', "100"},
    {"spDim", 'i', "4"},
    {"spSubSize", 'i', "1"},
    {"minDims", 'i', "64"},
    {"cumVar", 'f', "0.95"},
    {"samplingMultiplier", 'i', "1000"},
    {"portionOfSubspace", 'f', "0.65"},
    {"method", 's', "VAQ256m32min7max13var1,EA"},
    // VAQ256m32min7max13var1,SORT means VAQ 256 bits (32 bytes), 32 subvector, 7 min bits per segment, 1 variance (no compression), SORT the result
    // another description example: 
    // - VAQ64m16min3max6var0.99,EA
    // - PCA,PQ256m32,TI100
    // - VAQ128m32min6max9var0.95,EA_TI200
    // - VAQ128m32min6max9var0.95,Heap
    {"refine", 's', ""},
    {"hc-bitalloc", 's', ""},
    {"learn-ratio", 'f', "0.05"},
    {"visit-cluster", 'f', "1"},
    {"kmeans-ver", 'i', "0"},
    {"new-alg", 'i', "0"},
    {"segmentation_alg", 'i', "0"},
    {"bit_alloc_algo", 'i', "0"},
    {"sub-space-swapping", 'i', "1"},
    {"kmeans-sub-space", 'i', "0"},
    {"exp-var-multiplier", 'f', "2.0"},
    {"extra-bit", 'i', "1"},
    {"min-reduced-bit", 'i', "0"},
    {"max-increased-bit", 'i', "2"},
    {"faissKMeans", 'i', "0"},
    {"dataCentering", 'i', "0"},
    {"nonUniformSubspaces", 'i', "0"},
    {"storeResults", 'i', "0"},
    {"fixedSamplingForKmeans", 'i', "0"},
    {"swappingLimit", 'i', "-1"},
    {"minDimsPerSubspace", 'i', "1"},
    {"minReductionThreshold", 'f', "0.0"},
    {"minReducedDims", 'i', "-1"},
    {"minReducedDimsPercentage", 'f', "-1.0"},
    {"static_kmeans_sample", 'i', "1"}
  };
  ArgsParse args = ArgsParse(argc, argv, long_options, "HELP");
  char fn[255];
  strcpy(fn, "finalTests2/vaq-log-");
  strcat(fn,args["test-id"].c_str());
  strcat(fn, ".txt");
  freopen(fn,"w+",stdout);
  // std::cout<<"test id: "<<args["test-id"]<<endl;
  args.printArgs();

  // check if dataset and queries exist
  if (!isFileExists(args["dataset"]) || !isFileExists(args["queries"])) {
    std::cerr << "Dataset or queries file doesn't exists" << std::endl;
    return 1;
  }

  VAQ vaq;
  vaq.parseMethodString(args["method"]);
  vaq.mVisit = args.at<float>("visit-cluster");
  if (args.at<int>("kmeans-ver") == 1) {
    vaq.mHierarchicalKmeans = true;
  } else if (args.at<int>("kmeans-ver") == 2) {
    vaq.mBinaryKmeans = true;
  }
  if (args.at<int>("removeProjection") == 1) {
    vaq.removeProjection = true;
  }
  if (args.at<int>("new-alg") != 0) {
    vaq.new_alg = args.at<int>("new-alg");
  }
  // if (args.at<int>("sub-space-swapping") == 0) {
  //   vaq.sub_space_swapping = false;
  // }
  vaq.sub_space_swapping=args.at<int>("sub-space-swapping");
  if (args.at<int>("kmeans-sub-space") == 1) {
    vaq.kmeans_sub_space = true;
  }
  vaq.exp_var_multiplier = args.at<float>("exp-var-multiplier");
  vaq.extra_bit = args.at<int>("extra-bit");
  vaq.min_reduced_bit = args.at<int>("min-reduced-bit");
  vaq.max_increased_bit = args.at<int>("max-increased-bit");
  vaq.spDim = args.at<int>("spDim");
  vaq.spSubSize = args.at<int>("spSubSize");
  vaq.minDims = args.at<int>("minDims");
  vaq.cumVar = args.at<float>("cumVar");
  vaq.faissKMeans = args.at<int>("faissKMeans");
  int dataCentering = args.at<int>("dataCentering");
  vaq.dataCentering = args.at<int>("dataCentering");
  vaq.nonUniformSubspaces = args.at<int>("nonUniformSubspaces");
  vaq.fixedSamplingForKmeans = args.at<int>("fixedSamplingForKmeans");
  vaq.samplingMultiplier = args.at<int>("samplingMultiplier");
  vaq.portionOfSubspace = args.at<float>("portionOfSubspace");
  vaq.swappingLimit = args.at<int>("swappingLimit");
  vaq.minDimsPerSubspace = args.at<int>("minDimsPerSubspace");
  vaq.minReductionThreshold = args.at<float>("minReductionThreshold");
  vaq.minReducedDims = args.at<int>("minReducedDims");
  vaq.segmentation_alg = args.at<int>("segmentation_alg");
  vaq.minReducedDimsPercentage = args.at<float>("minReducedDimsPercentage");
  vaq.bit_alloc_algo = args.at<int>("bit_alloc_algo");
  vaq.static_kmeans_sample = args.at<int>("static_kmeans_sample");

  std::cout << "Preprocessing steps..\n" << std::endl;
  
  int dimPadding = 0;
  if (!(vaq.nonUniformSubspaces || vaq.new_alg==5 || vaq.new_alg==6 || vaq.new_alg==7 || vaq.new_alg==8 || vaq.new_alg==12 || vaq.new_alg==13 || vaq.new_alg==14) && args.at<int>("timeseries-size") % vaq.mSubspaceNum != 0) {
    int subvectorlen = args.at<int>("timeseries-size") / vaq.mSubspaceNum;
    subvectorlen += (args.at<int>("timeseries-size") % vaq.mSubspaceNum > 0) ? 1 : 0;
    dimPadding = (subvectorlen * vaq.mSubspaceNum) - args.at<int>("timeseries-size");
  }
  RowMatrixXf dataset = RowMatrixXf::Zero(args.at<int>("dataset-size"), args.at<int>("timeseries-size") + dimPadding);
  std:string datasetName=args["dataset"];
  // dataset = fillWithRandomNumber(dataset, datasetName);
  Eigen::VectorXf XTrainMean;
  {
    std::cout << "Read dataset" << std::endl;
    if (args["file-format-ori"] == "ascii") {
      readOriginalFromExternal<true>(args["dataset"], dataset, args.at<int>("timeseries-size"), ',');
    } else if (args["file-format-ori"] == "fvecs") {
      readFVecsFromExternal(args["dataset"], dataset, args.at<int>("timeseries-size"), args.at<int>("dataset-size"));
    } else if (args["file-format-ori"] == "bvecs") {
      readBVecsFromExternal(args["dataset"], dataset, args.at<int>("timeseries-size"), args.at<int>("dataset-size"));
    } else if (args["file-format-ori"] == "bin") {
      readFromExternalBin(args["dataset"], dataset, args.at<int>("timeseries-size"), args.at<int>("dataset-size"));
    }

    std::cout << "Training & encoding phase" << std::endl;

    if (args["save"] != "" && isFileExists(args["save"])) {
      std::cout << "Reading saved centroids from " << args["save"] << std::endl;
      vaq.mCentroidsPerSubs = loadCentroids(args["save"]);
      vaq.mCentroidsPerSubsCMajor.resize(vaq.mCentroidsPerSubs.size());
      for (int i=0; i<(int)vaq.mCentroidsPerSubs.size(); i++) {
        vaq.mCentroidsPerSubsCMajor[i] = vaq.mCentroidsPerSubs[i];
      }
    }
    if (args["hc-bitalloc"] != "") {
      vaq.mBitsAlloc = parseVAQHardcode(args["hc-bitalloc"]);
      std::cout << "Hardcoded bit allocation loaded" << std::endl;
    }

    // encoding phase
    START_TIMING(VAQ_TRAIN);
    if (dataCentering==1) {
      Eigen::VectorXf v(2);
      Eigen::MatrixXf mat(2,4);
      XTrainMean=dataset.colwise().mean();
      RowMatrixXf datasetTranspose=dataset.transpose();

      datasetTranspose.colwise()-=XTrainMean;
      dataset=datasetTranspose.transpose();
      std::cout << dataset.cols() << dataset.rows() << std::endl;
    }
    
    
    if (args["save"] == "" || !isFileExists(args["save"])) {
      std::cout << "Training the centroids" << std::endl;
      vaq.train(dataset, true);
    }
    END_TIMING(VAQ_TRAIN, "== Training time: ");
    // return 0;

    START_TIMING(VAQ_ENCODE);
    if (args["save-enc"] != "" && isFileExists(args["save-enc"])) {
      if (args["save"] == "") {
        std::cout << "Attempt to read encoded dataset without reading saved centroids" << std::endl;
        std::cout << "Exiting" << std::endl;
        exit(0);
      }
      std::cout << "using saved encoded dataset" << std::endl;
      vaq.mCodebook = loadCodebook<CodebookType>(args["save-enc"]);
    } else {
      vaq.encode(dataset);
    }

    if ((vaq.searchMethod() & VAQ::NNMethod::Fast) || (vaq.searchMethod() & VAQ::NNMethod::Fast3)) {
      START_TIMING(LEARN_QUANTIZATION);
      vaq.learnQuantization(dataset, args.at<float>("learn-ratio"));
      END_TIMING(LEARN_QUANTIZATION, "== Learn Quantization time: ");
    }
    END_TIMING(VAQ_ENCODE, "== Encoding time: ");

    dataset.resize(0, 0); // release dataset memory


    if (vaq.searchMethod() & VAQ::NNMethod::TI) {
      START_TIMING(TI_CLUSTER);
      vaq.clusterTI(true, true);
      END_TIMING(TI_CLUSTER, "== TI Clustering time: ");
    }

    if (args["save"] != "" && !isFileExists(args["save"])) {
      std::cout << "Saving centroids to " << args["save"] << std::endl;
      saveCentroids(vaq.mCentroidsPerSubs, args["save"]);
    }

    if (args["save-enc"] != "" && !isFileExists(args["save-enc"])) {
      std::cout << "Saving codebook to " << args["save-enc"] << std::endl;
      saveCodebook(vaq.mCodebook, args["save-enc"]);
    }
  }

  {
    RowMatrixXf queries = RowMatrixXf::Zero(args.at<int>("queries-size"), args.at<int>("timeseries-size") + dimPadding);
    // queries = fillWithRandomNumber(queries);

    std::cout << "Read queries" << std::endl;
    if (args["file-format-ori"] == "ascii") {
      readOriginalFromExternal<true>(args["queries"], queries, args.at<int>("timeseries-size"), ',');
    } else if (args["file-format-ori"] == "fvecs") {
      readFVecsFromExternal(args["queries"], queries, args.at<int>("timeseries-size"), args.at<int>("queries-size"));
    } else if (args["file-format-ori"] == "bvecs") {
      readBVecsFromExternal(args["queries"], queries, args.at<int>("timeseries-size"), args.at<int>("queries-size"));
    } else if (args["file-format-ori"] == "bin") {
      readFromExternalBin(args["queries"], queries, args.at<int>("timeseries-size"), args.at<int>("queries-size"));
    }
    if (dataCentering==1) {
      RowMatrixXf queriesTranspose=queries.transpose();
      queriesTranspose.colwise()-=XTrainMean;
      queries=queriesTranspose.transpose();
    }
    std::vector<std::vector<int>> topnn;
    std::vector<std::vector<float>> topnnDist;
    if (args["groundtruth"] != "") {
      std::cout << "Read groundtruth" << std::endl;
      if (args["groundtruth-format"] == "ascii") {
        readTOPNNExternal(args["groundtruth"], topnn, args.at<int>("k"), ',');
      } else if (args["groundtruth-format"] == "ivecs") {
        readIVecsFromExternal(args["groundtruth"], topnn, args.at<int>("k"));
      } else if (args["groundtruth-format"] == "bin") {
        readTOPNNExternalBin(args["groundtruth"], topnn, args.at<int>("k"));
      }
    }

    std::cout << "Querying phase" << std::endl;
    std::vector<int> refines;
    if (args["refine"] != "") {
      std::stringstream ss(args["refine"]);
      while (ss.good())
      {
        std::string substr;
        getline(ss, substr, ',');
        refines.push_back(std::stoi(substr));
      }
    } else {
      refines.push_back(0);
    }

    RowMatrixXf datasetrefine;
    if (refines.size() > 1 || refines[0] > 0) {
      datasetrefine.resize(args.at<int>("dataset-size"), args.at<int>("timeseries-size") + dimPadding);
      datasetrefine.setZero();
      std::cout << "Read refining dataset" << std::endl;
      if (args["file-format-ori"] == "ascii") {
        readOriginalFromExternal<true>(args["dataset"], datasetrefine, args.at<int>("timeseries-size"), ',');
      } else if (args["file-format-ori"] == "fvecs") {
        readFVecsFromExternal(args["dataset"], datasetrefine, args.at<int>("timeseries-size"), args.at<int>("dataset-size"));
      } else if (args["file-format-ori"] == "bvecs") {
        readBVecsFromExternal(args["dataset"], datasetrefine, args.at<int>("timeseries-size"), args.at<int>("dataset-size"));
      } else if (args["file-format-ori"] == "bin") {
        readFromExternalBin(args["dataset"], datasetrefine, args.at<int>("timeseries-size"), args.at<int>("dataset-size"));
      }
    }
    // datasetrefine.resize(0, 0);
    
    
    vaq.computeDistance(queries, datasetrefine, topnn, topnnDist, args.at<int>("k"));
    for (const int refine: refines) {
      START_TIMING(QUERY);                                                                                                                                                                    
      int searchK = refine >= args.at<int>("k") ? refine : args.at<int>("k");
      LabelDistVecF answers = vaq.search(queries, searchK, true);
      // cout<<answers.labels.size()<<endl;                                                                                                           
      // cout<<getAvgRecall(answers.labels, topnn, 10, searchK)<<endl;
      // std::cout << "recall@5 : " << getAvgRecall(answers.labels, topnn, 5, searchK)<<std::endl;
      // std::cout << "recall@10 : " << getAvgRecall(answers.labels, topnn, 10, searchK)<<std::endl;
      // std::cout << "recall@20 : " << getAvgRecall(answers.labels, topnn, 20, searchK)<<std::endl;
      if (refine == args.at<int>("k")) {
        END_TIMING(QUERY, "== Querying time: ");
      }
      if (refine >= args.at<int>("k")) {
        std::cout << "Refining the answer with Refine = " << refine << std::endl;
        answers = vaq.refine(queries, answers, datasetrefine, args.at<int>("k"));
      }
      // for(int i=0;i<answers.distances.size() & i<200;i++) {
      //   cout<<i<<" "<<answers.distances[i]<<endl;
      // }
      if (refine > args.at<int>("k")) {
        END_TIMING(QUERY, "== Querying time: ");
      }
      
      // if (args["result"] != "") {
      //   std::string resultFP = args["result"];
      //   if (refines.size() > 1) {
      //     resultFP.append("_R" + std::to_string(refine));
      //   }
      //   std::cout << "Writing knn results to " << resultFP << std::endl;
      //   writeKNNResults(resultFP, answers, queries.rows());;
      // }
      if (args["groundtruth"] != "") {
        // measure precision
        std::cout << "\trecall@R (Probably correct): " << getAvgRecall(answers.labels, topnn, args.at<int>("k"), args.at<int>("k"))<<" " <<answers.labels.size()<< 
        " "<<topnn.size()<<std::endl;
        std::cout << "\trecall@R (Probably correct) distance: " << getAvgRecallDistance(answers.distances, topnnDist, args.at<int>("k"))<<" " <<answers.labels.size()<< 
        " "<<topnn.size()<<std::endl;
        std::cout << "\trecall@R (Probably not correct): " << getRecallAtR(answers.labels, topnn, args.at<int>("k")) << std::endl;
        std::cout << "\tMAP: " << getMeanAveragePrecision(answers.labels, topnn, args.at<int>("k")) << std::endl;
        char res[255];
        // cout<<"Reach -1"<<endl;
        strcpy(res, "results/");
        // cout<<"Reach -2"<<endl;
        strcat(res,args["test-id"].c_str());
        // cout<<"Reach -3"<<endl;
        strcat(res,"-");
        // cout<<"Reach -4"<<endl;
        strcat(res,std::to_string(refine).c_str());
        // cout<<"Reach -5"<<endl;
        strcat(res, ".ivecs");
        if (args.at<int>("storeResults") == 1) {
          save_to_fvecsV1<int>(res, answers.labels, args.at<int>("queries-size"), args.at<int>("k"));
        }
        // save_to_fvecsV1<int>(res, answers.labels, args.at<int>("queries-size"), args.at<int>("k"));
      }
    }
    datasetrefine.resize(0, 0);
  }

  return 0;
}
