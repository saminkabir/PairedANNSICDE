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
#include "BitVecEngine.hpp"
using namespace std;
using namespace std::chrono;

double myrand(std::string datasetName) {
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
    {"test-id", 's', ""},
    {"dataset", 's', ""},
    {"queries", 's', ""},
    {"groundtruth", 's', ""},
    {"groundtruth-format", 's', "ascii"},
    {"file-format-ori", 's', "fvecs"},
    {"result", 's', ""},
    {"timeseries-size", 'i', "1"},
    {"dataset-size", 'i', "0"},
    {"queries-size", 'i', "0"},
    {"k", 'i', "10"},
    {"cumVar", 'f', "0.95"},
    {"visit-cluster", 'f', "1"},
    {"new-alg", 'i', "0"},
    {"sub-space-swapping", 'i', "1"},
    {"kmeans-sub-space", 'i', "0"},
    {"nonUniformSubspaces", 'i', "0"},
    {"fixedSamplingForKmeans", 'i', "0"},
    {"minDimsPerSubspace", 'i', "2"},
    {"bitbudget", 'i', "256"},
    {"m", 'i', "32"},
    {"balanced", 'i', "0"},
    {"sampleSize", 'i', "10000"},
    {"no_hyper_peram", 'i', "5"}
  };
  ArgsParse args = ArgsParse(argc, argv, long_options, "HELP");
  char fn[255];
  strcpy(fn, "finalTests2/vaq-log-");
  strcat(fn,args["test-id"].c_str());
  strcat(fn, ".txt");
  freopen(fn,"w+",stdout);
  args.printArgs();
  int k=args.at<int>("k");;
  if (!isFileExists(args["dataset"]) || !isFileExists(args["queries"])) {
    std::cerr << "Dataset or queries file doesn't exists" << std::endl;
    return 1;
  }
  RowMatrixXf dataset = RowMatrixXf::Zero(args.at<int>("dataset-size"), args.at<int>("timeseries-size"));
  std:string datasetName=args["dataset"];
  dataset = fillWithRandomNumber(dataset, datasetName);
  std::cout << "Read dataset" << std::endl;
  if (args["file-format-ori"] == "fvecs") {
    readFVecsFromExternal(args["dataset"], dataset, args.at<int>("timeseries-size"), args.at<int>("dataset-size"));
  }
  RowMatrixXf queries = RowMatrixXf::Zero(args.at<int>("queries-size"), args.at<int>("timeseries-size"));
  queries = fillWithRandomNumber(queries, datasetName);

  std::cout << "Read queries" << std::endl;
  if (args["file-format-ori"] == "fvecs") {
    readFVecsFromExternal(args["queries"], queries, args.at<int>("timeseries-size"), args.at<int>("queries-size"));
  }

  std::vector<int> perm(dataset.rows());
  randomPermutation(perm);
  RowMatrixXf trainDataset;
  RowMatrixXf sampleQueries;
  int sampleSize=min(args.at<int>("sampleSize"),(int) dataset.rows());
  std::cout<<"Sample size: "<<sampleSize<<endl;
  trainDataset.resize(sampleSize, dataset.cols());
  RowMatrixXf trainDatasetClone;
  for (int i=0; i<sampleSize; i++) {
    trainDataset.row(i) = dataset.row(perm[i]);
  }
  sampleQueries.resize(100, queries.cols());
  for (int i=0; i<100; i++) {
    sampleQueries.row(i) = queries.row(i);
  }

  BitVecEngine engine(0);
  std::vector<std::vector<IdxDistPairFloat>> pairsFloat;
  pairsFloat = engine.queryNaiveEigen(trainDataset, sampleQueries, k);
  std::vector<std::vector<int>> topnn;
  topnn.resize(queries.rows(), std::vector<int>(k));
  for (int row=0; row<(int)pairsFloat.size(); row++) {
    for (int col=0; col<(int)pairsFloat[row].size(); col++)  {
      topnn[row][col] = pairsFloat[row][col].idx;
    }
  }

  double bestPrecision = 0;
  int bestMaxBit=0;

  bool cached_eigen_no_reduction=false;
  bool cached_eigen_95_reduction=false;
  Eigen::EigenSolver<RowMatrixXf> cached_es_no_reduction;
  Eigen::EigenSolver<RowMatrixXf> cached_es_95_reduction;
  int best_alg=22;
  float total_construction_time=0.0f;
  float total_search_time=0.0f;
  auto evaluateParams = [&args, &topnn, &dataset, &trainDataset, &queries, &sampleQueries, &sampleSize, &trainDatasetClone, &cached_eigen_no_reduction, &cached_eigen_95_reduction, &cached_es_no_reduction, &cached_es_95_reduction, &best_alg, &total_construction_time, &total_search_time, &k](
    const int alg,const int minBits, const int maxBits, const float reductionThreshold, float visited_clusters=-1.0f, int KmeansSubspace=0) {
    int _k=10;
    VAQ vaq;
    vaq.mBitBudget = args.at<int>("bitbudget");
    vaq.mSubspaceNum = vaq.mBitBudget/8;
    vaq.mPercentVarExplained = 1;
    vaq.mMinBitsPerSubs = minBits;
    vaq.mMaxBitsPerSubs = maxBits;
    vaq.mMethods = VAQ::NNMethod::EA;
    if (visited_clusters>0) {
      cout<<"Reach here"<<endl;
      vaq.mMethods |= VAQ::NNMethod::TI;
      vaq.mTIClusterNum=100;
      vaq.mVisit=visited_clusters;
    }
    if (KmeansSubspace==1) {
      vaq.kmeans_sub_space=1;
    }
    vaq.new_alg=alg;
    vaq.cumVar=reductionThreshold;
    vaq.minDimsPerSubspace=2;
    vaq.nonUniformSubspaces=true;
    vaq.fixedSamplingForKmeans=true;
    if (alg==0) {
      vaq.cumVar=1.0f;
    }


    
    trainDatasetClone.resize(sampleSize, dataset.cols());
    for (int i=0; i<sampleSize; i++) {
      trainDatasetClone.row(i) = trainDataset.row(i);
    } 


    // encoding phase
    // if (alg==0 && cached_eigen_no_reduction) {
    //   vaq.cached_es=cached_es_no_reduction;
    //   vaq.eigenCached=true;
    // }
    // if (alg==22 && cached_eigen_95_reduction) {
    //   vaq.cached_es=cached_es_95_reduction;
    //   vaq.eigenCached=true;
    // }
    auto start_time_train = high_resolution_clock::now();
    try {
      vaq.train(trainDatasetClone, true);
    }
    catch (const exception& e) {
      cout << "Exception " << e.what() << endl;
      auto stop_time_train_early = high_resolution_clock::now();
      auto training_time = duration_cast<microseconds>(stop_time_train_early - start_time_train);
      total_construction_time+=training_time.count()/1000000.0f;
      return -1.00;
    }
    
    auto stop_time_train = high_resolution_clock::now();
    auto training_time = duration_cast<microseconds>(stop_time_train - start_time_train);
    total_construction_time+=training_time.count()/1000000.0f;
    cout << "Training time: "<< training_time.count()/1000000.0f << " microseconds" << endl;
    // std::cout<<"Alg test: "<<alg<<" "<<reductionThreshold<<" "<<endl;
    // if (alg==0) {
    //   cached_es_no_reduction=vaq.cached_es;
    //   cached_eigen_no_reduction=true;
    // }
    // else if (alg==22 && reductionThreshold==0.95f) {
    //   std::cout<<"stored eigen"<<endl;
    //   cached_es_95_reduction=vaq.cached_es;
    //   cached_eigen_95_reduction=true;
    // }
    auto start_time_encode = high_resolution_clock::now();
    vaq.encode(trainDatasetClone);
    auto stop_time_encode = high_resolution_clock::now();
    auto encoding_time = duration_cast<microseconds>(stop_time_encode - start_time_encode);
    total_construction_time+=encoding_time.count()/1000000.0f;
    if (visited_clusters>0) {
      cout<<"Reach here 2: "<<vaq.searchMethod()<<endl;
      if (vaq.searchMethod() & VAQ::NNMethod::TI) {
        START_TIMING(TI_CLUSTER);
        vaq.clusterTI(true, true);
        END_TIMING(TI_CLUSTER, "== TI Clustering time: ");
      }
    }

    cputime_t start = timeNow();
    auto start_time_search = high_resolution_clock::now();
    LabelDistVecF answers;
    answers = vaq.search(sampleQueries, k);
    auto stop_time_search = high_resolution_clock::now();
    auto search_time = duration_cast<microseconds>(stop_time_search - start_time_search);
    total_search_time+=search_time.count()/1000000.0f;
    double precision = getAvgRecall(answers.labels, topnn, k, k);
    std::cout << "Precision: " << precision << std::endl;
    return precision;
  };
  int best_min_bit, best_max_bit; 
  {
    int initial_min_bit=5;
    int initial_max_bit=12;
    best_min_bit=initial_min_bit;
    best_max_bit=initial_max_bit;
    std::vector<int> allMaxBit, allMinBit;
    int minBit=initial_min_bit;
    int maxBit=initial_max_bit;
    float reduction_threshold=0.95;
    float bestPrecisionTest=0.0f;
    int algo=0;
    int kmeans_sub_space=0;
    float precision=evaluateParams(algo,minBit, maxBit, 1);
    std::cout<<"algo: "<<algo<<" minBit: "<<minBit<<" maxBit: "<<maxBit<<" Precision: "<<precision<<endl;
    if (precision>=bestPrecisionTest) {
      best_alg=algo;
      bestPrecisionTest=precision;
    }
    algo=22;
    float test_reduction_threshold=0.9;
    precision=evaluateParams(algo,minBit, maxBit, test_reduction_threshold);
    std::cout<<"algo: "<<algo<<" reduction_threshold: "<<test_reduction_threshold<<" minBit: "<<minBit<<" maxBit: "<<maxBit<<" Precision: "<<precision<<endl;
    if (precision>=bestPrecisionTest) {
      best_alg=algo;
      reduction_threshold=test_reduction_threshold;
      bestPrecisionTest=precision;
    }
    test_reduction_threshold=0.92;
    precision=evaluateParams(algo,minBit, maxBit, test_reduction_threshold);
    std::cout<<"algo: "<<algo<<" reduction_threshold: "<<test_reduction_threshold<<" minBit: "<<minBit<<" maxBit: "<<maxBit<<" Precision: "<<precision<<endl;
    if (precision>=bestPrecisionTest) {
      best_alg=algo;
      reduction_threshold=test_reduction_threshold;
      bestPrecisionTest=precision;
    }
    test_reduction_threshold=0.91;
    precision=evaluateParams(algo,minBit, maxBit, test_reduction_threshold);
    std::cout<<"algo: "<<algo<<" reduction_threshold: "<<test_reduction_threshold<<" minBit: "<<minBit<<" maxBit: "<<maxBit<<" Precision: "<<precision<<endl;
    if (precision>=bestPrecisionTest) {
      best_alg=algo;
      reduction_threshold=test_reduction_threshold;
      bestPrecisionTest=precision;
    }
    test_reduction_threshold=0.93;
    precision=evaluateParams(algo,minBit, maxBit, test_reduction_threshold);
    std::cout<<"algo: "<<algo<<" reduction_threshold: "<<test_reduction_threshold<<" minBit: "<<minBit<<" maxBit: "<<maxBit<<" Precision: "<<precision<<endl;
    if (precision>=bestPrecisionTest) {
      best_alg=algo;
      reduction_threshold=test_reduction_threshold;
      bestPrecisionTest=precision;
    }
    test_reduction_threshold=0.94;
    precision=evaluateParams(algo,minBit, maxBit, test_reduction_threshold);
    std::cout<<"algo: "<<algo<<" reduction_threshold: "<<test_reduction_threshold<<" minBit: "<<minBit<<" maxBit: "<<maxBit<<" Precision: "<<precision<<endl;
    if (precision>=bestPrecisionTest) {
      best_alg=algo;
      reduction_threshold=test_reduction_threshold;
      bestPrecisionTest=precision;
    }
    test_reduction_threshold=0.96;
    precision=evaluateParams(algo,minBit, maxBit, test_reduction_threshold);
    std::cout<<"algo: "<<algo<<" reduction_threshold: "<<test_reduction_threshold<<" minBit: "<<minBit<<" maxBit: "<<maxBit<<" Precision: "<<precision<<endl;
    if (precision>=bestPrecisionTest) {
      best_alg=algo;
      reduction_threshold=test_reduction_threshold;
      bestPrecisionTest=precision;
    }
    test_reduction_threshold=0.97;
    precision=evaluateParams(algo,minBit, maxBit, test_reduction_threshold);
    std::cout<<"algo: "<<algo<<" reduction_threshold: "<<test_reduction_threshold<<" minBit: "<<minBit<<" maxBit: "<<maxBit<<" Precision: "<<precision<<endl;
    if (precision>=bestPrecisionTest) {
      best_alg=algo;
      reduction_threshold=test_reduction_threshold;
      bestPrecisionTest=precision;
    }
    test_reduction_threshold=0.98;
    precision=evaluateParams(algo,minBit, maxBit, test_reduction_threshold);
    std::cout<<"algo: "<<algo<<" reduction_threshold: "<<test_reduction_threshold<<" minBit: "<<minBit<<" maxBit: "<<maxBit<<" Precision: "<<precision<<endl;
    if (precision>=bestPrecisionTest) {
      best_alg=algo;
      reduction_threshold=test_reduction_threshold;
      bestPrecisionTest=precision;
    }
    test_reduction_threshold=0.99;
    precision=evaluateParams(algo,minBit, maxBit, test_reduction_threshold);
    std::cout<<"algo: "<<algo<<" reduction_threshold: "<<test_reduction_threshold<<" minBit: "<<minBit<<" maxBit: "<<maxBit<<" Precision: "<<precision<<endl;
    if (precision>=bestPrecisionTest) {
      best_alg=algo;
      reduction_threshold=test_reduction_threshold;
      bestPrecisionTest=precision;
    }
    test_reduction_threshold=0.95;
    precision=evaluateParams(algo,minBit, maxBit, test_reduction_threshold);
    std::cout<<"algo: "<<algo<<" reduction_threshold: "<<test_reduction_threshold<<" minBit: "<<minBit<<" maxBit: "<<maxBit<<" Precision: "<<precision<<endl;
    if (precision>=bestPrecisionTest) {
      best_alg=algo;
      reduction_threshold=test_reduction_threshold;
      bestPrecisionTest=precision;
    }
    float precision_uniform=evaluateParams(algo,minBit, maxBit, test_reduction_threshold,-1.0f,0);
    float precision_non_uniform=evaluateParams(algo,minBit, maxBit, test_reduction_threshold,-1.0f,1);
    if (precision_non_uniform>precision_uniform) {
      kmeans_sub_space=1;
    }
    else {
      kmeans_sub_space=0;
    }
    std::cout<<"Uniform recall: "<<precision_uniform<<" Non-uniform recall: "<<precision_non_uniform<<endl;
    if (best_alg==22) {
      allMaxBit = {13,13,8,9,12,11,13};
      allMinBit = {2 , 6,7,5, 7, 6, 5};
    }
    else {
      allMaxBit = {8,12,13,12,9,12};
      allMinBit = {7, 2, 5, 7,5, 1};
    }
    bestPrecision=bestPrecisionTest;
    for (int i=0;i<allMaxBit.size();i++) {
      minBit=allMinBit[i];
      maxBit=allMaxBit[i];
      precision=evaluateParams(best_alg,minBit, maxBit, reduction_threshold, -1.0f, kmeans_sub_space);
      std::cout<<"algo: "<<best_alg<<" reduction_threshold: "<<reduction_threshold<<" minBit: "<<minBit<<" maxBit: "<<maxBit<<" kmeans_sub_space: "<<kmeans_sub_space<<" Precision: "<<precision<<endl;
      if (precision>bestPrecision) {
        best_min_bit=minBit;
        best_max_bit=maxBit;
        bestPrecision=precision;
      }
    }

    cout<<"Total construction time: "<<total_construction_time<<", Total search time: "<<total_search_time<<", Best Alg: "<<best_alg<<", Reduction Threshold: "<<reduction_threshold<<" kmeans_sub_space: "<<kmeans_sub_space<<", Best Precision: "<<bestPrecision<<", Best min: "<<best_min_bit<<", Best max: "<<best_max_bit;
    std::vector<float> allVCParam = {0.5f,0.25f,0.225f,0.2f,0.175f,0.15f};
    for (int i=0;i<allVCParam.size();i++) {
      float vc=allVCParam[i];
      precision=evaluateParams(best_alg,best_min_bit, best_max_bit, reduction_threshold, vc, kmeans_sub_space);
      cout<<"Visited clusters: "<<vc<<" algo: "<<best_alg<<" Precision: "<<precision<<endl;
    }
    // precision=evaluateParams(best_alg,best_min_bit, best_max_bit, reduction_threshold, 0.25);
    // cout<<"Visited clusters: "<<0.25<<" "<<precision<<endl;
    // precision=evaluateParams(best_alg,best_min_bit, best_max_bit, reduction_threshold, 0.5);
    // cout<<"Visited clusters: "<<0.5<<" "<<precision<<endl;
    // precision=evaluateParams(best_alg,best_min_bit, best_max_bit, reduction_threshold, 0.75);
    // cout<<"Visited clusters: "<<0.75<<" "<<precision<<endl;
    cout<<"Final construction time: "<<total_construction_time<<", Final search time: "<<total_search_time<<endl;
  }

  return 0;
}
