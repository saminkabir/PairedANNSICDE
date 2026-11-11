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
    {"file-format-ori", 's', "fvecs"},
    {"groundtruth", 's', ""},
    {"groundtruth-format", 's', "ascii"},
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

  std::vector<std::vector<int>> topnnOrig;
  if (args["groundtruth"] != "") {
    std::cout << "Read groundtruth" << std::endl;
    if (args["groundtruth-format"] == "ascii") {
      readTOPNNExternal(args["groundtruth"], topnnOrig, args.at<int>("k"), ',');
    } else if (args["groundtruth-format"] == "ivecs") {
      readIVecsFromExternal(args["groundtruth"], topnnOrig, args.at<int>("k"));
    } else if (args["groundtruth-format"] == "bin") {
      readTOPNNExternalBin(args["groundtruth"], topnnOrig, args.at<int>("k"));
    }
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
  topnn.resize(sampleQueries.rows(), std::vector<int>(k));
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
  auto evaluateParams = [&args, &topnn, &dataset, &trainDataset, &queries, &sampleQueries, &sampleSize, &trainDatasetClone, &cached_eigen_no_reduction, &cached_eigen_95_reduction, &cached_es_no_reduction, &cached_es_95_reduction, &best_alg, &total_construction_time, &total_search_time, &k, &topnnOrig](
    const int alg,const int minBits, const int maxBits, const float reductionThreshold, float visited_clusters=-1.0f, int KmeansSubspace=0, bool tuning=true) {
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
      vaq.mTIClusterNum=1000;
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


    if (tuning) {
      trainDatasetClone.resize(sampleSize, dataset.cols());
      for (int i=0; i<sampleSize; i++) {
        trainDatasetClone.row(i) = trainDataset.row(i);
      } 
    }
    else {
      trainDatasetClone.resize(dataset.rows(), dataset.cols());
      for (int i=0; i<sampleSize; i++) {
        trainDatasetClone.row(i) = dataset.row(i);
      }
    }
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
    if (tuning) {
      answers = vaq.search(sampleQueries, k);
    }
    else {
      answers = vaq.search(queries, k);
    }
    
    auto stop_time_search = high_resolution_clock::now();
    auto search_time = duration_cast<microseconds>(stop_time_search - start_time_search);
    total_search_time+=search_time.count()/1000000.0f;
    double precision = 0;
    if (tuning) {
      precision=getAvgRecall(answers.labels, topnn, k, k);
    }
    else {
      precision=getAvgRecall(answers.labels, topnnOrig, k, k);
    }
    
    std::cout << "Query time: " << search_time.count()/1000000.0f << std::endl;
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
    cout<<"Phase 1 starting"<<endl;
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

    cout<<"Phase 2 starting"<<endl;
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
    // for (int i=0;i<allMaxBit.size();i++) {
    //   minBit=allMinBit[i];
    //   maxBit=allMaxBit[i];
    //   precision=evaluateParams(best_alg,minBit, maxBit, reduction_threshold, -1.0f, kmeans_sub_space);
    //   std::cout<<"algo: "<<best_alg<<" reduction_threshold: "<<reduction_threshold<<" minBit: "<<minBit<<" maxBit: "<<maxBit<<" kmeans_sub_space: "<<kmeans_sub_space<<" Precision: "<<precision<<endl;
    //   if (precision>bestPrecision) {
    //     best_min_bit=minBit;
    //     best_max_bit=maxBit;
    //     bestPrecision=precision;
    //   }
    // }

    cout<<"Phase 3 starting"<<endl;
    for (maxBit=8;maxBit<14;maxBit++) {
      int maxMinBit=7;
      if (maxBit==8) {
        maxMinBit=1;
      }
      
      for (minBit=1;minBit<=maxMinBit;minBit++) {
        precision=evaluateParams(best_alg,minBit, maxBit, reduction_threshold, -1.0f, kmeans_sub_space);
        std::cout<<"algo: "<<best_alg<<" reduction_threshold: "<<reduction_threshold<<" minBit: "<<minBit<<" maxBit: "<<maxBit<<" kmeans_sub_space: "<<kmeans_sub_space<<" Precision: "<<precision<<endl;
        if (precision>bestPrecision) {
          best_min_bit=minBit;
          best_max_bit=maxBit;
          bestPrecision=precision;
        }
      }
      
    }

    cout<<"Phase 4 starting"<<endl;
    std::vector<float> allRTParam = {0.9f,0.91f,0.92f,0.94f,0.94f,0.95f,0.96f,0.97f,0.98f,0.99f};
    for (int i=0;i<allRTParam.size();i++) {
      test_reduction_threshold=allRTParam[i];
      precision=evaluateParams(22,best_min_bit, best_max_bit, test_reduction_threshold, -1.0f, kmeans_sub_space);
      std::cout<<"algo: "<<22<<" reduction_threshold: "<<test_reduction_threshold<<" minBit: "<<best_min_bit<<" maxBit: "<<best_max_bit<<" kmeans_sub_space: "<<kmeans_sub_space<<" Precision: "<<precision<<endl;
      if (precision>bestPrecision) {
        best_alg=22;
        reduction_threshold=test_reduction_threshold;
      }
    }
    precision=evaluateParams(0,best_min_bit, best_max_bit, reduction_threshold, -1.0f, kmeans_sub_space);
    std::cout<<"algo: "<<0<<" reduction_threshold: "<<test_reduction_threshold<<" minBit: "<<best_min_bit<<" maxBit: "<<best_max_bit<<" kmeans_sub_space: "<<kmeans_sub_space<<" Precision: "<<precision<<endl;
    if (precision>bestPrecision) {
      best_alg=0;
    }

    cout<<"Phase 5 starting"<<endl;
    precision=evaluateParams(best_alg,best_min_bit, best_max_bit, reduction_threshold, -1.0f, 0);
    std::cout<<"algo: "<<best_alg<<" reduction_threshold: "<<test_reduction_threshold<<" minBit: "<<best_min_bit<<" maxBit: "<<best_max_bit<<" kmeans_sub_space: "<<0<<" Precision: "<<precision<<endl;
    if (precision>bestPrecision) {
      kmeans_sub_space=0;
    }

    precision=evaluateParams(best_alg,best_min_bit, best_max_bit, reduction_threshold, -1.0f, 1);
    std::cout<<"algo: "<<best_alg<<" reduction_threshold: "<<test_reduction_threshold<<" minBit: "<<best_min_bit<<" maxBit: "<<best_max_bit<<" kmeans_sub_space: "<<1<<" Precision: "<<precision<<endl;
    if (precision>bestPrecision) {
      kmeans_sub_space=1;
    }
    cout<<"Total construction time: "<<total_construction_time<<", Total search time: "<<total_search_time<<", Best Alg: "<<best_alg<<", Reduction Threshold: "<<reduction_threshold<<" kmeans_sub_space: "<<kmeans_sub_space<<", Best Precision: "<<bestPrecision<<", Best min: "<<best_min_bit<<", Best max: "<<best_max_bit;

    cout<<"Phase 6 starting"<<endl;
    std::vector<float> allVCParam = {0.75f,0.5f,0.25f,0.2f,0.15f,0.1f,0.05f};
    float best_vc=0.5f;
    float best_vc_recall=0;
    for (int i=0;i<allVCParam.size();i++) {
      float vc=allVCParam[i];
      precision=evaluateParams(best_alg,best_min_bit, best_max_bit, reduction_threshold, vc, kmeans_sub_space);
      if (precision>=best_vc_recall) {
        best_vc_recall=precision;
      }
      cout<<"Visited clusters: "<<vc<<" algo: "<<best_alg<<" Precision: "<<precision<<endl;
      if (best_vc_recall-precision>=0.005f) {
        break;
      }
      best_vc=vc;

    }
    allVCParam={};
    cout<<"Phase 7 starting"<<endl;
    if (best_vc==0.75f) {
      allVCParam = {0.7f,0.65f,0.6f,0.55f};
    }
    if (best_vc==0.5f) {
      allVCParam = {0.45f,0.4f,0.35f,0.3f};
    }
    if (best_vc==0.25f) {
      allVCParam = {0.24f,0.23f,0.22f,0.21f};
    }
    if (best_vc==0.2f) {
      allVCParam = {0.19f,0.18f,0.17f,0.16f};
    }
    if (best_vc==0.15f) {
      allVCParam = {0.14f,0.13f,0.12f,0.11f};
    }
    if (best_vc==0.1f) {
      allVCParam = {0.9f,0.8f,0.7f,0.6f};
    }

    for (int i=0;i<allVCParam.size();i++) {
      float vc=allVCParam[i];
      precision=evaluateParams(best_alg,best_min_bit, best_max_bit, reduction_threshold, vc, kmeans_sub_space);
      cout<<"Visited clusters: "<<vc<<" algo: "<<best_alg<<" Precision: "<<precision<<endl;
      if (precision>=best_vc_recall) {
        best_vc_recall=precision;
      }
      if (best_vc_recall-precision>=0.005f) {
        break;
      }
      best_vc=vc;

    }
    cout<<"Final construction time: "<<total_construction_time<<", Final search time: "<<total_search_time<<endl;

    cout<<"Final phase starting"<<endl;
    precision=evaluateParams(best_alg,best_min_bit, best_max_bit, reduction_threshold, best_vc, kmeans_sub_space, false);
    cout<<"Original recall: "<<precision<<endl;
  }

  return 0;
}
