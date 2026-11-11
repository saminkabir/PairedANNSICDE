#include "Metric.h"
#include<algorithm>


Metric::Metric()
{
}


Metric::~Metric()
{
}


vector<DATATYPE> Metric::calRecallAll(Real_Result& real_result_,vector <vector<pair<DATATYPE, int>>>& query_res_) {

	int a[] = { 1,10,20,30,40,50,60,70,80,90,100 };
	vector<int> knn_recall(a, a + sizeof(a) / sizeof(a[0]));
	int maxKNN = a[sizeof(a) / sizeof(a[0]) - 1];
	vector<DATATYPE> res;

	for (int c = 0; c < knn_recall.size(); ++c)
	{
		DATATYPE overRecall = 0;
		int target_knn = knn_recall[c];
		overRecall = Cal_Recall(target_knn, real_result_,query_res_);
		res.emplace_back(overRecall);
	}
	return res;
}

DATATYPE Metric::Cal_Recall(int targetKnn_,Real_Result & real_result_, vector<vector<pair<DATATYPE, int>>> query_res_)
{
	DATATYPE overRecall = 0;
	for (int i = 0; i < query_res_.size(); ++i)
	{
		DATATYPE recall = 0;
		for (int j = 0; j < targetKnn_; ++j)
		{
			for (int c = 0; c < targetKnn_; ++c)
			{
				if (real_result_[i][j].second == query_res_[i][c].second)
				{
					recall++;
					break;
				}
			}
		}
		overRecall += recall;
	}
	overRecall = overRecall / query_res_.size() / targetKnn_;
	return overRecall;
}

DATATYPE Metric::Cal_MAP(int targetKnn_,Real_Result & real_result_, vector<vector<pair<DATATYPE, int>>> query_res_)
{
	float map = 0;
	for (int nq = 0; nq < query_res_.size(); ++nq)
	{
		float ap = 0;
    	for (int r=1; r<=targetKnn_; r++) {
			bool isR_kExact = false;
			for (int j=0; j<targetKnn_; j++) {
				if (query_res_[nq][r-1].second == real_result_[nq][j].second) {
					isR_kExact = true;
					break;
				}
			}
			if (isR_kExact) {
				int ct = 0;
				for (int j=0; j<r; j++) {
					for (int jj=0; jj<r; jj++) {
						if (query_res_[nq][j].second == real_result_[nq][jj].second) {
							ct++;
							break;
						}
					}
				}
				ap += (double)ct/r;
			}
    	}
		map+=ap/targetKnn_;
	}
	
	return map/(float) query_res_.size();
}


vector<DATATYPE> Metric::calOverRatioAll(Real_Result& real_result_, vector <vector<pair<DATATYPE, int>>>& query_res_) 
{
	int a[] = { 1,10,20,30,40,50,60,70,80,90,100 };
	vector<int> knn_Ratio(a, a + sizeof(a) / sizeof(a[0]));
	int maxKNN = a[sizeof(a) / sizeof(a[0]) - 1];
	vector<DATATYPE> res;
	for (int c = 0; c < knn_Ratio.size(); ++c) {
		DATATYPE overRatio = 0;
		int target_knn = knn_Ratio[c];
		overRatio = Cal_Ratio(target_knn, real_result_, query_res_);
		res.emplace_back(overRatio);
	}

	return res;
}

DATATYPE Metric::Cal_Ratio(int target_knn_, Real_Result & real_result_, vector<vector<pair<DATATYPE, int>>> query_res_)
{

	DATATYPE overRatio = 0;
	for (int i = 0; i < query_res_.size(); ++i)
	{
		DATATYPE ratio = 0;
		int count = min(target_knn_, (int)query_res_[i].size());
		for (int j = 0; j < count; ++j)
		{
			if (real_result_[i][j].first == 0) {
				ratio += 1.0f;
			}
			else {
				ratio += query_res_[i][j].first / real_result_[i][j].first;
			}
			
		}
		if (query_res_[i].size() != 0)
			overRatio += ratio / count;
		else
			overRatio += 1;
	}
	overRatio = overRatio / query_res_.size();
	return overRatio;
}

DATATYPE Metric::calRecall(vector<pair<DATATYPE, int>>& realResult, vector<pair<DATATYPE, int>>& approResult)
{
	if (realResult.size() == 0)
		return 1.0;
	int count = 0;
	for (int i = 0; i < approResult.size(); ++i) {
		for (int j = 0; j < realResult.size();++j) {
			if (approResult[i].second == realResult[j].second)
				count++;
		}
	}
	return count/ approResult.size();
}







