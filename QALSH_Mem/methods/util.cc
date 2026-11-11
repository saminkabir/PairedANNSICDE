#include "util.h"

namespace nns {

timeval g_start_time;               // global param: start time
timeval g_end_time;                 // global param: end time

float g_indexing_time = -1.0f;      // global param: indexing time
float g_estimated_mem = -1.0f;      // global param: estimated memory
float g_runtime       = -1.0f;      // global param: running time
float g_ratio         = -1.0f;      // global param: overall ratio
float g_recall        = -1.0f;      // global param: recall

// -----------------------------------------------------------------------------
void create_dir(                    // create dir if the path exists
    char *path)                         // input path
{
    int len = (int) strlen(path);
    for (int i = 0; i < len; ++i) {
        if (path[i] != '/') continue;
        
        char ch = path[i+1]; path[i+1] = '\0';
        if (access(path, F_OK) != 0) {
            if (mkdir(path, 0755) != 0) {
                printf("Could not create %s\n", path); exit(1);
            }
        }
        path[i+1] = ch;
    }
}

// -----------------------------------------------------------------------------
int write_ground_truth(             // write ground truth to disk
    int   n,                            // number of ground truth results
    int   d,                            // dimension of ground truth results
    float p,                            // l_p distance
    const char *prefix,                 // prefix of truth set
    const Result *truth)                // ground truth
{
    char fname[200]; sprintf(fname, "%s.gt%3.1f", prefix, p);
    FILE *fp = fopen(fname, "wb");
    if (!fp) { printf("Could not create %s\n", fname); return 1; }
    
    uint64_t size = (uint64_t) n*d;
    fwrite(truth, sizeof(Result), size, fp);
    fclose(fp);
    return 0;
}

// -----------------------------------------------------------------------------
float calc_ratio(                   // calc overall ratio [1,\infinity)
    int   k,                            // top-k value
    const Result *truth,                // ground truth results 
    MinK_List *list)                    // top-k approximate results
{
    float ratio = 0.0f;
    for (int i = 0; i < k; ++i) {
        ratio += list->ith_key(i) / truth[i].key_;
    }
    return ratio / k;
}

// -----------------------------------------------------------------------------
float calc_recall(                  // calc recall (percentage)
    int   k,                            // top-k value
    const Result *truth,                // ground truth results 
    MinK_List *list)                    // top-k approximate results
{
    int i = k - 1;
    int last = k - 1;
    while (i >= 0 && list->ith_key(i) > truth[last].key_) {
        i--;
    }
    return (i + 1) * 100.0f / k;
}

float calc_map(                  // calc recall (percentage)
    int   K,                            // top-k value
    const Result *truth,                // ground truth results 
    MinK_List *list)                    // top-k approximate results
{
    float ap = 0;
    for (int r=1; r<=K; r++) {
        bool isR_kExact = false;
        for (int j=0; j<K; j++) {
            if (list->ith_id(r-1) == truth[j].id_-1) {
                isR_kExact = true;
                break;
            }
        }
        if (isR_kExact) {
            int ct = 0;
            for (int j=0; j<r; j++) {
                for (int jj=0; jj<r; jj++) {
                    if (list->ith_id(j) == truth[jj].id_-1) {
                        ct++;
                        break;
                    }
                }
            }
            ap += (double)ct/r;
        }
    }
    return ap/K;
}

} // end namespace nns
