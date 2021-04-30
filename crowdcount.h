#ifndef CROWDCOUNT_H
#define CROWDCOUNT_H

#include <unistd.h>
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <time.h>
#include <vector>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "logger.h"
#include "common.h"


#include <opencv2/opencv.hpp>

using namespace nvinfer1;

class CrowdCount{
public:
    CrowdCount(const int input_c, const int input_h, const int input_w, const int output_c, const int output_h, const int output_w,
               const int maxBatchSize, const int batchSize);
    bool onnx2trt(const std::string onnxModelName, const std::string trtModelName);
    void initEngine(const std::string trtModelName);
    bool doInference(const float* input, float* output, const int batchSize);
    void preprocess(const std::string& fileName, float* preInput);
    void preprocess(const cv::Mat& img, float* preInput);
    void process(const std::vector<cv::Mat>& input, std::vector<float> &outputs);

    ~CrowdCount();


private:
    int input_c_;
    int input_h_;
    int input_w_;

    int output_c_;
    int output_h_;
    int output_w_;

    int batchSize_;
    int maxBatchSize_;

    IRuntime* runtime_;
    ICudaEngine* engine_;
    IExecutionContext* context_;
};

#endif








