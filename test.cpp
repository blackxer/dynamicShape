#include "crowdcount.h"
using namespace sample;
int main(int argc, char** argv)
{
    int INPUT_C = 3;
    int INPUT_H = 512;
    int INPUT_W = 512;//1104;

    int OUTPUT_C = 1;
    int OUTPUT_H = INPUT_H / 8;
    int OUTPUT_W = INPUT_W / 8;

    CrowdCount crowdCount(INPUT_C, INPUT_H, INPUT_W, OUTPUT_C, OUTPUT_H, OUTPUT_W, 4, 1);
    const char* onnxModelName = "../model/model_dynamic.onnx";//"../src/crowdcount/model/model.onnx";
    std::string trtModelName = "../model/AlgCrowdCount.giemodel";//"./AlgCrowdCount.giemodel";
    crowdCount.onnx2trt(onnxModelName, trtModelName);
    crowdCount.initEngine(trtModelName);

//    float data[INPUT_C*INPUT_H*INPUT_W];
    //"/media/zw/DL/ly/workspace/project10/Bayesian-Crowd-Counting-master/datasets/ShanghaiTech_Crowd_Counting_Dataset/part_A_final_processed/test/IMG_11.jpg"
    std::string  filename = "../model/IMG_11.jpg";//"../src/crowdcount/model/IMG_11.jpg";
//    crowdCount.preprocess(filename, data);
    cv::Mat img = cv::imread(filename);
//    crowdCount.preprocess(img, data);
    std::vector<cv::Mat> input;
    std::vector<float> outputs;
    for(int i=0; i<4; i++){
        input.push_back(img.clone());
    }

    // run inference
//    float scores[OUTPUT_C*OUTPUT_H*OUTPUT_W];
    auto start_time = std::chrono::system_clock::now();
    for(int i=0; i<1; i++){
//        crowdCount.doInference(data, scores, 1);
        crowdCount.process(input, outputs);
    }
    auto end_time = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time).count()/100.0 << "ms" << std::endl;
//    float people_count = 0;
//    for(int i=0;i<OUTPUT_C*OUTPUT_H*OUTPUT_W;i++){
//        people_count+=scores[i];
//    }
//    std::cout << "finished" << people_count << std::endl;
    for(int i=0; i<outputs.size(); i++){
        std::cout << outputs[i] << " ";
    }

    return 0;
}

// 282.958 282.958 282.958 282.95
