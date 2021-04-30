#include "crowdcount.h"
using namespace sample;
CrowdCount::CrowdCount(const int input_c, const int input_h, const int input_w, const int output_c, const int output_h, const int output_w,
                       const int maxBatchSize, const int batchSize){
    input_c_ = input_c;
    input_h_ = input_h;
    input_w_ = input_w;

    output_c_ = output_c;
    output_h_ = output_h;
    output_w_ = output_w;

    maxBatchSize_ = maxBatchSize;
    batchSize_ = batchSize;
    assert(maxBatchSize_ >= batchSize_);
}

CrowdCount::~CrowdCount(){
    if(context_){
        context_->destroy();
        engine_->destroy();
        runtime_->destroy();
    }
}

bool CrowdCount::onnx2trt(const std::string onnxModelName, const std::string trtModelName){

    // File existed, directly return
    if(access( trtModelName.c_str(), F_OK ) != -1){
        return true;
    }

    // create the builder
    IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
    assert(builder != nullptr);

    //nvinfer1::INetworkDefinition* network = builder->createNetwork();
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(explicitBatch);


    auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());

    if ( !parser->parseFromFile(onnxModelName.c_str(), static_cast<int>(gLogger.getReportableSeverity()) ) )
    {
       gLogError << "Failure while parsing ONNX file" << std::endl;
       return false;
    }

    auto input = network->getInput(0);
    auto config = builder->createBuilderConfig();
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions(input->getName(), OptProfileSelector::kMIN, Dims4{1, 3, 512, 512});
    profile->setDimensions(input->getName(), OptProfileSelector::kOPT, Dims4{1, 3, 1024, 1024});
    profile->setDimensions(input->getName(), OptProfileSelector::kMAX, Dims4{1, 3, 2048, 2048});
    config->addOptimizationProfile(profile);

    // Build the engine
//    builder->setMaxBatchSize(maxBatchSize_);
    config->setMaxWorkspaceSize(1 << 30);
    config->setFlag(BuilderFlag::kFP16);

    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    assert(engine);

    // we can destroy the parser
    parser->destroy();

    // serialize the engine, then close everything down
    IHostMemory *trtModelStream{nullptr};
    trtModelStream = engine->serialize();

    // 设置保存文件的名称为cached_model.bin
    std::ofstream serialize_output_stream;

    // 将序列化的模型结果拷贝至serialize_str字符串
    std::string serialize_str;
    serialize_str.resize( trtModelStream->size() );
    memcpy((void*)serialize_str.data(), trtModelStream->data(), trtModelStream->size());

    // 将serialize_str字符串的内容输出至cached_model.bin文件
    serialize_output_stream.open(trtModelName);
    serialize_output_stream << serialize_str;
    serialize_output_stream.close();

    engine->destroy();
    network->destroy();
    builder->destroy();

    return true;
}

void CrowdCount::initEngine(const std::string trtModelName){
    char *trtModelStream{nullptr};
    size_t size{0};
    std::ifstream file(trtModelName, std::ios::binary);
    if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
    }

    // deserialize the engine
    runtime_ = createInferRuntime(gLogger);
    assert(runtime_ != nullptr);

    engine_ = runtime_->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine_ != nullptr);
    delete[] trtModelStream;
    context_ = engine_->createExecutionContext();
    assert(context_ != nullptr);
}

bool CrowdCount::doInference(const float* input, float* output, const int batchSize){
    Dims4 inputDims{1, 3, input_h_, input_w_};
    // Set the input size for the preprocessor
    CHECK_RETURN_W_MSG(context_->setBindingDimensions(0, inputDims), false, "Invalid binding dimensions.");
    // We can only run inference once all dynamic input shapes have been specified.
    if (!context_->allInputDimensionsSpecified())
    {
        std::cout <<  "allInputDimensionsSpecified failed" << std::endl;
    }

    const ICudaEngine& engine = context_->getEngine();
    // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly one input and one output.
//    assert(batchSize==1);
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    //std::cout << engine.getBindingDimensions(0) << std::endl;


    // create GPU buffers and a stream
    CHECK(cudaMalloc(&buffers[0], batchSize * input_c_ * input_h_ * input_w_ * sizeof(float)));
    CHECK(cudaMalloc(&buffers[1], batchSize * output_c_ * output_h_ * output_w_ * sizeof(float)));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * input_c_ * input_h_ * input_w_* sizeof(float), cudaMemcpyHostToDevice, stream));
    context_->enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * output_c_ * output_h_ * output_w_* sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[0]));
    CHECK(cudaFree(buffers[1]));

    return true;
}

void CrowdCount::preprocess(const std::string& fileName, float* preInput){
    cv::Mat img = cv::imread(fileName);
    cv::cvtColor(img,img,cv::COLOR_BGR2RGB);

    cv::resize(img,img,cv::Size(input_w_, input_h_));

    assert(img.channels() == input_c_);
    assert(img.rows == input_h_);
    assert(img.cols == input_w_);

    int predHeight = img.rows;
    int predWidth = img.cols;
    int size = predHeight * predWidth;
    uint8_t buffer[input_c_*input_h_*input_w_];
    // 注意imread读入的图像格式是unsigned char，如果你的网络输入要求是float的话，下面的操作就不对了。
    for (auto i=0; i<predHeight; i++) {
        //printf("+\n");
        for (auto j=0; j<predWidth; j++) {
            buffer[i * predWidth + j + 0*size] = (uint8_t)img.data[(i*predWidth + j) * 3 + 0];
            buffer[i * predWidth + j + 1*size] = (uint8_t)img.data[(i*predWidth + j) * 3 + 1];
            buffer[i * predWidth + j + 2*size] = (uint8_t)img.data[(i*predWidth + j) * 3 + 2];
        }
    }
    // mean, std
    for(int k=0; k< input_c_*input_h_*input_w_; k++){
        preInput[k] = ((float)buffer[k]/255.0 - 0.5)/0.5;
    }
}

void CrowdCount::preprocess(const cv::Mat& img, float* preInput){

    cv::Mat re = img.clone();
    cv::cvtColor(re,re,cv::COLOR_BGR2RGB);
    cv::resize(re,re,cv::Size(input_w_, input_h_));

    int predHeight = re.rows;
    int predWidth = re.cols;
    int size = predHeight * predWidth;
    uint8_t buffer[input_c_*input_h_*input_w_];
    // 注意imread读入的图像格式是unsigned char，如果你的网络输入要求是float的话，下面的操作就不对了。
    for (auto i=0; i<predHeight; i++) {
        //printf("+\n");
        for (auto j=0; j<predWidth; j++) {
            buffer[i * predWidth + j + 0*size] = (uint8_t)re.data[(i*predWidth + j) * 3 + 0];
            buffer[i * predWidth + j + 1*size] = (uint8_t)re.data[(i*predWidth + j) * 3 + 1];
            buffer[i * predWidth + j + 2*size] = (uint8_t)re.data[(i*predWidth + j) * 3 + 2];
        }
    }
    // mean, std
    for(int k=0; k< input_c_*input_h_*input_w_; k++){
        preInput[k] = ((float)buffer[k]/255.0 - 0.5)/0.5;
    }
}

void CrowdCount::process(const std::vector<cv::Mat>& input, std::vector<float> &outputs){

    int fcount = 0;
    float *data = new float[batchSize_*input_c_*input_h_*input_w_];
    float *prob = new float[batchSize_*output_c_*output_h_*output_w_];
    int t;
    double runtime=0;
    for(int f=0; f<(int)input.size(); f++){
        fcount++;
        if(fcount<batchSize_ && f+1 != (int)input.size()) {
            continue;
        }
        for(int b=0; b<fcount; b++){
            cv::Mat img = input[f-fcount+1+b];
            if(img.empty()) continue;
            float img_data[input_c_*input_h_*input_w_];
            preprocess(img, img_data);
            for(int i=0; i<input_c_*input_h_*input_w_; i++){
                data[b*input_c_*input_h_*input_w_+i] = img_data[i];
            }
        }

        // Run inference
        auto start_time = std::chrono::system_clock::now();
        doInference(data, prob, fcount);
        auto end_time = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time);
        runtime+=double(duration.count());
        t += fcount;
        std::cout <<  "CrowdCount inference cost: " << runtime/t << "  ms"<< std::endl;

        float sum = prob[0];
        for(int b=1; b<batchSize_*output_c_*output_h_*output_w_; b++){
           if(0 == b%(output_c_*output_h_*output_w_-1)){
               sum += prob[b];
               outputs.push_back(sum);
               if(b+1<batchSize_*output_c_*output_h_*output_w_) sum = prob[b];
           }else{
               sum+=prob[b];
           }
        }
        fcount = 0;
    }
    delete[] data;
    delete[] prob;
}







