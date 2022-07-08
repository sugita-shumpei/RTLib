#include <RTLibExtOPX7Test.h>
int main(int argc, const char* argv[]) {
    bool isAllRange = true;
    auto xCenter = unsigned int(262);
    auto yCenter = unsigned int(662);
    auto xRange = 128;
    auto yRange = 128;
    int imageSizeX = 1024;
    int imageSizeY = 1024;
    auto baseSamples = 500000;
    if (argc > 1) {
        isAllRange = false;
        if (std::string(argv[1]) == "--xcenter") {
            xCenter = std::stoi(std::string(argv[2]));
        }
        if (std::string(argv[3]) == "--ycenter") {
            yCenter = std::stoi(std::string(argv[4]));
        }
        if (std::string(argv[5]) == "--xrange") {
            xRange = std::stoi(std::string(argv[6]));
        }
        if (std::string(argv[7]) == "--yrange") {
            yRange = std::stoi(std::string(argv[8]));
        }
    }
    //return RTLibExtOPX7TestApplication(RTLIB_EXT_OPX7_TEST_CUDA_PATH "/../scene.json", "DEF", false).Run();
    auto filePath = std::filesystem::path(RTLIB_EXT_OPX7_TEST_DATA_PATH"\\..\\Result\\Scene0\\Depth=4");
    auto baseImageData = std::vector<float3>();
    {
        baseImageData.resize(1024 * 1024);
        std::ifstream imageFile(filePath / "DEF" / std::string("result_DEF_" + std::to_string(baseSamples) + ".bin"), std::ios::binary);
        if (imageFile.is_open()) {
            imageFile.read((char*)baseImageData.data(), baseImageData.size() * sizeof(baseImageData[0]));
        }
        imageFile.close();
    }
    auto defMAEs = std::vector<std::pair<unsigned int, float>>();
    auto neeMAEs = std::vector<std::pair<unsigned int, float>>();
    auto risMAEs = std::vector<std::pair<unsigned int, float>>();
    for (std::filesystem::directory_entry pipelineDir : std::filesystem::directory_iterator(filePath)) {
        if (pipelineDir.is_directory()) {
            for (std::filesystem::directory_entry imageDir : std::filesystem::directory_iterator(pipelineDir.path())) {
                auto compImageData = std::vector<float3>();
                if (imageDir.path().extension() == ".bin") {
                    auto filename = imageDir.path().filename();
                    std::string result, pipeline, sampleStr;
                    std::stringstream ss(filename.string());
                    {
                        std::getline(ss, result, '_');
                        std::getline(ss, pipeline, '_');
                        std::getline(ss, sampleStr, '.');
                    }
                    compImageData.resize(imageSizeX * imageSizeY);
                    std::ifstream imageFile(imageDir.path(), std::ios::binary);
                    if (imageFile.is_open()) {
                        imageFile.read((char*)compImageData.data(), compImageData.size() * sizeof(compImageData[0]));
                    }
                    imageFile.close();
                    auto mae = float(0.0f);
                    if (isAllRange) {
                        for (int i = 0; i < baseImageData.size(); ++i) {
                            if (!(isnan(baseImageData[i].x) || isnan(baseImageData[i].y) || isnan(baseImageData[i].z) ||
                                isnan(compImageData[i].x) || isnan(compImageData[i].y) || isnan(compImageData[i].z))) {
                                float  delt = (fabsf(baseImageData[i].x - compImageData[i].x)) + (fabsf(baseImageData[i].y - compImageData[i].y)) + (fabsf(baseImageData[i].z - compImageData[i].z));
                                if ((baseImageData[i].x + baseImageData[i].y + baseImageData[i].z) > 0.0f) {
                                    mae += delt / (baseImageData[i].x + baseImageData[i].y + baseImageData[i].z);
                                }
                            }
                        }
                        mae /= static_cast<float>(compImageData.size());
                    }
                    else {
                        for (int j = yCenter - yRange / 2; j < yCenter + yRange / 2; ++j) {

                            for (int i = xCenter - xRange / 2; i < xCenter + xRange / 2; ++i) {
                                auto baseColor = baseImageData[imageSizeX * j + i];
                                auto compColor = compImageData[imageSizeX * j + i];
                                float  delt = (fabsf(baseColor.x - compColor.x)) + (fabsf(baseColor.y - compColor.y)) + (fabsf(baseColor.z - compColor.z));
                                mae += delt / (baseColor.x + baseColor.y + baseColor.z);
                            }
                        }
                        mae /= static_cast<float>(xRange * yRange);
                    }
                    if (pipeline == "DEF") {
                        defMAEs.push_back({ std::stoi(sampleStr),mae });
                    }
                    if (pipeline == "NEE") {
                        neeMAEs.push_back({ std::stoi(sampleStr),mae });
                    }
                    if (pipeline == "RIS") {
                        risMAEs.push_back({ std::stoi(sampleStr),mae });
                    }
                }

            }
        }
    }
    std::sort(std::begin(defMAEs), std::end(defMAEs), [](const auto& a, const auto& b) {
        return a.first < b.first;
        });
    std::sort(std::begin(neeMAEs), std::end(neeMAEs), [](const auto& a, const auto& b) {
        return a.first < b.first;
        });
    std::sort(std::begin(risMAEs), std::end(risMAEs), [](const auto& a, const auto& b) {
        return a.first < b.first;
        });
    for (auto& [sample, value] : defMAEs) {
        std::cout << "DEF: " << sample << "," << value << std::endl;
    }
    for (auto& [sample, value] : neeMAEs) {
        std::cout << "NEE: " << sample << "," << value << std::endl;
    }
    for (auto& [sample, value] : risMAEs) {
        std::cout << "RIS: " << sample << "," << value << std::endl;
    }
}