#include <RTLibExtOPX7TestApplication.h>
void Compare() {
    //return RTLibExtOPX7TestApplication(RTLIB_EXT_OPX7_TEST_CUDA_PATH "/../scene.json", "DEF", false).Run();
    auto filePath = std::filesystem::path("D:\\Users\\shums\\Documents\\CMake\\RTLib\\Result\\Scene0");
    auto baseSamples = 10000;
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
                    compImageData.resize(1024 * 1024);
                    std::ifstream imageFile(imageDir.path(), std::ios::binary);
                    if (imageFile.is_open()) {
                        imageFile.read((char*)compImageData.data(), compImageData.size() * sizeof(compImageData[0]));
                    }
                    imageFile.close();

                    auto mae = float(0.0f);
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
int main()
{
    Compare();
    //return RTLibExtOPX7TestApplication(RTLIB_EXT_OPX7_TEST_CUDA_PATH "/../scene.json", "RIS", false).Run();
}
