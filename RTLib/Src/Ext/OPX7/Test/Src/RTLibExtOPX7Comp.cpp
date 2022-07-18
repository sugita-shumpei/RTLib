#include <RTLibExtOPX7Test.h>
int main(int argc, const char* argv[]) {
    bool isAllRange = true;
    auto xCenter = unsigned int(262);
    auto yCenter = unsigned int(662);
    auto xRange = 128;
    auto yRange = 128;
    auto baseSamples = 10000;
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
    auto filePath = std::filesystem::path(RTLIB_EXT_OPX7_TEST_DATA_PATH"\\..\\Result\\Scene0\\Depth=10").make_preferred();
    auto baseImageData = std::vector<float3>();
    auto imageSizeX = static_cast<int>(0);
    auto imageSizeY = static_cast<int>(0);
    {
        std::ifstream jsonFile(filePath / "DEF" / std::string("config_DEF_" + std::to_string(baseSamples) + ".json"), std::ios::binary);
        auto jsonData = nlohmann::json::parse(jsonFile);
        jsonFile.close();
        imageSizeX = jsonData.at("Width" ).get<int>();
        imageSizeY = jsonData.at("Height").get<int>();
    }
    {
        baseImageData.resize(imageSizeX * imageSizeY);
        std::ifstream imageFile(filePath / "DEF" / std::string("result_DEF_" + std::to_string(baseSamples) + ".bin"), std::ios::binary);
        if (imageFile.is_open()) {
            imageFile.read((char*)baseImageData.data(), baseImageData.size() * sizeof(baseImageData[0]));
        }
        imageFile.close();
    }
    auto defMAEs   = std::vector<std::tuple<unsigned int, float, float>>();
    auto neeMAEs   = std::vector<std::tuple<unsigned int, float, float>>();
    auto risMAEs   = std::vector<std::tuple<unsigned int, float, float>>();
    auto pgdefMAEs = std::vector<std::tuple<unsigned int, float, float>>();
    auto pgneeMAEs = std::vector<std::tuple<unsigned int, float, float>>();
    auto pgrisMAEs = std::vector<std::tuple<unsigned int, float, float>>();
    auto htdefMAEs = std::vector<std::tuple<unsigned int, float, float>>();
    auto htneeMAEs = std::vector<std::tuple<unsigned int, float, float>>();
    auto htrisMAEs = std::vector<std::tuple<unsigned int, float, float>>();
    for (std::filesystem::directory_entry pipelineDir : std::filesystem::directory_iterator(filePath)) {
        if (pipelineDir.is_directory()) {
            for (std::filesystem::directory_entry imageDir : std::filesystem::directory_iterator(pipelineDir.path())) {
                auto compImageData = std::vector<float3>();
                if (imageDir.path().extension() == ".json") {
                    auto filename = imageDir.path().filename();
                    std::string result, pipeline, sampleStr;
                    std::stringstream ss(filename.string());
                    {
                        std::getline(ss, result, '_');
                        std::getline(ss, pipeline, '_');
                        std::getline(ss, sampleStr, '.');
                    }
                    auto binFilePath = std::filesystem::path();
                    auto time        = float(0.0f);
                    {
                        std::ifstream jsonFile(imageDir.path());
                        auto json   = nlohmann::json::parse(jsonFile);
                        time        = json.at("Time").get<float>();
                    }
                    compImageData.resize(imageSizeX * imageSizeY);
                    std::ifstream imageFile(pipelineDir.path()/("result_"+ pipeline+"_"+sampleStr+".bin"), std::ios::binary);
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
                        defMAEs.push_back({ std::stoi(sampleStr),time,mae });
                    }
                    if (pipeline == "NEE") {
                        neeMAEs.push_back({ std::stoi(sampleStr),time,mae });
                    }
                    if (pipeline == "RIS") {
                        risMAEs.push_back({ std::stoi(sampleStr),time,mae });
                    }
                    if (pipeline == "PGDEF") {
                        pgdefMAEs.push_back({ std::stoi(sampleStr),time,mae });
                    }
                    if (pipeline == "PGNEE") {
                        pgneeMAEs.push_back({ std::stoi(sampleStr),time,mae });
                    }
                    if (pipeline == "PGRIS") {
                        pgrisMAEs.push_back({ std::stoi(sampleStr),time,mae });
                    }
                    if (pipeline == "HTDEF") {
                        htdefMAEs.push_back({ std::stoi(sampleStr),time,mae });
                    }
                    if (pipeline == "HTNEE") {
                        htneeMAEs.push_back({ std::stoi(sampleStr),time,mae });
                    }
                    if (pipeline == "HTRIS") {
                        htrisMAEs.push_back({ std::stoi(sampleStr),time,mae });
                    }
                }

            }
        }
    }
    std::sort(std::begin(defMAEs), std::end(defMAEs), [](const auto& a, const auto& b) {
        return std::get<0>(a) < std::get<0>(b);
        });
    std::sort(std::begin(neeMAEs), std::end(neeMAEs), [](const auto& a, const auto& b) {
        return std::get<0>(a) < std::get<0>(b);
        });
    std::sort(std::begin(risMAEs), std::end(risMAEs), [](const auto& a, const auto& b) {
        return std::get<0>(a) < std::get<0>(b);
        });
    std::sort(std::begin(pgdefMAEs), std::end(pgdefMAEs), [](const auto& a, const auto& b) {
        return std::get<0>(a) < std::get<0>(b);
        });
    std::sort(std::begin(pgneeMAEs), std::end(pgneeMAEs), [](const auto& a, const auto& b) {
        return std::get<0>(a) < std::get<0>(b);
        });
    std::sort(std::begin(pgrisMAEs), std::end(pgrisMAEs), [](const auto& a, const auto& b) {
        return std::get<0>(a) < std::get<0>(b);
        });
    std::sort(std::begin(htdefMAEs), std::end(htdefMAEs), [](const auto& a, const auto& b) {
        return std::get<0>(a) < std::get<0>(b);
        });
    std::sort(std::begin(htneeMAEs), std::end(htneeMAEs), [](const auto& a, const auto& b) {
        return std::get<0>(a) < std::get<0>(b);
        });
    std::sort(std::begin(htrisMAEs), std::end(htrisMAEs), [](const auto& a, const auto& b) {
        return std::get<0>(a) < std::get<0>(b);
        });
    for (auto& [sample, time,value] : defMAEs) {
        std::cout << "DEF: " << sample << "," << time << ", " << value << std::endl;
    }
    for (auto& [sample, time, value] : neeMAEs) {
        std::cout << "NEE: " << sample << "," << time << ", " << value << std::endl;
    }
    for (auto& [sample, time, value] : risMAEs) {
        std::cout << "RIS: " << sample << "," << time << ", " << value << std::endl;
    }
    for (auto& [sample, time, value] : pgdefMAEs) {
        std::cout << "PGDEF: " << sample << "," << time << ", " << value << std::endl;
    }
    for (auto& [sample, time, value] : pgneeMAEs) {
        std::cout << "PGNEE: " << sample << "," << time << ", " << value << std::endl;
    }
    for (auto& [sample, time, value] : pgrisMAEs) {
        std::cout << "PGRIS: " << sample << "," << time << ", " << value << std::endl;
    }
    for (auto& [sample, time, value] : htdefMAEs) {
        std::cout << "HTDEF: " << sample << "," << time << ", " << value << std::endl;
    }
    for (auto& [sample, time, value] : htneeMAEs) {
        std::cout << "HTNEE: " << sample << "," << time << ", " << value << std::endl;
    }
    for (auto& [sample, time, value] : htrisMAEs) {
        std::cout << "HTRIS: " << sample << "," << time << ", " << value << std::endl;
    }
}