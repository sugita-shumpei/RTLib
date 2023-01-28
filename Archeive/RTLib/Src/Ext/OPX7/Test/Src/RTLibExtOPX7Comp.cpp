#include <RTLibExtOPX7Test.h>
#include <RTLib/Core/BinaryWriter.h>
int main(int argc, const char* argv[]) {
    bool isAllRange = true;
    auto xCenter = unsigned int(262);
    auto yCenter = unsigned int(662);
    auto xRange = 128;
    auto yRange = 128;
    auto baseSamples = 1000000;
    auto defPath  = std::string(RTLIB_EXT_OPX7_TEST_DATA_PATH"\\..\\Result\\Scene1\\Depth=10_10sec");
    auto baseDir = std::string("");
    auto compDir = std::string("");
    bool imgDiff = false;
    if (argc > 1) {
        for (int i = 0; i < argc-1; ++i) {
            if (std::string(argv[i]) == "--comp_dir") {
                compDir = std::string(argv[i + 1]);
            }
            if (std::string(argv[i]) == "--base_dir") {
                baseDir = std::string(argv[i + 1]);
            }
            if (std::string(argv[i]) == "--base_smp") {
                baseSamples = std::stoi(std::string(argv[i + 1]));
            }
            if (std::string(argv[i]) == "--xcenter") {
                xCenter = std::stoi(std::string(argv[i+1]));
                isAllRange = false;
            }
            if (std::string(argv[i]) == "--ycenter") {
                yCenter = std::stoi(std::string(argv[i + 1]));
                isAllRange = false;
            }
            if (std::string(argv[i]) == "--xrange") {
                xRange = std::stoi(std::string(argv[i + 1]));
                isAllRange = false;
            }
            if (std::string(argv[i]) == "--yrange") {
                yRange = std::stoi(std::string(argv[i + 1]));
                isAllRange = false;
            }
            if (std::string(argv[i]) == "--img_diff") {
                std::string img_diff_mode = std::string(argv[i + 1]);
                if (((img_diff_mode == "ON") || (img_diff_mode == "On") || (img_diff_mode == "on")) ||
                    ((img_diff_mode == "TRUE") || (img_diff_mode == "True") || (img_diff_mode == "true")))
                {
                    imgDiff = true;
                }
                if (((img_diff_mode == "OFF") || (img_diff_mode == "Off") || (img_diff_mode == "off")) ||
                    ((img_diff_mode == "FALSE") || (img_diff_mode == "False") || (img_diff_mode == "false")))
                {
                    imgDiff = true;
                }
            }
        }
        
    }
    if (baseDir.empty()) {
        baseDir = defPath;
    }
    if (compDir.empty()) {
        compDir = baseDir;
    }
    //return RTLibExtOPX7TestApplication(RTLIB_EXT_OPX7_TEST_CUDA_PATH "/../scene.json", "DEF", false).Run();
    auto filePath = std::filesystem::path(baseDir).make_preferred();
    auto baseImageData = std::vector<float3>();
    auto imageSizeX = static_cast<int>(0);
    auto imageSizeY = static_cast<int>(0);
    {
        std::ifstream jsonFile(std::filesystem::path(compDir).make_preferred() / "DEF" / std::string("config_DEF_" + std::to_string(baseSamples) + ".json"), std::ios::binary);
        auto jsonData = nlohmann::json::parse(jsonFile);
        jsonFile.close();
        imageSizeX = jsonData.at("Width" ).get<int>();
        imageSizeY = jsonData.at("Height").get<int>();
    }
    {
        baseImageData.resize(imageSizeX * imageSizeY);
        std::ifstream imageFile(std::filesystem::path(compDir).make_preferred() / "DEF" / std::string("result_DEF_" + std::to_string(baseSamples) + ".bin"), std::ios::binary);
        if (imageFile.is_open()) {
            imageFile.read((char*)baseImageData.data(), baseImageData.size() * sizeof(baseImageData[0]));
        }
        else {
            std::cout << "FAILED TO OPEN FILE!\n";
        }
        imageFile.close();
    }
    auto defMAEs   = std::vector<std::tuple<unsigned int, float, float, float>>();
    auto neeMAEs   = std::vector<std::tuple<unsigned int, float, float, float>>();
    auto risMAEs   = std::vector<std::tuple<unsigned int, float, float, float>>();
    auto pgdefMAEs = std::vector<std::tuple<unsigned int, float, float, float>>();
    auto pgneeMAEs = std::vector<std::tuple<unsigned int, float, float, float>>();
    auto pgrisMAEs = std::vector<std::tuple<unsigned int, float, float, float>>();
    auto htdefMAEs = std::vector<std::tuple<unsigned int, float, float, float>>();
    auto htneeMAEs = std::vector<std::tuple<unsigned int, float, float, float>>();
    auto htrisMAEs = std::vector<std::tuple<unsigned int, float, float, float>>();
    for (std::filesystem::directory_entry pipelineDir : std::filesystem::directory_iterator(filePath)) {\
        if (pipelineDir.is_directory()) {
            for (std::filesystem::directory_entry imageDir : std::filesystem::directory_iterator(pipelineDir.path())) {
                auto compImageData = std::vector<float3>();
                if (imageDir.path().extension() == ".json") {
                    auto filename = imageDir.path().filename();
                    if (filename == "scene.json") {
                        continue;
                    }
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
                    auto mean = float(0.0f);
                    auto rmae  = float(0.0f);
                    auto smape = float(0.0f);
                    if (isAllRange) {
                        for (int i = 0; i < baseImageData.size(); ++i) {
                            if (!(isnan(baseImageData[i].x) || isnan(baseImageData[i].y) || isnan(baseImageData[i].z) ||
                                  isnan(compImageData[i].x) || isnan(compImageData[i].y) || isnan(compImageData[i].z))) {
                                /*                              mae += (fabsf(baseImageData[i].x - compImageData[i].x)) + (fabsf(baseImageData[i].y - compImageData[i].y)) + (fabsf(baseImageData[i].z - compImageData[i].z));*/
                                if (sqrtf(powf(baseImageData[i].x, 2.0f) + powf(baseImageData[i].y, 2.0f) + powf(baseImageData[i].z, 2.0f)) > 0.0f) {
                                    smape+= sqrtf((powf(baseImageData[i].x - compImageData[i].x, 2.0f)) + (powf(baseImageData[i].y - compImageData[i].y, 2.0f)) + (powf(baseImageData[i].z - compImageData[i].z, 2.0f))) / sqrtf(powf(baseImageData[i].x, 2.0f) + powf(baseImageData[i].y, 2.0f) + powf(baseImageData[i].z, 2.0f));
                                    rmae += sqrtf((powf(baseImageData[i].x - compImageData[i].x, 2.0f)) + (powf(baseImageData[i].y - compImageData[i].y, 2.0f)) + (powf(baseImageData[i].z - compImageData[i].z, 2.0f)));
                                    mean +=(sqrtf( powf(baseImageData[i].x, 2.0f) + powf(baseImageData[i].y, 2.0f) + powf(baseImageData[i].z, 2.0f))+ sqrtf(powf(compImageData[i].x, 2.0f) + powf(compImageData[i].y, 2.0f) + powf(compImageData[i].z, 2.0f)))*0.5f;
                                }
                            }
                        }
                        smape /= static_cast<float>(baseImageData.size());
                        rmae /= mean;
                    }
                    else {
                        for (int j = yCenter - yRange / 2; j < yCenter + yRange / 2; ++j) {
                            for (int i = xCenter - xRange / 2; i < xCenter + xRange / 2; ++i) {
                                auto baseColor = baseImageData[imageSizeX * j + i];
                                auto compColor = compImageData[imageSizeX * j + i];
                                if (!(isnan(baseColor.x) || isnan(baseColor.y) || isnan(baseColor.z) ||
                                    isnan(compColor.x) || isnan(compColor.y) || isnan(compColor.z))) {
                                    smape+= sqrtf((powf(baseColor.x - compColor.x, 2.0f)) + (powf(baseColor.y - compColor.y, 2.0f)) + (powf(baseColor.z - compColor.z, 2.0f))) / sqrtf(powf(baseColor.x, 2.0f) + powf(baseColor.y, 2.0f) + powf(baseColor.z, 2.0f));
                                    rmae += sqrtf((powf(baseColor.x - compColor.x, 2.0f)) + (powf(baseColor.y - compColor.y, 2.0f)) + (powf(baseColor.z - compColor.z, 2.0f)));
                                    mean += (sqrtf(powf(baseColor.x, 2.0f) + powf(baseColor.y, 2.0f) + powf(baseColor.z, 2.0f))+ sqrtf(powf(compColor.x, 2.0f) + powf(compColor.y, 2.0f) + powf(compColor.z, 2.0f)))/2.0f;
                                }
                            }
                        }
                        smape /= static_cast<float>(baseImageData.size());
                        rmae  /= mean;
                    }
                    if (imgDiff)
                    {
                        std::vector<float> errImages(baseImageData.size(),0.0f);
                        for (int i = 0; i < baseImageData.size(); ++i) {
                            if (!(isnan(baseImageData[i].x) || isnan(baseImageData[i].y) || isnan(baseImageData[i].z) ||
                                isnan(compImageData[i].x) || isnan(compImageData[i].y) || isnan(compImageData[i].z))) {
                                float  delt = (fabsf(baseImageData[i].x - compImageData[i].x)) + (fabsf(baseImageData[i].y - compImageData[i].y)) + (fabsf(baseImageData[i].z - compImageData[i].z));
                                if ((baseImageData[i].x + baseImageData[i].y + baseImageData[i].z) > 0.0f) {
                                    errImages[i] = delt/(baseImageData[i].x + baseImageData[i].y + baseImageData[i].z);
                                }
                            }
                        }
                        std::filesystem::path savePath = pipelineDir.path() / ("diff_" + pipeline + "_" + sampleStr + ".hdr");
                        std::string savePathStr = savePath.string();
                        RTLib::Core::SaveHdrImage(savePathStr.c_str(), imageSizeX, imageSizeY,errImages);
                    }
                    if (pipeline == "DEF") {
                        defMAEs.push_back({ std::stoi(sampleStr),time,rmae ,smape });
                    }
                    if (pipeline == "NEE") {
                        neeMAEs.push_back({ std::stoi(sampleStr),time,rmae ,smape });
                    }
                    if (pipeline == "RIS") {
                        risMAEs.push_back({ std::stoi(sampleStr),time,rmae ,smape });
                    }
                    if (pipeline == "PGDEF") {
                        pgdefMAEs.push_back({ std::stoi(sampleStr),time,rmae ,smape });
                    }
                    if (pipeline == "PGNEE") {
                        pgneeMAEs.push_back({ std::stoi(sampleStr),time,rmae ,smape });
                    }
                    if (pipeline == "PGRIS") {
                        pgrisMAEs.push_back({ std::stoi(sampleStr),time,rmae,smape });
                    }
                    if (pipeline == "HTDEF") {
                        htdefMAEs.push_back({ std::stoi(sampleStr),time,rmae,smape });
                    }
                    if (pipeline == "HTNEE") {
                        htneeMAEs.push_back({ std::stoi(sampleStr),time,rmae,smape });
                    }
                    if (pipeline == "HTRIS") {
                        htrisMAEs.push_back({ std::stoi(sampleStr),time,rmae,smape });
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
    std::cout << "Type, Sample, Time, MAE, MAPE" << std::endl;
    for (auto& [sample, time,mae, mape] : defMAEs) {
        std::cout << "DEF," << sample << "," << time << ", " << mae << ", " << mape << std::endl;
    }
    for (auto& [sample, time, mae, mape]: neeMAEs) {
        std::cout << "NEE," << sample << "," << time << ", " << mae << ", " << mape << std::endl;
    }
    for (auto& [sample, time, mae, mape] : risMAEs) {
        std::cout << "RIS," << sample << "," << time << ", " << mae << ", " << mape << std::endl;
    }
    for (auto& [sample, time, mae, mape] : pgdefMAEs) {
        std::cout << "PGDEF," << sample << "," << time << ", " << mae << ", " << mape << std::endl;
    }
    for (auto& [sample, time, mae, mape] : pgneeMAEs) {
        std::cout << "PGNEE," << sample << "," << time << ", " << mae << ", " << mape << std::endl;
    }
    for (auto& [sample, time, mae, mape] : pgrisMAEs) {
        std::cout << "PGRIS," << sample << "," << time << ", " << mae << ", " << mape << std::endl;
    }
    for (auto& [sample, time, mae, mape] : htdefMAEs) {
        std::cout << "HTDEF," << sample << "," << time << ", " << mae << ", " << mape << std::endl;
    }
    for (auto& [sample, time, mae, mape] : htneeMAEs) {
        std::cout << "HTNEE," << sample << "," << time << ", " << mae << ", " << mape << std::endl;
    }
    for (auto& [sample, time, mae, mape] : htrisMAEs) {
        std::cout << "HTRIS," << sample << "," << time << ", " << mae << ", " << mape << std::endl;
    }
}