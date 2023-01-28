#include <RTLibExtOPX7TestApplication.h>
void SampleTest()
{
    auto quadTreeBuffer = std::vector<float>(341, 0.0f);
    auto quadTree = RTLib::Ext::OPX7::Utils::MortonQuadTreeT<4>(4, quadTreeBuffer.data());
    //00|00|00|00    21->840mb
    //00|00|00|00|00 85->3.4gb
    auto rnd = std::random_device();
    std::mt19937 mt(rnd());
    auto rng = RTLib::Ext::CUDA::Math::Xorshift32(rnd());
    for (auto i = 0; i < 1000; ++i)
    {
        std::cout << "level " << i << std::endl;
        quadTree.Record({ std::uniform_real_distribution{ 0.0f,1.0f }(mt), std::uniform_real_distribution{ 0.0f,1.0f }(mt) }, 1.0f);
    }
    for (auto i = 0; i < 1000; ++i)
    {
        auto  pdf = float(0.0f);
        auto val = quadTree.SampleAndPdf(pdf, rng);
        std::cout << "(" << val.x << "," << val.y << "): " << pdf << "vs" << quadTree.Pdf(val) << std::endl;
    }
}
void TracerTest(int argc, const char* argv[])
{
    {
        auto scenePath = std::string(RTLIB_EXT_OPX7_TEST_CUDA_PATH "/../scene.json");
        {
            if (argc > 1) {
                for (int i = 1; i < argc - 1; ++i) {
                    if (std::string(argv[i]) == "--ScenePath") {
                        auto tmpScenePath = std::filesystem::path(std::string(argv[i + 1]));
                        scenePath = tmpScenePath.string();
                        
                    }
                }
            }
        }
        auto testApp = RTLibExtOPX7TestApplication(scenePath, "RIS", true, true, true);
        try
        {

            testApp.Initialize(argc,argv);
            auto tracerName        = testApp.GetTracerName();
            auto maxDepth          = testApp.GetMaxDepth();
            auto maxSamples        = testApp.GetMaxSamples();
            auto maxTimes          = testApp.GetMaxTimes();
            auto samplesPerSave    = testApp.GetSamplesPerSave();
            auto fraction          = testApp.GetTraceConfig().custom.GetFloat1Or("MortonTree.Fraction"         , 0.3f);
            auto iterationForBuilt = testApp.GetTraceConfig().custom.GetUInt32Or("MortonTree.IterationForBuilt", 3);
            auto ratioForBudget    = testApp.GetTraceConfig().custom.GetFloat1Or("MortonTree.RatioForBudget"   , 0.5f);
            auto hashGridCellSize  = testApp.GetTraceConfig().custom.GetFloat1Or("HashGrid.CellSize"           , 32768);
            auto imagePath         = testApp.GetTraceConfig().imagePath;
            {
                if (argc > 1) {
                    for (int i = 1; i < argc-1; ++i) {
                        if (std::string(argv[i]) == "--MaxDepth"      ) {
                            testApp.SetMaxDepth(std::stoi(std::string(argv[i + 1])));
                        }
                        if (std::string(argv[i]) == "--MaxTimes"      ) {
                            testApp.SetMaxTimes(std::stof(std::string(argv[i + 1])));
                        }
                        if (std::string(argv[i]) == "--MaxSamples"    ) {
                            testApp.SetMaxSamples(std::stoi(std::string(argv[i + 1])));
                        }
                        if (std::string(argv[i]) == "--SamplesPerSave") {
                            testApp.SetSamplesPerSave(std::stoi(std::string(argv[i + 1])));
                        }
                        if (std::string(argv[i]) == "--DefTracer"     ) {
                            testApp.SetTracerName(std::string(argv[i + 1]));
                        }
                        if (std::string(argv[i]) == "--ImagePath") {
                            testApp.GetTraceConfig().imagePath = std::string(argv[i + 1]);
                        }
                    }
                }
                testApp.ResetSdTree();
                testApp.ResetGrids();
                testApp.MainLoop();
            }
            testApp.GetTraceConfig().custom.SetFloat1(  "HashGrid.CellSize"         , hashGridCellSize);
            testApp.GetTraceConfig().custom.SetFloat1("MortonTree.RatioForBudget"   , ratioForBudget);
            testApp.GetTraceConfig().custom.SetFloat1("MortonTree.Fraction"         , fraction);
            testApp.GetTraceConfig().custom.SetUInt32("MortonTree.IterationForBuilt", iterationForBuilt);
            testApp.GetTraceConfig().imagePath = imagePath;
            testApp.SetTracerName(tracerName);
            testApp.SetMaxSamples(maxSamples);
            testApp.SetMaxTimes(maxTimes);
            testApp.SetMaxDepth(maxDepth);
            testApp.SetSamplesPerSave(samplesPerSave);
            testApp.Terminate();
        }
        catch (std::runtime_error& err)
        {
            std::cerr << err.what() << std::endl;
        }
        //return 0;
        //RTLibExtOPX7TestApplication(RTLIB_EXT_OPX7_TEST_CUDA_PATH "/../scene.json", "DEF", false).Run();
        //RTLibExtOPX7TestApplication(RTLIB_EXT_OPX7_TEST_CUDA_PATH "/../scene.json", "NEE", false).Run();
        //RTLibExtOPX7TestApplication(RTLIB_EXT_OPX7_TEST_CUDA_PATH "/../scene.json", "RIS", false).Run();
    }
}
int main(int argc,const char** argv)
{   
    TracerTest(argc,argv);
}
