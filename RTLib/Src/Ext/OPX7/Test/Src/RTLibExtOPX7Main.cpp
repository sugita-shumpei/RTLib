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
void TracerTest()
{
    {
        auto testApp = RTLibExtOPX7TestApplication(RTLIB_EXT_OPX7_TEST_CUDA_PATH "/../scene.json", "RIS", false, false, false);
        try
        {
            testApp.Initialize();

            auto tracerName        = testApp.GetTracerName();
            auto maxSamples        = testApp.GetMaxSamples();
            auto samplesPerSave    = testApp.GetSamplesPerSave();
            auto fraction          = testApp.GetTraceConfig().custom.GetFloat1Or("MortonTree.Fraction",0.3f);
            auto iterationForBuilt = testApp.GetTraceConfig().custom.GetUInt32Or("MortonTree.IterationForBuilt", 3);
            auto ratioForBudget    = testApp.GetTraceConfig().custom.GetFloat1Or("MortonTree.RatioForBudget", 0.5f);
            auto hashGridCellSize  = testApp.GetTraceConfig().custom.GetFloat1Or("HashGrid.CellSize", 32768);
            auto imagePath         = testApp.GetTraceConfig().imagePath;
            //std::filesystem::create_directory(std::filesystem::path(imagePath)/"Exp1");
            //std::filesystem::create_directory(std::filesystem::path(imagePath)/"Exp2");
            //std::filesystem::create_directory(std::filesystem::path(imagePath)/"Exp5");
            //for (int i = 0; i < 6;++i) 
            {
                {

                    //float newFraction  = static_cast<float>(i + 1) * 0.1f;
                    //auto  newImagePath = std::filesystem::path(imagePath+("/Exp1/Fraction=" + std::to_string(newFraction))).lexically_normal();
                    //std::filesystem::create_directory(newImagePath);
                    //std::cout << newImagePath << std::endl;
                    //testApp.GetTraceConfig().imagePath = newImagePath.string();
                    //testApp.GetTraceConfig().custom.SetFloat1("MortonTree.Fraction", newFraction);
                } 
                {

                //    auto  newImagePath = std::filesystem::path(imagePath+("/Exp2/IterationForBuilt=" + std::to_string(i))).lexically_normal();
                //    std::filesystem::create_directory(newImagePath);
                //    std::cout << newImagePath << std::endl;
                //    testApp.GetTraceConfig().imagePath = newImagePath.string();
                //    testApp.GetTraceConfig().custom.SetUInt32("MortonTree.IterationForBuilt", i);
                }
                {
                    //float newRatioForBudget  = static_cast<float>(i + 1) * 0.1f;
                    //auto  newImagePath = std::filesystem::path(imagePath+("/Exp3/RatioForBudget=" + std::to_string(newRatioForBudget))).lexically_normal();
                    //std::filesystem::create_directory(newImagePath);
                    //std::cout << newImagePath << std::endl;
                    //testApp.GetTraceConfig().imagePath = newImagePath.string();
                    //testApp.GetTraceConfig().custom.SetFloat1("MortonTree.RatioForBudget", newRatioForBudget);
                }
                //{
                //    auto  newImagePath = std::filesystem::path(imagePath+("/Exp4/Level=" + std::to_string(rtlib::test::MortonQTreeWrapper::kMaxTreeLevel))).lexically_normal();
                //    std::filesystem::create_directory(newImagePath);
                //    std::cout << newImagePath << std::endl;
                //    testApp.GetTraceConfig().imagePath = newImagePath.string();
                ////}
                //{
                //    auto newHashGridCellSize = (128 * 128 * 64 / 128) * static_cast<unsigned int>(1 << i);
                //    auto  newImagePath = std::filesystem::path(imagePath + ("/Exp5/CellSize=" + std::to_string(newHashGridCellSize))).lexically_normal();
                //    std::filesystem::create_directory(newImagePath);
                //    std::cout << newImagePath << std::endl;
                //    testApp.GetTraceConfig().imagePath = newImagePath.string();
                //    testApp.GetTraceConfig().custom.SetFloat1("HashGrid.CellSize", newHashGridCellSize);
                //}
                //testApp.ResetGrids();
                //testApp.SetMaxSamples(100);
                //testApp.SetSamplesPerSave(100);
                //testApp.MainLoop();
                //testApp.ResetGrids();
                //testApp.SetMaxSamples(1000);
                //testApp.SetSamplesPerSave(1000);
                //testApp.MainLoop();
                //testApp.ResetGrids();
                testApp.SetMaxSamples(1000000);
                testApp.SetSamplesPerSave(100000);
                testApp.MainLoop();
            }
            testApp.GetTraceConfig().custom.SetFloat1("HashGrid.CellSize", hashGridCellSize);
            testApp.GetTraceConfig().custom.SetFloat1("MortonTree.RatioForBudget", ratioForBudget);
            testApp.GetTraceConfig().custom.SetFloat1("MortonTree.Fraction",fraction);
            testApp.GetTraceConfig().custom.SetUInt32("MortonTree.IterationForBuilt", iterationForBuilt);
            testApp.GetTraceConfig().imagePath = imagePath;
            testApp.SetTracerName(tracerName);
            testApp.SetMaxSamples(maxSamples);
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
int main()
{   
    TracerTest();
}
