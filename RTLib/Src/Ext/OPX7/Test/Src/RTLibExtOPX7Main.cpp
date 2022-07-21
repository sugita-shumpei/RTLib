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

    auto testApp = RTLibExtOPX7TestApplication(RTLIB_EXT_OPX7_TEST_CUDA_PATH "/../scene.json", "HTDEF", false, true, false);
    try
    {
        testApp.Initialize();

        auto pipelineName = testApp.GetTracerName();
        auto maxSamples = testApp.GetMaxSamples();
        auto samplesPerSave = testApp.GetSamplesPerSave();
        {
            testApp.SetMaxSamples(100);
            testApp.SetSamplesPerSave(10);
            testApp.MainLoop();
        }
        testApp.SetTracerName(pipelineName);
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
int main()
{   
    TracerTest();
}
