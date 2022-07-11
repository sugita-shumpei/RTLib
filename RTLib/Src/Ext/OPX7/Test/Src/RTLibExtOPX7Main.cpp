#include <RTLibExtOPX7TestApplication.h>
int main()
{
    auto testApp = RTLibExtOPX7TestApplication(RTLIB_EXT_OPX7_TEST_CUDA_PATH "/../scene2.json", "DEF", true,false,true);
    try
    {
        testApp.Initialize();

        auto pipelineName = testApp.GetTracerName();
        auto maxSamples = testApp.GetMaxSamples();
        auto samplesPerSave = testApp.GetSamplesPerSave();
        {
            testApp.SetTracerName("DEF");
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
    return 0;
    //RTLibExtOPX7TestApplication(RTLIB_EXT_OPX7_TEST_CUDA_PATH "/../scene.json", "DEF", false).Run();
    //RTLibExtOPX7TestApplication(RTLIB_EXT_OPX7_TEST_CUDA_PATH "/../scene.json", "NEE", false).Run();
    //RTLibExtOPX7TestApplication(RTLIB_EXT_OPX7_TEST_CUDA_PATH "/../scene.json", "RIS", false).Run();
}
