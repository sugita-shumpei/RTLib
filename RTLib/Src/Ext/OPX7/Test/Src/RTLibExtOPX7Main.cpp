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
            testApp.SetTracerName("PGNEE");
            testApp.SetMaxSamples(100);
            testApp.SetSamplesPerSave(10);
            testApp.MainLoop();
            testApp.SetMaxSamples(1000);
            testApp.SetSamplesPerSave(100);
            testApp.MainLoop();
            testApp.SetMaxSamples(10000);
            testApp.SetSamplesPerSave(1000);
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
