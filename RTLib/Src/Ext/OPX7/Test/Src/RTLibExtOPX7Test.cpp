#include <RTLibExtOPX7TestApplication.h>
int main()
{
    return RTLibExtOPX7TestApplication(RTLIB_EXT_OPX7_TEST_CUDA_PATH "/../scene.json", "DEF", true).Run();
}
