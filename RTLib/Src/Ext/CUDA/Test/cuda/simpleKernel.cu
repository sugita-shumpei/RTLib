#define __CUDACC__
#include <RTLib/Ext/CUDA/Math/Random.h>
#include <RTLib/Ext/CUDA/Math/VectorFunction.h>
namespace rtlib = RTLib::Ext::CUDA::Math;
class Perline {
public:
    static RTLIB_DEVICE auto Generate(float x, float y, float z) -> float
    {
        inline constexpr unsigned char p[512] = {
123,61,55,66,162,185,6,175,34,233,24,13,201,151,89,136,
192,73,103,121,213,78,190,247,105,114,108,163,95,27,132,193,
115,143,180,118,133,255,177,171,60,200,43,110,240,206,199,124,
3,75,65,126,209,140,187,134,97,232,173,179,253,54,183,86,
150,215,84,29,212,48,207,117,197,222,181,148,69,170,92,19,
88,106,56,196,241,226,82,155,71,182,25,22,216,1,35,120,
235,58,244,20,41,122,221,227,62,137,11,246,211,129,63,254,
243,178,141,131,228,116,112,50,30,234,8,94,217,74,204,98,
135,67,229,76,237,10,45,72,158,100,236,149,31,119,40,18,
0,38,15,127,64,85,107,152,109,231,28,195,93,42,159,157,
172,113,81,49,9,218,164,242,79,198,90,16,219,205,37,208,
5,59,2,169,77,32,230,160,238,168,142,147,111,156,186,53,
87,161,248,214,36,154,139,57,251,21,225,210,191,14,184,245,
102,176,202,52,23,165,174,80,138,224,239,104,194,166,249,83,
4,128,99,96,220,44,125,223,33,46,144,26,91,167,51,68,
7,189,146,203,101,145,17,39,153,130,250,70,188,12,47,252,
123,61,55,66,162,185,6,175,34,233,24,13,201,151,89,136,
192,73,103,121,213,78,190,247,105,114,108,163,95,27,132,193,
115,143,180,118,133,255,177,171,60,200,43,110,240,206,199,124,
3,75,65,126,209,140,187,134,97,232,173,179,253,54,183,86,
150,215,84,29,212,48,207,117,197,222,181,148,69,170,92,19,
88,106,56,196,241,226,82,155,71,182,25,22,216,1,35,120,
235,58,244,20,41,122,221,227,62,137,11,246,211,129,63,254,
243,178,141,131,228,116,112,50,30,234,8,94,217,74,204,98,
135,67,229,76,237,10,45,72,158,100,236,149,31,119,40,18,
0,38,15,127,64,85,107,152,109,231,28,195,93,42,159,157,
172,113,81,49,9,218,164,242,79,198,90,16,219,205,37,208,
5,59,2,169,77,32,230,160,238,168,142,147,111,156,186,53,
87,161,248,214,36,154,139,57,251,21,225,210,191,14,184,245,
102,176,202,52,23,165,174,80,138,224,239,104,194,166,249,83,
4,128,99,96,220,44,125,223,33,46,144,26,91,167,51,68,
7,189,146,203,101,145,17,39,153,130,250,70,188,12,47,252,
        };
        x = fmodf(x, 256.0f);
        y = fmodf(y, 256.0f);
        z = fmodf(z, 256.0f);
        int xi = static_cast<int>(floorf(x));
        int yi = static_cast<int>(floorf(y));
        int zi = static_cast<int>(floorf(z));
        float xf = x - static_cast<float>(xi);
        float yf = y - static_cast<float>(yi);
        float zf = z - static_cast<float>(zi);
        int aaa = p[p[p[xi+0]+yi+0]+zi+0];
        int bbb = p[p[p[xi+1]+yi+1]+zi+1];
        int baa = p[p[p[xi+1]+yi+0]+zi+0];
        int aba = p[p[p[xi+0]+yi+1]+zi+0];
        int aab = p[p[p[xi+0]+yi+0]+zi+1];
        int abb = p[p[p[xi+0]+yi+1]+zi+1];
        int bab = p[p[p[xi+1]+yi+0]+zi+1];
        int bba = p[p[p[xi+1]+yi+1]+zi+0];
        float u = Fade(xf);
        float v = Fade(yf);
        float w = Fade(zf);
        float x1, x2, y1, y2;
        x1 = Lerp(Grad(aaa, xf, yf     , zf), Grad(baa, xf - 1.0f, yf       , zf), u);
        x2 = Lerp(Grad(aba, xf, yf-1.0f, zf), Grad(bba, xf - 1.0f, yf - 1.0f, zf), u);
        y1 = Lerp(x1, x2, v);
        x1 = Lerp(Grad(aab, xf, yf     , zf-1.0f), Grad(bab, xf - 1.0f, yf       , zf - 1.0f), u);
        x2 = Lerp(Grad(abb, xf, yf-1.0f, zf-1.0f), Grad(bbb, xf - 1.0f, yf - 1.0f, zf - 1.0f), u);
        y2 = Lerp(x1, x2, v);
        return Lerp(y1,y2,w);
    }
private:
    static RTLIB_DEVICE auto Lerp(float x, float y, float t) -> float {
        //y * t+ (1.0f-t)*x
        //x + (y-x)*t
        return fmaf(y - x, t, x);
    }
    static RTLIB_DEVICE auto Grad(int hash, float x, float y, float z)-> float
    {
        switch (hash & 0xF)
        {
        case 0x0: return  x + y;
        case 0x1: return -x + y;
        case 0x2: return  x - y;
        case 0x3: return -x - y;
        case 0x4: return  x + z;
        case 0x5: return -x + z;
        case 0x6: return  x - z;
        case 0x7: return -x - z;
        case 0x8: return  y + z;
        case 0x9: return -y + z;
        case 0xA: return  y - z;
        case 0xB: return -y - z;
        case 0xC: return  y + x;
        case 0xD: return -y + z;
        case 0xE: return  y - x;
        case 0xF: return -y - z;
        default: return 0.0f; // never happens
        }
    }
    static RTLIB_DEVICE auto Fade(float t) -> float {
        return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
    }
};
extern "C" __global__ void rgbKernel(uchar4* inBuffer,uchar4* outBuffer, int width, int height){
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   int j = blockIdx.y * blockDim.y + threadIdx.y;
   if(i<width&&j<height){
       outBuffer[j*width+i] = rtlib::srgb_to_rgba(inBuffer[j*width+i]);
   }
}
extern "C" __global__ void blurKernel(uchar4* inBuffer,uchar4* outBuffer, int width, int height){
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   int j = blockIdx.y * blockDim.y + threadIdx.y;
   if(i<width&&j<height){
       unsigned long long seed = static_cast<unsigned long long>(j)*width+i;
       auto rng      = rtlib::Xorshift128(seed);
       auto random_v = rtlib::random_float2(-5.0f,5.0f,rng);
       auto new_i    = rtlib::clamp((int)(i+random_v.x),0,width-1);
       auto new_j    = rtlib::clamp((int)(j+random_v.y),0,height-1);
       outBuffer[j*width+i] = inBuffer[new_j*width+new_i];
   }
};
extern "C" __global__ void perlineKernel(uchar4 * outBuffer, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < width && j < height) {
        float noise1 = Perline::Generate(i / (float)64, j / (float)64, 0.0f);
        float noise2 = Perline::Generate(i / (float)32, j / (float)32, 0.0f);
        float noise3 = Perline::Generate(i / (float)16, j / (float)16, 0.0f);
        float noise4 = Perline::Generate(i / (float)8 , j / (float)8, 0.0f);
        float noise  = (noise1 +1.0f)/2.0f;
        outBuffer[j * width + i] = make_uchar4(noise*255.0f, noise * 255.0f, noise * 255.0f,255);
    }
}