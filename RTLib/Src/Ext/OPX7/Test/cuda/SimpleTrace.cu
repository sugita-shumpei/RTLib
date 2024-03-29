#define __CUDACC__
#include "SimpleKernel.h"
struct HitRecordUserData
{
    float3       radiance;
    float3       throughPut;
    float3       bsdfVal;
    float        bsdfPdf;
    unsigned int depth;
};
struct HitRecordUserDebugData
{
    float3 diffuse;
    float3 specular;
    float3 emission;
    float3 gridValue;
    unsigned int  gridIndex;
    float  shinness;
    float  refrIndx;
};
extern "C" {
    __constant__ Params params;
}
template<typename RNG>
static __forceinline__ __device__ float3       sampleCosinePDF(const float3& normal, RNG& rng)
{
    RTLib::Ext::CUDA::Math::ONB onb(normal);
    return onb.local(RTLib::Ext::CUDA::Math::random_cosine_direction(rng));
}
template<typename RNG>
static __forceinline__ __device__ float3       samplePhongPDF(const float3& reflectDir, float shinness, RNG& rng)
{
    RTLib::Ext::CUDA::Math::ONB onb(reflectDir);
    const auto cosTht = powf(RTLib::Ext::CUDA::Math::random_float1(0.0f, 1.0f, rng), 1.0f / (shinness + 1.0f));
    const auto sinTht = sqrtf(1.0f - cosTht * cosTht);
    const auto phi = RTLib::Ext::CUDA::Math::random_float1(0.0f, static_cast<float>(RTLIB_M_2PI), rng);
    return onb.local(make_float3(sinTht * cosf(phi), sinTht * sinf(phi), cosTht));
}
static __forceinline__ __device__ float        getValPhongPDF(const float3& direction, const float3& reflectDir, float shinness)
{

    const auto reflCos = RTLib::Ext::CUDA::Math::max(RTLib::Ext::CUDA::Math::dot(reflectDir, direction), 0.0f);
    return (shinness + 1.0f) * RTLib::Ext::CUDA::Math::powf(reflCos, shinness) * static_cast<float>(RTLIB_M_INV_2PI);
}
extern "C" __global__ void     __raygen__default () {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    auto* rgData    = reinterpret_cast<RayGenData*>(optixGetSbtDataPointer());
    const auto seed = params.seedBuffer[params.width * idx.y + idx.x];
    float3 result   = params.accumBuffer[params.width * idx.y + idx.x];
    BasicHitRecord<HitRecordUserData> hrec; 
    hrec.seed = seed;

    float3 color = make_float3(0.0f);
    RTLib::Ext::CUDA::Math::Xorshift32 xor32(hrec.seed);

    const auto gitter = RTLib::Ext::CUDA::Math::random_float2(xor32);
    const float2 d = make_float2(
        (2.0f * static_cast<float>(idx.x + gitter.x) / static_cast<float>(dim.x)) - 1.0,
        (2.0f * static_cast<float>(idx.y + gitter.y) / static_cast<float>(dim.y)) - 1.0);

    hrec.rayOrigin    = rgData->GetRayOrigin();
    hrec.rayDirection = rgData->GetRayDirection(d);
    hrec.rayDistance  = 0.0f;
    hrec.cosine       = 0.0f;
    hrec.seed         = xor32.m_seed;
    hrec.flags        = HIT_RECORD_FLAG_COUNT_EMITTED;

    hrec.userData.throughPut = make_float3(1.0f);
    hrec.userData.radiance   = make_float3(0.0f);
    hrec.userData.bsdfVal    = make_float3(0.0f);
    hrec.userData.bsdfPdf    = 0.0f;
    hrec.userData.depth      = 0;

    while (true) {
        TraceRadiance(params.gasHandle, hrec.rayOrigin, hrec.rayDirection, 0.0001f, 1.0e20f, hrec);

        color += hrec.userData.radiance;

        if (isnan(hrec.rayDirection.x)|| isnan(hrec.rayDirection.y)|| isnan(hrec.rayDirection.z)||
                isnan(hrec.userData.throughPut.x) || isnan(hrec.userData.throughPut.y) || isnan(hrec.userData.throughPut.z)||
            isnan(hrec.userData.radiance.x) || isnan(hrec.userData.radiance.y) || isnan(hrec.userData.radiance.z)) {
            /*printf("error\n");*/
            break;
        }
        if ((hrec.flags & HIT_RECORD_FLAG_FINISH) || (hrec.userData.depth >= (params.maxDepth - 1))) {
            break;
        }
        ++hrec.userData.depth;
    }
    result += color;
    // printf("%f, %lf\n", texCoord.x, texCoord.y);
    params.accumBuffer[params.width * idx.y + idx.x] = result;
    result =  result  / static_cast<float>(params.samplesForLaunch + params.samplesForAccum);
    result = (result) / (make_float3(1.0f) + result);
    params.frameBuffer[params.width * idx.y + idx.x] = make_uchar4(
        RTLib::Ext::CUDA::Math::min(static_cast<int>(RTLib::Ext::CUDA::Math::linear_to_gamma(result.x) * 255.99f), 255),
        RTLib::Ext::CUDA::Math::min(static_cast<int>(RTLib::Ext::CUDA::Math::linear_to_gamma(result.y) * 255.99f), 255),
        RTLib::Ext::CUDA::Math::min(static_cast<int>(RTLib::Ext::CUDA::Math::linear_to_gamma(result.z) * 255.99f), 255),
        255
    );
    params.seedBuffer [params.width * idx.y + idx.x] = hrec.seed;
}
extern "C" __global__ void     __raygen__debug   () {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    auto* rgData = reinterpret_cast<RayGenData*>(optixGetSbtDataPointer());
    const auto seed = params.seedBuffer[params.width * idx.y + idx.x];
    BasicHitRecord<HitRecordUserDebugData> hrec;
    hrec.seed = seed;

    float3 color = make_float3(0.0f);
    RTLib::Ext::CUDA::Math::Xorshift32 xor32(hrec.seed);

    const auto gitter = RTLib::Ext::CUDA::Math::random_float2(xor32);
    const float2 d = make_float2(
        (2.0f * static_cast<float>(idx.x + gitter.x) / static_cast<float>(dim.x)) - 1.0,
        (2.0f * static_cast<float>(idx.y + gitter.y) / static_cast<float>(dim.y)) - 1.0);

    hrec.rayOrigin          = rgData->GetRayOrigin();
    hrec.rayDirection       = rgData->GetRayDirection(d);
    hrec.rayDistance        = 0.0f;
    hrec.cosine             = 0.0f;
    hrec.seed               = xor32.m_seed;
    hrec.userData.gridValue = make_float3(0.0f);
    hrec.userData.diffuse   = make_float3(1.0f);
    hrec.userData.emission  = make_float3(0.0f);
    hrec.userData.specular  = make_float3(0.0f);
    hrec.userData.shinness  = 0.0f;
    hrec.userData.refrIndx  = 0;

    TraceRadiance(params.gasHandle, hrec.rayOrigin, hrec.rayDirection, 0.0001f, 1.0e20f, hrec);
    // printf("%f, %lf\n", texCoord.x, texCoord.y);
    //float3 result = make_float3(hrec.userData.gridValue /static_cast<float>(params.samplesForAccum));
    if (params.debugFrameType == DEBUG_FRAME_TYPE_NORMAL)
    {
        auto color = (hrec.normal + make_float3(1.0f)) / 2.0f;
        params.frameBuffer[params.width * idx.y + idx.x] = make_uchar4(
            RTLib::Ext::CUDA::Math::min(static_cast<int>(RTLib::Ext::CUDA::Math::linear_to_gamma(color.x) * 255.99f), 255),
            RTLib::Ext::CUDA::Math::min(static_cast<int>(RTLib::Ext::CUDA::Math::linear_to_gamma(color.y) * 255.99f), 255),
            RTLib::Ext::CUDA::Math::min(static_cast<int>(RTLib::Ext::CUDA::Math::linear_to_gamma(color.z) * 255.99f), 255),
            255
        );
    }
    if (params.debugFrameType == DEBUG_FRAME_TYPE_DEPTH)
    {
        auto color = make_float3(hrec.rayDistance / (1.0f + hrec.rayDistance));
        params.frameBuffer[params.width * idx.y + idx.x] = make_uchar4(
            RTLib::Ext::CUDA::Math::min(static_cast<int>(RTLib::Ext::CUDA::Math::linear_to_gamma(color.x) * 255.99f), 255),
            RTLib::Ext::CUDA::Math::min(static_cast<int>(RTLib::Ext::CUDA::Math::linear_to_gamma(color.y) * 255.99f), 255),
            RTLib::Ext::CUDA::Math::min(static_cast<int>(RTLib::Ext::CUDA::Math::linear_to_gamma(color.z) * 255.99f), 255),
            255
        );
    }
    if (params.debugFrameType == DEBUG_FRAME_TYPE_DIFFUSE)
    {
        params.frameBuffer[params.width * idx.y + idx.x] = make_uchar4(
            RTLib::Ext::CUDA::Math::min(static_cast<int>(RTLib::Ext::CUDA::Math::linear_to_gamma(hrec.userData.diffuse.x) * 255.99f), 255),
            RTLib::Ext::CUDA::Math::min(static_cast<int>(RTLib::Ext::CUDA::Math::linear_to_gamma(hrec.userData.diffuse.y) * 255.99f), 255),
            RTLib::Ext::CUDA::Math::min(static_cast<int>(RTLib::Ext::CUDA::Math::linear_to_gamma(hrec.userData.diffuse.z) * 255.99f), 255),
            255
        );
    }
    if (params.debugFrameType == DEBUG_FRAME_TYPE_SPECULAR)
    {
        params.frameBuffer[params.width * idx.y + idx.x] = make_uchar4(
            RTLib::Ext::CUDA::Math::min(static_cast<int>(RTLib::Ext::CUDA::Math::linear_to_gamma(hrec.userData.specular.x) * 255.99f), 255),
            RTLib::Ext::CUDA::Math::min(static_cast<int>(RTLib::Ext::CUDA::Math::linear_to_gamma(hrec.userData.specular.y) * 255.99f), 255),
            RTLib::Ext::CUDA::Math::min(static_cast<int>(RTLib::Ext::CUDA::Math::linear_to_gamma(hrec.userData.specular.z) * 255.99f), 255),
            255
        );
    }
    if (params.debugFrameType == DEBUG_FRAME_TYPE_EMISSION)
    {
        params.frameBuffer[params.width * idx.y + idx.x] = make_uchar4(
            RTLib::Ext::CUDA::Math::min(static_cast<int>(RTLib::Ext::CUDA::Math::linear_to_gamma(hrec.userData.emission.x) * 255.99f), 255),
            RTLib::Ext::CUDA::Math::min(static_cast<int>(RTLib::Ext::CUDA::Math::linear_to_gamma(hrec.userData.emission.y) * 255.99f), 255),
            RTLib::Ext::CUDA::Math::min(static_cast<int>(RTLib::Ext::CUDA::Math::linear_to_gamma(hrec.userData.emission.z) * 255.99f), 255),
            255
        );
    }
    if (params.debugFrameType == DEBUG_FRAME_TYPE_SHINNESS)
    {
        auto color = make_float3(hrec.userData.shinness / (1.0f + hrec.userData.shinness));
        params.frameBuffer[params.width * idx.y + idx.x] = make_uchar4(
            RTLib::Ext::CUDA::Math::min(static_cast<int>(RTLib::Ext::CUDA::Math::linear_to_gamma(color.x) * 255.99f), 255),
            RTLib::Ext::CUDA::Math::min(static_cast<int>(RTLib::Ext::CUDA::Math::linear_to_gamma(color.y) * 255.99f), 255),
            RTLib::Ext::CUDA::Math::min(static_cast<int>(RTLib::Ext::CUDA::Math::linear_to_gamma(color.z) * 255.99f), 255),
            255
        );
    }
    if (params.debugFrameType == DEBUG_FRAME_TYPE_REFR_INDEX)
    {
        auto color = make_float3(hrec.userData.refrIndx / (1.0f + hrec.userData.refrIndx));
        params.frameBuffer[params.width * idx.y + idx.x] = make_uchar4(
            RTLib::Ext::CUDA::Math::min(static_cast<int>(RTLib::Ext::CUDA::Math::linear_to_gamma(color.x) * 255.99f), 255),
            RTLib::Ext::CUDA::Math::min(static_cast<int>(RTLib::Ext::CUDA::Math::linear_to_gamma(color.y) * 255.99f), 255),
            RTLib::Ext::CUDA::Math::min(static_cast<int>(RTLib::Ext::CUDA::Math::linear_to_gamma(color.z) * 255.99f), 255),
            255
        );
    }
    if (params.debugFrameType == DEBUG_FRAME_TYPE_GRID_VALUE)
    {

        params.frameBuffer[params.width * idx.y + idx.x] = make_uchar4(
            RTLib::Ext::CUDA::Math::min(static_cast<int>(RTLib::Ext::CUDA::Math::linear_to_gamma(hrec.userData.gridValue.x) * 255.99f), 255),
            RTLib::Ext::CUDA::Math::min(static_cast<int>(RTLib::Ext::CUDA::Math::linear_to_gamma(hrec.userData.gridValue.y) * 255.99f), 255),
            RTLib::Ext::CUDA::Math::min(static_cast<int>(RTLib::Ext::CUDA::Math::linear_to_gamma(hrec.userData.gridValue.z) * 255.99f), 255),
            255
        );
    }
    params.seedBuffer[params.width * idx.y + idx.x] = hrec.seed;
}
extern "C" __global__ void       __miss__radiance() {
    auto* hrec = BasicHitRecord<HitRecordUserData>::GetGlobalPointer();
    auto* msData = reinterpret_cast<MissData*>(optixGetSbtDataPointer());

    hrec->SetGlobalRayOrigin(optixGetWorldRayOrigin());
    hrec->SetGlobalRayDirAndTmax(make_float4(optixGetWorldRayDirection(), optixGetRayTmax()));

    hrec->cosine = 0.0f;
    hrec->flags |= HIT_RECORD_FLAG_FINISH;

    hrec->userData.radiance = hrec->userData.throughPut * make_float3(msData->bgColor.x, msData->bgColor.y, msData->bgColor.z);
    hrec->userData.bsdfVal  = make_float3(1.0f);
}
extern "C" __global__ void __closesthit__radiance() {
    auto* hrec   = BasicHitRecord<HitRecordUserData>::GetGlobalPointer();
    auto* hgData = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    auto primitiveId = optixGetPrimitiveIndex();
    auto uv = optixGetTriangleBarycentrics();

    auto distance  = optixGetRayTmax();
    auto texCrd    = hgData->GetTexCrd (uv,primitiveId);
    auto fNormal   = hgData->GetTriangleFNormal(uv,primitiveId);
    auto vNormal   = hgData->GetTriangleVNormal(uv,primitiveId);
    auto inDir     = optixGetWorldRayDirection();
    auto position  = optixGetWorldRayOrigin() + distance * inDir;

    auto diffuse   = hgData->SampleDiffuse(texCrd);
    auto specular  = hgData->SampleSpecular(texCrd);
    auto emission  = hgData->SampleEmission(texCrd);
    auto refIndex = hgData->refIndex;
    auto shinness = hgData->shinness;

    auto xor32 = RTLib::Ext::CUDA::Math::Xorshift32(hrec->seed);

    auto direction      = make_float3(0.0f);
    auto cosine         = float(0.0f);
    auto bsdfVal        = make_float3(0.0f);
    auto bsdfPdf        = float(0.0f);
    auto radiance       = make_float3(0.0f);
    auto prevThroughput = hrec->userData.throughPut;
    auto currThroughput = make_float3(0.0f);
    auto prevHitFlags = hrec->flags;
    auto currHitFlags = static_cast<unsigned int>(0);
    const auto countEmitted = ((prevHitFlags & HIT_RECORD_FLAG_COUNT_EMITTED) == HIT_RECORD_FLAG_COUNT_EMITTED) || (hgData->type == HIT_GROUP_TYPE_DEF_LIGHT);
    //FOR DEBUG
    if (params.flags & PARAM_FLAG_USE_GRID) {
        unsigned int gridIndex = params.grid.FindFromCur(position);
        if (gridIndex != UINT32_MAX) {
            auto& val = params.diffuseGridBuffer[gridIndex];
            atomicAdd(&val.x, diffuse.x);
            atomicAdd(&val.y, diffuse.y);
            atomicAdd(&val.z, diffuse.z);
            atomicAdd(&val.w, 1.0f);
        }
    }

    do{
        if (hgData->type == HIT_GROUP_TYPE_PHONG) {
            auto reflDir     = RTLib::Ext::CUDA::Math::normalize(RTLib::Ext::CUDA::Math::reflect(inDir, fNormal));
            auto direction0  = sampleCosinePDF(fNormal, xor32);
            auto direction1  = samplePhongPDF(reflDir, shinness, xor32);
            auto cosine0     = RTLib::Ext::CUDA::Math::dot(direction0, fNormal);
            auto cosine1     = RTLib::Ext::CUDA::Math::dot(direction1, fNormal);
            auto cosinePdf0  = RTLib::Ext::CUDA::Math::max(cosine0 * static_cast<float>(RTLIB_M_INV_PI), 0.0f);
            auto cosinePdf1  = RTLib::Ext::CUDA::Math::max(cosine1 * static_cast<float>(RTLIB_M_INV_PI), 0.0f);
            auto phongPdf0   = getValPhongPDF(direction0, reflDir, shinness);
            auto phongPdf1   = getValPhongPDF(direction1, reflDir, shinness);
            auto aver_diff   = 1.0e-10f+(diffuse.x + diffuse.y + diffuse.z) / 3.0f;
            auto aver_spec   = 1.0e-10f+(specular.x + specular.y + specular.z) / 3.0f;
            auto select_prob = (aver_diff) / (aver_diff + aver_spec);
            if (RTLib::Ext::CUDA::Math::random_float1(xor32) < select_prob) {
                auto reflCos = RTLib::Ext::CUDA::Math::dot(reflDir, direction0);
                direction = direction0;
                cosine = cosine0;
                bsdfVal = diffuse * static_cast<float>(RTLIB_M_INV_PI) + specular * ((shinness+2.0f)/(shinness+1.0f)) * phongPdf0;
                bsdfPdf = (select_prob * cosinePdf0 + (1.0f - select_prob) * phongPdf0);
            }
            else {
                auto reflCos = RTLib::Ext::CUDA::Math::dot(reflDir, direction1);
                direction = direction1;
                cosine = cosine1;
                bsdfVal = diffuse * static_cast<float>(RTLIB_M_INV_PI) + specular * ((shinness+2.0f)/(shinness+1.0f)) * phongPdf1;
                bsdfPdf = (select_prob * cosinePdf1 + (1.0f - select_prob) * phongPdf1);
            }
            currThroughput = prevThroughput * ((bsdfPdf > 0.0f) ? (bsdfVal * RTLib::Ext::CUDA::Math::max(cosine, 0.0f) / bsdfPdf) : make_float3(0.0f));

            if (params.flags & PARAM_FLAG_NEE) {
                if (hrec->userData.depth < params.maxDepth - 1) {
                    if (params.flags & PARAM_FLAG_RIS) {
                        Reservoir<LightRecord> resv = {};
                        auto f_y    = make_float3(0.0f);
                        auto f_a_y  = 0.0f;
                        auto lDir_y = make_float3(0.0f);
                        auto dist_y = float(0.0f);
                        for (int i = 0; i < params.numCandidates; ++i) {
                            LightRecord lRec = params.lights.Sample(position, xor32);
                            auto  ndl   =  RTLib::Ext::CUDA::Math::dot(lRec.direction, fNormal    );
                            auto lndl   = -RTLib::Ext::CUDA::Math::dot(lRec.direction, lRec.normal);
                            auto  e    = lRec.emission;
                            auto  b    = diffuse * static_cast<float>(RTLIB_M_INV_PI) + specular * ((shinness+2.0f)/(shinness+1.0f)) *  getValPhongPDF(lRec.direction, reflDir, shinness);
                            auto  g     = RTLib::Ext::CUDA::Math::max(ndl, 0.0f) * RTLib::Ext::CUDA::Math::max(lndl, 0.0f) / (lRec.distance * lRec.distance);
                            auto  f    = b * e * g;
                            auto  f_a   = RTLib::Ext::CUDA::Math::to_average_rgb(f);
                            if (resv.Update(lRec, f_a * lRec.invPdf, RTLib::Ext::CUDA::Math::random_float1(xor32))) {
                                f_y    = f;
                                f_a_y  = f_a;
                                lDir_y = lRec.direction;
                                dist_y = lRec.distance;
                            }
                        }
                        if (resv.w_sum > 0.0f && f_a_y > 0.0f) {
                            if (!TraceOccluded(params.gasHandle, position, lDir_y, 0.0001f, dist_y - 0.0001f)) {
                                resv.w = resv.w_sum / (f_a_y * static_cast<float>(resv.m));
                            }
                        }
                        radiance += prevThroughput * f_y * resv.w;
                    }
                    else {
                        auto lRec = params.lights.Sample(position, xor32);
                        if (!TraceOccluded(params.gasHandle, position, lRec.direction, 0.0001f, lRec.distance - 0.0001f)) {
                            auto e = lRec.emission;
                            auto b = diffuse * static_cast<float>(RTLIB_M_INV_PI) + specular * ((shinness+2.0f)/(shinness+1.0f)) * getValPhongPDF(lRec.direction, reflDir, shinness);
                            auto g = RTLib::Ext::CUDA::Math::max(-RTLib::Ext::CUDA::Math::dot(lRec.direction, lRec.normal), 0.0f) * fabsf(RTLib::Ext::CUDA::Math::dot(lRec.direction, fNormal)) / (lRec.distance * lRec.distance);
                            radiance += prevThroughput * b * e * g * lRec.invPdf;
                        }
                    }
                }
            }
            else {
                currHitFlags |= HIT_RECORD_FLAG_COUNT_EMITTED;
            }
            break;
        }
        if (hgData->type == HIT_GROUP_TYPE_GLASS) {
            float3 rNormal = {};
            float  rRefIdx = 0.0f;
            float cosine_i = RTLib::Ext::CUDA::Math::dot(vNormal, inDir);
            if (cosine_i < 0.0f) {
                rNormal  = vNormal;
                rRefIdx  = 1.0f / refIndex; 
                cosine_i =-cosine_i;
            }
            else {
                rNormal  = make_float3(-vNormal.x, -vNormal.y, -vNormal.z); 
                rRefIdx  = refIndex;

            }
            auto sine_o_2 = (1.0f - RTLib::Ext::CUDA::Math::pow2(cosine_i)) * RTLib::Ext::CUDA::Math::pow2(rRefIdx);
            auto fresnell = 0.0f;
            {
                //float cosine_o = sqrtf(RTLib::Ext::CUDA::Math::max(1.0f - sine_o_2, 0.0f));
                //float r_p = (cosine_i - rRefIdx * cosine_o) / (cosine_i + rRefIdx * cosine_o);
                //float r_s = (rRefIdx * cosine_i - cosine_o) / (rRefIdx * cosine_i + cosine_o);
                //fresnell = (r_p * r_p + r_s * r_s) / 2.0f;

                float  f0 = RTLib::Ext::CUDA::Math::pow2((1 - rRefIdx) /   (1    + rRefIdx));
                fresnell  = f0 + (1.0f - f0) * RTLib::Ext::CUDA::Math::pow5(1.0f - cosine_i);
            }
            auto reflDir = RTLib::Ext::CUDA::Math::normalize(RTLib::Ext::CUDA::Math::reflect(inDir, rNormal));
            if (RTLib::Ext::CUDA::Math::random_float1(0.0f, 1.0f, xor32)<fresnell || sine_o_2 > 1.0f) {
                position      += 0.0001f * rNormal;
                direction      = reflDir;
                cosine         = fabsf(cosine_i);
                bsdfVal        = specular;
                bsdfPdf        = 0.0f;
                /*currThroughput = prevThroughput * specular ;*/
                currThroughput = prevThroughput * specular;
            }
            else {
                position       -= 0.0001f * rNormal;
                float  sine_i_2 = RTLib::Ext::CUDA::Math::max(1.0f - cosine_i * cosine_i,0.0f);
                float  cosine_o = sqrtf(1.0f - sine_o_2);
                float3 refrDir  = make_float3(0.0f);
                if (sine_i_2 > 0.0f) {
                    float3 k = (inDir + cosine_i * rNormal) / sqrtf(sine_i_2);
                    refrDir  = RTLib::Ext::CUDA::Math::normalize(sqrtf(sine_o_2) * k - cosine_o * rNormal);
                }
                else {
                    refrDir  = inDir;
                }
                direction       = refrDir;
                cosine          = fabsf(RTLib::Ext::CUDA::Math::dot(refrDir, rNormal));
                bsdfVal         = make_float3(1.0f);
                bsdfPdf         = 0.0f;
                /*currThroughput  = prevThroughput;*/
                currThroughput  = prevThroughput ;
            }
            if (isnan(direction.x) || isnan(direction.y) || isnan(direction.z)) {
                printf("IOR: %lf Cos: %lf IDir: (%lf %lf %lf) Norm: (%lf %lf %lf) Refl: (%lf %lf %lf) ODir: (%lf %lf %lf) fresnell=%lf\n", rRefIdx, cosine_i, inDir.x, inDir.y, inDir.z, rNormal.x, rNormal.y, rNormal.z, reflDir.x, reflDir.y, reflDir.z, direction.x, direction.y, direction.z, fresnell);
            }
            currHitFlags |= HIT_RECORD_FLAG_COUNT_EMITTED;
            currHitFlags |= HIT_RECORD_FLAG_DELTA_MATERIAL;
            break;
        }
    } while (0);

    radiance += prevThroughput * emission * static_cast<float>(RTLib::Ext::CUDA::Math::dot(inDir, fNormal) < 0.0f) * static_cast<float>(countEmitted);
    if (emission.x + emission.y + emission.z > 0.0f) {
        currHitFlags |= HIT_RECORD_FLAG_FINISH;
    }

    hrec->SetGlobalRayOrigin(position);
    hrec->SetGlobalRayDirAndTmax(make_float4(direction, distance));
    hrec->normal = fNormal;
    hrec->seed   = xor32.m_seed;
    hrec->cosine = cosine;
    hrec->flags  = currHitFlags;
    hrec->userData.radiance   = radiance;
    hrec->userData.throughPut = currThroughput;
    hrec->userData.bsdfVal    = bsdfVal;
    hrec->userData.bsdfPdf    = bsdfPdf;
}
extern "C" __global__ void __closesthit__radiance_sphere() {
    auto* hrec = BasicHitRecord<HitRecordUserData>::GetGlobalPointer();
    auto* hgData = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    auto primitiveId = optixGetPrimitiveIndex();
    auto uv = optixGetTriangleBarycentrics();

    auto distance = optixGetRayTmax();
    auto inDir    = optixGetWorldRayDirection();
    auto position = optixGetWorldRayOrigin() + distance * inDir;
    auto fNormal  = hgData->GetSphereNormal(position, primitiveId);
    auto vNormal  = fNormal;

    auto diffuse  = hgData->diffuse ;
    auto specular = hgData->specular;
    auto emission = hgData->emission;
    auto refIndex = hgData->refIndex;
    auto shinness = hgData->shinness;

    auto xor32 = RTLib::Ext::CUDA::Math::Xorshift32(hrec->seed);

    auto direction = make_float3(0.0f);
    auto cosine = float(0.0f);
    auto bsdfVal = make_float3(0.0f);
    auto bsdfPdf = float(0.0f);
    auto radiance = make_float3(0.0f);
    auto prevThroughput = hrec->userData.throughPut;
    auto currThroughput = make_float3(0.0f);
    auto prevHitFlags = hrec->flags;
    auto currHitFlags = static_cast<unsigned int>(0);
    const auto countEmitted = ((prevHitFlags & HIT_RECORD_FLAG_COUNT_EMITTED) == HIT_RECORD_FLAG_COUNT_EMITTED) || (hgData->type == HIT_GROUP_TYPE_DEF_LIGHT);

    if (params.flags & PARAM_FLAG_USE_GRID) {
        unsigned int gridIndex = params.grid.FindFromCur(position);
        if (gridIndex != UINT32_MAX) {
            auto& val = params.diffuseGridBuffer[gridIndex];
            atomicAdd(&val.x, diffuse.x);
            atomicAdd(&val.y, diffuse.y);
            atomicAdd(&val.z, diffuse.z);
            atomicAdd(&val.w, 1.0f);
        }
    }

    do {
        if (hgData->type == HIT_GROUP_TYPE_PHONG) {
            auto reflDir = RTLib::Ext::CUDA::Math::normalize(RTLib::Ext::CUDA::Math::reflect(inDir, fNormal));
            auto direction0 = sampleCosinePDF(fNormal, xor32);
            auto direction1 = samplePhongPDF(reflDir, shinness, xor32);
            auto cosine0 = RTLib::Ext::CUDA::Math::dot(direction0, fNormal);
            auto cosine1 = RTLib::Ext::CUDA::Math::dot(direction1, fNormal);
            auto cosinePdf0 = RTLib::Ext::CUDA::Math::max(cosine0 * static_cast<float>(RTLIB_M_INV_PI), 0.0f);
            auto cosinePdf1 = RTLib::Ext::CUDA::Math::max(cosine1 * static_cast<float>(RTLIB_M_INV_PI), 0.0f);
            auto phongPdf0 = getValPhongPDF(direction0, reflDir, shinness);
            auto phongPdf1 = getValPhongPDF(direction1, reflDir, shinness);
            auto aver_diff = 1.0e-10f + (diffuse.x + diffuse.y + diffuse.z) / 3.0f;
            auto aver_spec = 1.0e-10f + (specular.x + specular.y + specular.z) / 3.0f;
            auto select_prob = (aver_diff) / (aver_diff + aver_spec);
            if (RTLib::Ext::CUDA::Math::random_float1(xor32) < select_prob) {
                auto reflCos = RTLib::Ext::CUDA::Math::dot(reflDir, direction0);
                direction = direction0;
                cosine = cosine0;
                bsdfVal = diffuse * static_cast<float>(RTLIB_M_INV_PI) + specular * ((shinness + 2.0f) / (shinness + 1.0f)) * phongPdf0;
                bsdfPdf = (select_prob * cosinePdf0 + (1.0f - select_prob) * phongPdf0);
            }
            else {
                auto reflCos = RTLib::Ext::CUDA::Math::dot(reflDir, direction1);
                direction = direction1;
                cosine = cosine1;
                bsdfVal = diffuse * static_cast<float>(RTLIB_M_INV_PI) + specular * ((shinness + 2.0f) / (shinness + 1.0f)) * phongPdf1;
                bsdfPdf = (select_prob * cosinePdf1 + (1.0f - select_prob) * phongPdf1);
            }
            currThroughput = prevThroughput * ((bsdfPdf > 0.0f) ? (bsdfVal * RTLib::Ext::CUDA::Math::max(cosine, 0.0f) / bsdfPdf) : make_float3(0.0f));

            if (params.flags & PARAM_FLAG_NEE) {
                if (hrec->userData.depth < params.maxDepth - 1) {
                    if (params.flags & PARAM_FLAG_RIS) {
                        Reservoir<LightRecord> resv = {};
                        auto f_y = make_float3(0.0f);
                        auto f_a_y = 0.0f;
                        auto lDir_y = make_float3(0.0f);
                        auto dist_y = float(0.0f);
                        for (int i = 0; i < params.numCandidates; ++i) {
                            LightRecord lRec = params.lights.Sample(position, xor32);
                            auto  ndl = RTLib::Ext::CUDA::Math::dot(lRec.direction, fNormal);
                            auto lndl = -RTLib::Ext::CUDA::Math::dot(lRec.direction, lRec.normal);
                            auto  e = lRec.emission;
                            auto  b = diffuse * static_cast<float>(RTLIB_M_INV_PI) + specular * ((shinness + 2.0f) / (shinness + 1.0f)) * getValPhongPDF(lRec.direction, reflDir, shinness);
                            auto  g = RTLib::Ext::CUDA::Math::max(ndl, 0.0f) * RTLib::Ext::CUDA::Math::max(lndl, 0.0f) / (lRec.distance * lRec.distance);
                            auto  f = b * e * g;
                            auto  f_a = RTLib::Ext::CUDA::Math::to_average_rgb(f);
                            if (resv.Update(lRec, f_a * lRec.invPdf, RTLib::Ext::CUDA::Math::random_float1(xor32))) {
                                f_y = f;
                                f_a_y = f_a;
                                lDir_y = lRec.direction;
                                dist_y = lRec.distance;
                            }
                        }
                        if (resv.w_sum > 0.0f && f_a_y > 0.0f) {
                            if (!TraceOccluded(params.gasHandle, position, lDir_y, 0.0001f, dist_y - 0.0001f)) {
                                resv.w = resv.w_sum / (f_a_y * static_cast<float>(resv.m));
                            }
                        }
                        radiance += prevThroughput * f_y * resv.w;
                    }
                    else {
                        auto lRec = params.lights.Sample(position, xor32);
                        if (!TraceOccluded(params.gasHandle, position, lRec.direction, 0.0001f, lRec.distance - 0.0001f)) {
                            auto e = lRec.emission;
                            auto b = diffuse * static_cast<float>(RTLIB_M_INV_PI) + specular * ((shinness + 2.0f) / (shinness + 1.0f)) * getValPhongPDF(lRec.direction, reflDir, shinness);
                            auto g = RTLib::Ext::CUDA::Math::max(-RTLib::Ext::CUDA::Math::dot(lRec.direction, lRec.normal), 0.0f) * fabsf(RTLib::Ext::CUDA::Math::dot(lRec.direction, fNormal)) / (lRec.distance * lRec.distance);
                            radiance += prevThroughput * b * e * g * lRec.invPdf;
                        }
                    }
                }
            }
            else {
                currHitFlags |= HIT_RECORD_FLAG_COUNT_EMITTED;
            }
            break;
        }
        if (hgData->type == HIT_GROUP_TYPE_GLASS) {
            float3 rNormal = {};
            float  rRefIdx = 0.0f;
            float cosine_i = RTLib::Ext::CUDA::Math::dot(fNormal, inDir);
            if (cosine_i < 0.0f) {
                rNormal = fNormal;
                rRefIdx = 1.0f / refIndex;
                cosine_i = -cosine_i;
            }
            else {
                rNormal = make_float3(-fNormal.x, -fNormal.y, -fNormal.z);
                rRefIdx = refIndex;

            }
            auto sine_o_2 = (1.0f - RTLib::Ext::CUDA::Math::pow2(cosine_i)) * RTLib::Ext::CUDA::Math::pow2(rRefIdx);
            auto fresnell = 0.0f;
            {
                //float cosine_o = sqrtf(RTLib::Ext::CUDA::Math::max(1.0f - sine_o_2, 0.0f));
                //float r_p = (cosine_i - rRefIdx * cosine_o) / (cosine_i + rRefIdx * cosine_o);
                //float r_s = (rRefIdx * cosine_i - cosine_o) / (rRefIdx * cosine_i + cosine_o);
                //fresnell = (r_p * r_p + r_s * r_s) / 2.0f;

                float  f0 = RTLib::Ext::CUDA::Math::pow2((1 - rRefIdx) / (1 + rRefIdx));
                fresnell = f0 + (1.0f - f0) * RTLib::Ext::CUDA::Math::pow5(1.0f - cosine_i);
            }
            auto reflDir = RTLib::Ext::CUDA::Math::normalize(RTLib::Ext::CUDA::Math::reflect(inDir, rNormal));
            if (RTLib::Ext::CUDA::Math::random_float1(0.0f, 1.0f, xor32) < fresnell || sine_o_2 > 1.0f) {
                position += 0.0001f * rNormal;
                direction = reflDir;
                cosine = fabsf(cosine_i);
                bsdfVal = specular;
                bsdfPdf = 0.0f;
                /*currThroughput = prevThroughput * specular ;*/
                currThroughput = prevThroughput * specular;
            }
            else {
                position -= 0.0001f * rNormal;
                float  sine_i_2 = RTLib::Ext::CUDA::Math::max(1.0f - cosine_i * cosine_i, 0.0f);
                float  cosine_o = sqrtf(1.0f - sine_o_2);
                float3 refrDir = make_float3(0.0f);
                if (sine_i_2 > 0.0f) {
                    float3 k = (inDir + cosine_i * rNormal) / sqrtf(sine_i_2);
                    refrDir = RTLib::Ext::CUDA::Math::normalize(sqrtf(sine_o_2) * k - cosine_o * rNormal);
                }
                else {
                    refrDir = inDir;
                }
                direction = refrDir;
                cosine = fabsf(RTLib::Ext::CUDA::Math::dot(refrDir, rNormal));
                bsdfVal = make_float3(1.0f);
                bsdfPdf = 0.0f;
                /*currThroughput  = prevThroughput;*/
                currThroughput = prevThroughput;
            }
            if (isnan(direction.x) || isnan(direction.y) || isnan(direction.z)) {
                printf("IOR: %lf Cos: %lf IDir: (%lf %lf %lf) Norm: (%lf %lf %lf) Refl: (%lf %lf %lf) ODir: (%lf %lf %lf) fresnell=%lf\n", rRefIdx, cosine_i, inDir.x, inDir.y, inDir.z, rNormal.x, rNormal.y, rNormal.z, reflDir.x, reflDir.y, reflDir.z, direction.x, direction.y, direction.z, fresnell);
            }
            currHitFlags |= HIT_RECORD_FLAG_COUNT_EMITTED;
            currHitFlags |= HIT_RECORD_FLAG_DELTA_MATERIAL;
            break;
        }
    } while (0);

    radiance += prevThroughput * emission * static_cast<float>(RTLib::Ext::CUDA::Math::dot(inDir, fNormal) < 0.0f) * static_cast<float>(countEmitted);

    if (emission.x + emission.y + emission.z > 0.0f) {
        currHitFlags |= HIT_RECORD_FLAG_FINISH;
    }

    hrec->SetGlobalRayOrigin(position);
    hrec->SetGlobalRayDirAndTmax(make_float4(direction, distance));
    hrec->normal = fNormal;
    hrec->seed = xor32.m_seed;
    hrec->cosine = cosine;
    hrec->flags = currHitFlags;
    hrec->userData.radiance = radiance;
    hrec->userData.throughPut = currThroughput;
    hrec->userData.bsdfVal = bsdfVal;
    hrec->userData.bsdfPdf = bsdfPdf;
}
extern "C" __global__ void       __miss__occluded() {
    optixSetPayload_0(false);
}
extern "C" __global__ void __closesthit__occluded() {
    optixSetPayload_0(true);
}
extern "C" __global__ void    __closesthit__debug() {
    auto* hrec = BasicHitRecord<HitRecordUserDebugData>::GetGlobalPointer();
    auto* hgData = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    auto primitiveId = optixGetPrimitiveIndex();
    auto uv = optixGetTriangleBarycentrics();

    auto distance  = optixGetRayTmax();
    auto texCrd   = hgData->GetTexCrd(uv, primitiveId);
    auto normal   = hgData->GetTriangleFNormal(uv, primitiveId);
    auto position = optixGetWorldRayOrigin() + distance * optixGetWorldRayDirection();
    auto reflDir  = RTLib::Ext::CUDA::Math::normalize(RTLib::Ext::CUDA::Math::reflect(optixGetWorldRayDirection(), normal));

    auto  direction = make_float3(0.0f);

    hrec->SetGlobalRayOrigin(position);
    hrec->SetGlobalRayDirAndTmax(make_float4(direction, distance));
    hrec->normal = normal;
    hrec->cosine = 0.0f;
    hrec->flags  = 0;

    if (params.flags & PARAM_FLAG_USE_GRID)
    {
        float3 gridIndexF = (position - params.grid.aabbOffset) / params.grid.aabbSize;
        hrec->userData.gridIndex = params.grid.FindFromPrv(position);
        if (hrec->userData.gridIndex != UINT32_MAX) {
            //auto gridValue = params.diffuseGridBuffer[hrec->userData.gridIndex];
            auto gridValue = make_float3(params.mortonTree.GetBuildingTree(hrec->userData.gridIndex).weights[0]);
            hrec->userData.gridValue = gridValue;
        }
        else {
            hrec->userData.gridIndex = UINT32_MAX;
            hrec->userData.gridValue = make_float3(0.0f);
        }

    }
    else {
        hrec->userData.gridValue = make_float3(0.0f);
    }
    hrec->userData.diffuse   = hgData->SampleDiffuse(texCrd);
    hrec->userData.specular  = hgData->SampleSpecular(texCrd);
    hrec->userData.emission  = hgData->SampleEmission(texCrd);
    hrec->userData.shinness  = hgData->shinness;
    hrec->userData.refrIndx  = hgData->refIndex;
}
extern "C" __global__ void    __closesthit__debug_sphere() {
    auto* hrec = BasicHitRecord<HitRecordUserDebugData>::GetGlobalPointer();
    auto* hgData = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    auto primitiveId = optixGetPrimitiveIndex();
    auto uv = optixGetTriangleBarycentrics();

    auto distance  = optixGetRayTmax();
    auto position = optixGetWorldRayOrigin() + distance * optixGetWorldRayDirection();
    auto normal   = hgData->GetSphereNormal(position, primitiveId);
    auto reflDir  = RTLib::Ext::CUDA::Math::normalize(RTLib::Ext::CUDA::Math::reflect(optixGetWorldRayDirection(), normal));

    auto  direction = make_float3(0.0f);

    hrec->SetGlobalRayOrigin(position);
    hrec->SetGlobalRayDirAndTmax(make_float4(direction, distance));
    hrec->normal = normal;
    hrec->cosine = 0.0f;
    hrec->flags = 0;
    if (params.flags & PARAM_FLAG_USE_GRID)
    {

        float3 gridIndexF = (position - params.grid.aabbOffset) / params.grid.aabbSize;
        hrec->userData.gridIndex = params.grid.FindFromPrv(position);
        if (hrec->userData.gridIndex != UINT32_MAX) {
            //auto gridValue = params.diffuseGridBuffer[hrec->userData.gridIndex];
            auto gridValue = params.diffuseGridBuffer[hrec->userData.gridIndex];
            hrec->userData.gridValue = (gridValue.w > 0.0f) ? make_float3(gridValue.x, gridValue.y, gridValue.z) / gridValue.w : make_float3(0.0f);
        }
        else {
            hrec->userData.gridIndex = UINT32_MAX;
            hrec->userData.gridValue = make_float3(0.0f);
        }
    }
    else {
        hrec->userData.gridValue = make_float3(0.0f);
    }
    hrec->userData.diffuse  = hgData->diffuse;
    hrec->userData.specular = hgData->specular;
    hrec->userData.emission = hgData->emission;
    hrec->userData.shinness = hgData->shinness;
    hrec->userData.refrIndx = hgData->refIndex;
}
extern "C" __global__ void          __miss__debug(){
    auto* hrec = BasicHitRecord<HitRecordUserDebugData>::GetGlobalPointer();
    auto* msData = reinterpret_cast<MissData*>(optixGetSbtDataPointer());

    hrec->SetGlobalRayOrigin(optixGetWorldRayOrigin());
    hrec->SetGlobalRayDirAndTmax(make_float4(optixGetWorldRayDirection(), optixGetRayTmax()));
    hrec->normal = make_float3(1.0f,0.0f,0.0f);
    hrec->cosine = 0.0f;
    hrec->flags  = 0;
    hrec->userData.gridIndex = UINT32_MAX;
    hrec->userData.gridValue = make_float3(0.0f);
    hrec->userData.diffuse   = make_float3(0.0f);
    hrec->userData.specular  = make_float3(0.0f);
    hrec->userData.emission  = make_float3(0.0f);
    hrec->userData.shinness  = 0.0f;
    hrec->userData.refrIndx  = 0.0f;

}
extern "C" __global__ void           __anyhit__ah() {
    auto* hgData = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
}
extern "C" __global__ void        __exception__ep() {
    auto code = optixGetExceptionCode();
    if (code == OPTIX_EXCEPTION_CODE_TRAVERSAL_DEPTH_EXCEEDED)
    {
        printf("%d\n", optixGetTransformListSize());
    }
}