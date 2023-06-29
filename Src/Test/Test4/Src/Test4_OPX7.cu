#define __CUDACC__
#include <Test4_OPX7.h>
extern "C" {
	__constant__ Params params;
}
// p0, p1, p2: RG_R  | CH_W -> 0
// p3, p4, p5: RG_R  | CH_W -> 0
// p6, p7, p8: RG_RW | MS_W  | CH_W  ->1
// p9,p10.p11: RG_RW | MS_RW | CH_RW ->2
// p12       : RG_RW | CH_RW         ->3
// p13       : RG_RW | MS_W  | CH_W  ->4
// RG
// p0, p1, p2: R
// p3, p4, p5: R
// p6, p7, p8: RW
// p9,p10.p11: RW
// p12: RW
// p13: RW

static __forceinline__ __device__ void load_ray_origin(float3& rayOrigin){	       rayOrigin = get_payload_float3<0>();}
static __forceinline__ __device__ void store_ray_origin(const float3& rayOrigin){  set_payload_float3<0>(rayOrigin);   }

static __forceinline__ __device__ void load_ray_direction(float3& rayDirection) { rayDirection = get_payload_float3<3>(); }
static __forceinline__ __device__ void store_ray_direction(const float3& rayDirection) { set_payload_float3<3>(rayDirection); }

static __forceinline__ __device__ void load_radiance(float3& radiance) { radiance = get_payload_float3<6>(); }
static __forceinline__ __device__ void store_radiance(const float3& radiance) { set_payload_float3<6>(radiance); }

static __forceinline__ __device__ void load_attenuation(float3& attenuation) { attenuation = get_payload_float3<9>(); }
static __forceinline__ __device__ void store_attenuation(const float3& attenuation) { set_payload_float3<9>(attenuation); }

static __forceinline__ __device__ void load_seed(unsigned int& seed) { seed = get_payload_uint1<12>(); }
static __forceinline__ __device__ void store_seed(unsigned int  seed) { set_payload_uint1<12>(seed); }

static __forceinline__ __device__ void load_done(unsigned int& done) { done = get_payload_uint1<13>(); }
static __forceinline__ __device__ void store_done(unsigned int done) { set_payload_uint1<13>(done); }

static __forceinline__ __device__ void TraceRadiance(float3& rayOrigin, float3& rayDirection, float3& radiance, float3& attenuation, unsigned int& seed, unsigned int& done)
{
	unsigned int p0 , p1 , p2 ;// ray origin
	unsigned int p3 , p4 , p5 ;// ray direction
	unsigned int p6 , p7 , p8 ;// radiance
	unsigned int p9 , p10, p11;// attenuation 
	unsigned int p12, p13, p14;// seed done depth

	pack_float3(radiance, p6, p7, p8);
	pack_float3(attenuation, p9, p10, p11);
	p12 = seed;
	p13 = done;

	optixTrace(
		OPTIX_PAYLOAD_TYPE_ID_0,
		params.tlas,
		rayOrigin,
		rayDirection,
		0.0f,
		1e16f,
		0.0f,
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_NONE,
		0,
		1,
		0,
		p0, p1, p2,
		p3, p4, p5,
		p6, p7, p8,
		p9, p10, p11,
		p12, p13
	);

	rayOrigin    = unpack_float3(p0, p1, p2);
	rayDirection = unpack_float3(p3, p4, p5);
	radiance     = unpack_float3(p6, p7, p8);
	attenuation  = unpack_float3(p9, p10,p11);
	seed         = p12;
	done         = p13;
}

extern "C" __global__ void __raygen__Test4()
{
	using namespace otk;
	uint3 idx = optixGetLaunchIndex();
	uint3 dim = optixGetLaunchDimensions();
	unsigned int pixelIdx = params.width * idx.y + idx.x;

	constexpr int numSamples = 10;
	float3 result = make_float3(0.0f, 0.0f, 0.0f);

	unsigned int seed = params.seedbuffer[pixelIdx];

	for (unsigned int i = 0; i < params.samples; ++i)
	{
		auto d = 2.0f * make_float2(
			static_cast<float>(idx.x+ xorshift32_f32_01(seed)) / static_cast<float>(params.width ),
			static_cast<float>(idx.y+ xorshift32_f32_01(seed)) / static_cast<float>(params.height)
		) - 1.0f;

		auto rayOrigin    = params.camEye;
		auto rayDirection = params.generate_ray(d);
		auto radiance     = make_float3(0.0f,0.0f,0.0f);
		auto attenuation  = make_float3(1.0f,1.0f,1.0f);
		unsigned int done = 0;

		for (unsigned int j = 0; j < params.depth; ++j)
		{
			if (done) {
				break;
			}
			TraceRadiance(rayOrigin, rayDirection, radiance, attenuation, seed, done);
			result += radiance;
		}
	}

	float4 accumData   = params.accumbuffer[pixelIdx];
	float3 accumColor  = make_float3(accumData.x, accumData.y, accumData.z);
	float  accumSample = accumData.w;

	accumColor  += result;
	accumSample += (params.samples);
	params.accumbuffer[pixelIdx] = make_float4(accumColor.x, accumColor.y, accumColor.z, accumSample);

	params.seedbuffer[pixelIdx]  = seed;
	params.framebuffer[pixelIdx] = make_color(accumColor/accumSample);
}

// p6 p7  p8  -> MS W
// p9 p10 p11 -> MS RW
// p13        -> MS W
extern "C" __global__ void __miss__Test4()
{
	using namespace otk;
	optixSetPayloadTypes(OPTIX_PAYLOAD_TYPE_ID_0);

	float3 attenuation;
	load_attenuation(attenuation);

	unsigned int done  = 1;

	store_radiance(attenuation * params.bgColor);
	store_done(done);
}
// p0 p1  p2  -> CH W
// p3 p4  p5  -> CH W
// p6 p7  p8  -> CH W
// p9 p10 p11 -> CH RW
// p12        -> CH RW
// p13        -> CH W
extern "C" __global__ void __closesthit__Test4()
{
	using namespace otk;
	optixSetPayloadTypes(OPTIX_PAYLOAD_TYPE_ID_0);

	auto hgData = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());

	float3 attenuation;	
	unsigned int seed;
	
	load_attenuation(attenuation);
	load_seed(seed);

	auto vertices = get_triangle_data(0.0f);
	auto worldDir = optixGetWorldRayDirection();
	auto worldPos = optixGetWorldRayOrigin() + optixGetRayTmax() * worldDir;

	auto worldV0 = optixTransformPointFromWorldToObjectSpace(vertices.row0);
	auto worldV1 = optixTransformPointFromWorldToObjectSpace(vertices.row1);
	auto worldV2 = optixTransformPointFromWorldToObjectSpace(vertices.row2);
	auto worldVN = otk::normalize(otk::cross(worldV1 - worldV0, worldV2 - worldV0));

	auto diffuse  = make_float3(hgData->diffuse.x, hgData->diffuse.y, hgData->diffuse.z);
	auto emission = make_float3(hgData->emission.x, hgData->emission.y, hgData->emission.z);

	if (dot(worldVN, worldDir) > 0.0f) {
		worldVN = -worldVN;
	}

	auto w = worldVN;
	auto u = cross(make_float3(0, 0, 1), w);
	if (dot(w, make_float3(0, 0, 1)) > 0.01f) {
		 u = cross(make_float3(0, 1, 0), w);
	}

	auto v         = cross(w, u);
	auto dirCosine = xorshift32_cosine_distribution(seed);
	auto newDir    = normalize(dirCosine.x * u + dirCosine.y * v + dirCosine.z * w);

	float3 radiance = attenuation * emission;
	attenuation     = attenuation * diffuse * fmaxf(dirCosine.z, 0.0f) / M_1_PIf;

	unsigned int done = 0;
	if (dot(emission, make_float3(1, 1, 1)) > 0.0f) {
		done = 1;
	}

	store_ray_origin(worldPos + 0.01f * worldVN);
	store_ray_direction(newDir);
	store_radiance(radiance);
	store_attenuation(attenuation);
	store_seed(seed);
	store_done(done);
}