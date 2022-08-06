#ifndef RTLIB_EXT_OPX7_UTILS_OPX7_UTILS_PATH_GUIDING_H
#define RTLIB_EXT_OPX7_UTILS_OPX7_UTILS_PATH_GUIDING_H
#include <RTLib/Ext/CUDA/Math/Math.h>
#include <RTLib/Ext/CUDA/Math/Random.h>
#include <RTLib/Ext/CUDA/Math/VectorFunction.h>
#ifndef __CUDACC__
#include <RTLib/Ext/CUDA/CUDABuffer.h>
#include <fstream>
#include <stack>
#include <vector>
#include <iterator>
#include <fstream>
#include <stack>
#include <algorithm>
#include <functional>
#include <vector>
#include <iterator>
#include <fstream>
#include <stack>
#include <algorithm>
#include <functional>
#endif
namespace RTLib
{
    namespace Ext
    {
        namespace OPX7
        {
            namespace Utils
            {
				enum   SpatialFilter
				{
					//Suitable in GPU
					SpatialFilterNearest,
					//Too Slow in GPU
					SpatialFilterBox,
				};
				enum   DirectionalFilter {
					//Suitable in GPU
					DirectionalFilterNearest,
					//Too Slow in GPU
					DirectionalFilterBox,
				};
				template<unsigned int kDTreeStackDepth>
				struct DTreeNodeT {
					using DTreeNode = DTreeNodeT<kDTreeStackDepth>;
					RTLIB_INLINE RTLIB_HOST_DEVICE DTreeNodeT()noexcept {
						for (int i = 0; i < 4; ++i) {
							children[i] = 0;
							sums[i] = 0.0f;
						}
					}
					RTLIB_INLINE RTLIB_HOST_DEVICE DTreeNodeT(const DTreeNode& node)noexcept {
						CopyFrom(node);
					}
					RTLIB_INLINE RTLIB_HOST_DEVICE DTreeNode& operator=(const DTreeNode& node)noexcept {
						CopyFrom(node);
						return *this;
					}
					RTLIB_INLINE RTLIB_HOST_DEVICE void CopyFrom(const DTreeNode& node)noexcept {
						for (int i = 0; i < 4; ++i) {
							sums[i] = node.sums[i];
							children[i] = node.children[i];
						}
					}
					RTLIB_INLINE RTLIB_HOST_DEVICE bool IsLeaf(int childIdx)const noexcept {
						return children[childIdx] == 0;
					}
					RTLIB_INLINE RTLIB_HOST_DEVICE auto GetChildIdx(float2& p)const noexcept -> int {
						int result = 0;
						float* p_A = reinterpret_cast<float*>(&p);
						for (int i = 0; i < 2; ++i) {
							if (p_A[i] < 0.5f) {
								MoveToLeft(p_A[i]);
							}
							else {
								MoveToRight(p_A[i]);
								result |= (1 << i);
							}
						}
						return result;
					}
					//SUM
					RTLIB_INLINE RTLIB_HOST_DEVICE auto GetSum(int childIdx)const noexcept -> float {
						return sums[childIdx];
					}
					RTLIB_INLINE RTLIB_HOST_DEVICE void SetSum(int childIdx, float val)noexcept {
						sums[childIdx] = val;
					}
					RTLIB_INLINE RTLIB_HOST_DEVICE void SetSumAll(float val)noexcept {
						for (int i = 0; i < 4; ++i) {
							sums[i] = val;
						}
					}
					RTLIB_INLINE RTLIB_HOST_DEVICE auto GetSumOfAll()const noexcept -> float {
						return sums[0] + sums[1] + sums[2] + sums[3];
					}
					RTLIB_INLINE RTLIB_DEVICE      void AddSumAtomic(int idx, float val)noexcept {
#ifdef __CUDA_ARCH__
						atomicAdd(&sums[idx], val);
#else
						sums[idx] += val;
#endif
					}
					//CHILD
					RTLIB_INLINE RTLIB_HOST_DEVICE auto GetChild(int childIdx)const noexcept -> unsigned short {
						return children[childIdx];
					}
					RTLIB_INLINE RTLIB_HOST_DEVICE void SetChild(int childIdx, unsigned short val)noexcept {
						children[childIdx] = val;
					}
					//RECORD
					RTLIB_INLINE RTLIB_DEVICE      void Record(float2& p, float irradiance, DTreeNode* nodes)noexcept {
						DTreeNode* cur = this;
						int        idx = cur->GetChildIdx(p);
						int      depth = 1;
						while (!cur->IsLeaf(idx)) {
							//Leafだったら加算する
							cur = &nodes[cur->children[idx]];
							idx = cur->GetChildIdx(p);
							++depth;
						}
						cur->AddSumAtomic(idx, irradiance);
						return;
					}
					RTLIB_INLINE RTLIB_DEVICE      void  Record(const float2& origin, float size, float2 nodeOrigin, float nodeSize, float value, DTreeNode* nodes) noexcept {
						struct StackNode {
							DTreeNode* curNode;
							float2     nodeOrigin;
							float      nodeSize;
						};

						StackNode stackNodes[kDTreeStackDepth * 3] = {};
						int       stackNodeSize = 1;
						const int stackNodeCapacity = kDTreeStackDepth * 3;

						stackNodes[0].curNode = this;
						stackNodes[0].nodeOrigin = nodeOrigin;
						stackNodes[0].nodeSize = nodeSize;

						while (stackNodeSize > 0) {

							auto  top = stackNodes[stackNodeSize - 1];
							stackNodeSize--;

							if (stackNodeSize > stackNodeCapacity - 4) {
								continue;
							}

							float childSize = top.nodeSize / 2.0f;
							for (int i = 0; i < 4; ++i) {
								float2 childOrigin = top.nodeOrigin;
								if (i & 1) { childOrigin.x += childSize; }
								if (i & 2) { childOrigin.y += childSize; }
								float w = ComputeOverlappingVolume(origin, origin + make_float2(size), childOrigin, childOrigin + make_float2(childSize));
								if (w > 0.0f) {
									if (top.curNode->IsLeaf(i)) {
										top.curNode->AddSumAtomic(i, value * w);
									}
									else {
										stackNodeSize++;
										stackNodes[stackNodeSize - 1].curNode = &nodes[top.curNode->GetChild(i)];
										stackNodes[stackNodeSize - 1].nodeOrigin = childOrigin;
										stackNodes[stackNodeSize - 1].nodeSize = childSize;
									}
								}
							}
						}
					}
					RTLIB_INLINE RTLIB_HOST_DEVICE static auto ComputeOverlappingVolume(const float2& min1, const float2& max1, const float2& min2, const float2& max2)noexcept->float {
						float lengths[2] = {
							fmaxf(fminf(max1.x,max2.x) - fmaxf(min1.x,min2.x),0.0f),
							fmaxf(fminf(max1.y,max2.y) - fmaxf(min1.y,min2.y),0.0f)
						};
						return lengths[0] * lengths[1];
					}

					template<typename RNG>
					RTLIB_INLINE RTLIB_HOST_DEVICE auto SampleAndPdf(RNG& rng, const DTreeNode* nodes, float& pdf_value)const noexcept -> float2 {
						const DTreeNode* cur = this;
						float2 result = make_float2(0.0f);
						pdf_value = 1.0f;
						double size = 1.0f;
						for (;;) {
							int   idx = 0;
							float topLeft = cur->sums[0];
							float topRight = cur->sums[1];
							float partial = cur->sums[0] + cur->sums[2];
							float total = cur->GetSumOfAll();
							//use Two RND Value
							//use Signle RND Value
							//(s0+s2)/(s0+s1+s2+s3)
							float boundary = partial / total;
							auto  origin = make_float2(0.0f);
							float sample = RTLib::Ext::CUDA::Math::random_float1(rng);

							if (sample < boundary)
							{
								sample = sample / boundary;
								boundary = topLeft / partial;
							}
							else
							{
								partial = total - partial;
								origin.x = 0.5f;
								sample = (sample - boundary) / (1.0f - boundary);
								boundary = topRight / partial;
								idx |= (1 << 0);
							}
							if (sample >= boundary) {
								origin.y = 0.5f;
								idx |= (1 << 1);
							}
							pdf_value *= (4.0f * cur->sums[idx] / total);
							result += size * origin;
							size *= 0.5f;

							if (cur->IsLeaf(idx) || cur->sums[idx] <= 0.0f)
							{
								result += size * RTLib::Ext::CUDA::Math::random_float2(rng);
								break;
							}

							cur = &nodes[cur->children[idx]];
						}
						return result;
					}

					template<typename RNG>
					RTLIB_INLINE RTLIB_HOST_DEVICE auto Sample(RNG& rng, const DTreeNode* nodes)const noexcept -> float2 {
						const DTreeNode* cur = this;
						float2 result = make_float2(0.0f);
						double size = 1.0f;
						for (;;) {
							int   idx = 0;
							float topLeft = cur->sums[0];
							float topRight = cur->sums[1];
							float partial = cur->sums[0] + cur->sums[2];
							float total = cur->GetSumOfAll();
							//use Two RND Value
							//use Signle RND Value
							//(s0+s2)/(s0+s1+s2+s3)
							float boundary = partial / total;
							auto  origin = make_float2(0.0f);
							float sample = RTLib::Ext::CUDA::Math::random_float1(rng);

							if (sample < boundary)
							{
								sample = sample / boundary;
								boundary = topLeft / partial;
							}
							else
							{
								partial = total - partial;
								origin.x = 0.5f;
								sample = (sample - boundary) / (1.0f - boundary);
								boundary = topRight / partial;
								idx |= (1 << 0);
							}
							if (sample >= boundary) {
								origin.y = 0.5f;
								idx |= (1 << 1);
							}

							result += size * origin;
							size *= 0.5f;

							if (cur->IsLeaf(idx) || cur->sums[idx] <= 0.0f)
							{
								result += size * RTLib::Ext::CUDA::Math::random_float2(rng);
								break;
							}

							cur = &nodes[cur->children[idx]];
						}
						return result;
					}

					RTLIB_INLINE RTLIB_HOST_DEVICE auto Pdf(float2& p, const DTreeNode* nodes)const noexcept -> float
					{

						float         result = 1.0f;
						const DTreeNode* cur = this;
						int              idx = cur->GetChildIdx(p);
						int            depth = 1;
						for (;;) {
							auto total = cur->GetSumOfAll();
							//if (total <= 0.0f) {
								//break;
#if 0
							float mulOfSum = cur->sums[0] * cur->sums[1] * cur->sums[2] * cur->sums[3];
							//if( total <=0.0f){
							if (mulOfSum <= 0.0f)
#else
							if (cur->GetSum(idx) <= 0.0f)
#endif
							{
								result = 0.0f;
								break;
							}
							//if (isnan(cur->sums[idx])) {
							//	printf("sums = (%f, %f, %f, %f)\n", cur->sums[0], cur->sums[1], cur->sums[2], cur->sums[3]);
							//}
							//if (cur == nullptr) {
							//	printf("cur = nullptr\n");
							//}
							const auto factor = 4.0f * cur->GetSum(idx) / total;
							result *= factor;
							if (cur->IsLeaf(idx)) {
								break;
							}
							cur = &nodes[cur->children[idx]];
							idx = cur->GetChildIdx(p);
							++depth;
						}
						//if (isnan(result)) {
						//	printf("result is nan\n");
						//}
						return result;
					}

					RTLIB_INLINE RTLIB_HOST_DEVICE auto GetDepth(float2& p, const DTreeNode* nodes)const noexcept -> int {
						const DTreeNode* cur = this;
						int              idx = cur->GetChildIdx(p);
						int            depth = 1;
						while (!cur->IsLeaf(idx))
						{
							depth++;
							cur = &nodes[cur->GetChild(idx)];
							idx = cur->GetChildIdx(p);
						}
						return depth;
					}

					RTLIB_INLINE RTLIB_HOST_DEVICE static void MoveToLeft(float& p)noexcept {
						p *= 2.0f;
					}
					RTLIB_INLINE RTLIB_HOST_DEVICE static void MoveToRight(float& p)noexcept {
						p = 2.0f * p - 1.0f;
					}
#ifndef __CUDA_ARCH__
					auto GetArea(const std::vector<DTreeNode>& nodes)const noexcept -> float {
						float  result = 0.0f;
						for (int i = 0; i < 4; ++i) {
							if (GetSum(i) > 0.0f) {
								if (IsLeaf(i)) {
									result += 1.0f / 4.0f;
								}
								else {
									result += nodes[GetChild(i)].GetArea(nodes) / 4.0f;
								}
							}
						}
						return result;
					}
					void Build(std::vector<DTreeNode>& nodes) {
						for (int i = 0; i < 4; ++i) {
							if (this->IsLeaf(i)) {
								continue;
							}
							auto& c = nodes[children[i]];
							c.Build(nodes);
							float sum = 0.0f;
							for (int j = 0; j < 4; ++j) {
								sum += c.sums[j];
							}
							sums[i] = sum;
						}
					}
					void Dump(std::fstream& jsonFile, const std::vector<DTreeNode>& nodes)const noexcept
					{
						jsonFile << "{\n";
						jsonFile << "\"sums\"             : [" << sums[0] << ", " << sums[1] << ", " << sums[2] << ", " << sums[3] << "],\n";
						jsonFile << "\"children\"         : [\n";
						if (!IsLeaf(0)) {
							nodes[children[0]].Dump(jsonFile, nodes);
							jsonFile << ",\n";
						}
						else {
							jsonFile << "{},\n";
						}
						if (!IsLeaf(1)) {
							nodes[children[1]].Dump(jsonFile, nodes);
							jsonFile << ",\n";
						}
						else {
							jsonFile << "{},\n";
						}
						if (!IsLeaf(2)) {
							nodes[children[2]].Dump(jsonFile, nodes);
							jsonFile << ",\n";
						}
						else {
							jsonFile << "{},\n";
						}
						if (!IsLeaf(3)) {
							nodes[children[3]].Dump(jsonFile, nodes);
							jsonFile << "\n";
						}
						else {
							jsonFile << "{}\n";
						}
						jsonFile << "]\n";
						jsonFile << "}";
					}
#endif
					float          sums[4];
					unsigned short children[4];
				};

				template<unsigned int kDTreeStackDepth>
				struct DTreeT {
					using DTree     = DTreeT<kDTreeStackDepth>;
					using DTreeNode = DTreeNodeT<kDTreeStackDepth>;
					template<DirectionalFilter dFilter>
					struct Impl;
					template<>
					struct Impl<DirectionalFilterNearest> {
						RTLIB_INLINE RTLIB_DEVICE static void RecordIrradiance(DTree& dTree, float2 p, float irradiance, float statisticalWeight)noexcept
						{
							if (isfinite(statisticalWeight) && statisticalWeight > 0.0f) {
								dTree.AddStatisticalWeightAtomic(statisticalWeight);
								if (isfinite(irradiance) && irradiance > 0.0f) {
									dTree.nodes[0].Record(p, irradiance * statisticalWeight, dTree.nodes);
								}
							}
						}
					};
					template<>
					struct Impl<DirectionalFilterBox> {
						RTLIB_INLINE RTLIB_DEVICE static void RecordIrradiance(DTree& dTree, float2 p, float irradiance, float statisticalWeight)noexcept
						{
							if (isfinite(statisticalWeight) && statisticalWeight > 0.0f) {
								dTree.AddStatisticalWeightAtomic(statisticalWeight);
								if (isfinite(irradiance) && irradiance > 0.0f) {
									const int depth = dTree.GetDepth(p);
									float size = powf(0.5f, depth);
									auto  origin = p - make_float2(size / 2.0f);
									dTree.nodes[0].Record(origin, size, make_float2(0.0f), 1.0f, irradiance * statisticalWeight / (size * size), dTree.nodes);
								}
							}
						}
					};
					RTLIB_INLINE RTLIB_HOST_DEVICE auto  GetArea()const noexcept -> float {
						return area;
					}
					RTLIB_INLINE RTLIB_HOST_DEVICE auto  GetMean()const noexcept -> float {
						if (statisticalWeight <= 0.0f) { return 0.0f; }
						const float factor = 1.0f / (4.0f * RTLIB_M_PI * statisticalWeight);
						return factor * sum;
					}
					RTLIB_INLINE RTLIB_DEVICE      void  AddStatisticalWeightAtomic(float val)noexcept {
#ifdef __CUDACC__
						atomicAdd(&statisticalWeight, val);
#else
						statisticalWeight += val;
#endif
					}
					RTLIB_INLINE RTLIB_DEVICE      void  AddSumAtomic(float val)noexcept {
#ifdef __CUDACC__
						atomicAdd(&sum, val);
#else
						sum += val;
#endif
					}
					template<DirectionalFilter dFilter>
					RTLIB_INLINE RTLIB_DEVICE      void  RecordIrradiance(float2 p, float irradiance, float statisticalWeight)noexcept {
						Impl<dFilter>::RecordIrradiance(*this, p, irradiance, statisticalWeight);
					}
					template<typename RNG>
					RTLIB_INLINE RTLIB_HOST_DEVICE auto SampleAndPdf(RNG& rng, float& pdf_value)const noexcept -> float2 {
						if (GetMean() <= 0.0f) {
							pdf_value = 1.0f / (4.0f * RTLIB_M_PI);
							return RTLib::Ext::CUDA::Math::random_float2(rng);
						}
						return RTLib::Ext::CUDA::Math::clamp(nodes[0].SampleAndPdf(rng, nodes, pdf_value), make_float2(0.0f), make_float2(1.0f));
					}
					template<typename RNG>
					RTLIB_INLINE RTLIB_HOST_DEVICE auto Sample(RNG& rng)const noexcept -> float2 {
						if (GetMean() <= 0.0f) {
							return RTLib::Ext::CUDA::Math::random_float2(rng);
						}
						return RTLib::Ext::CUDA::Math::clamp(nodes[0].Sample(rng, nodes), make_float2(0.0f), make_float2(1.0f));
					}
					RTLIB_INLINE RTLIB_HOST_DEVICE auto Pdf(float2 p)const noexcept -> float {
						if (GetMean() <= 0.0f) {
							return 1.0f / (4.0f * RTLIB_M_PI);
						}
						return nodes[0].Pdf(p, nodes) / (4.0f * RTLIB_M_PI);
					}
					RTLIB_INLINE RTLIB_HOST_DEVICE auto GetDepth(float2 p)const noexcept -> int {
						return nodes[0].GetDepth(p, nodes);
					}
					DTreeNode* nodes;
					float      area;
					float      sum;
					float      statisticalWeight;
				};

				struct DTreeRecord {
					float3 direction;
					float  radiance;
					float  product;
					float  woPdf, bsdfPdf, dTreePdf;
					float  statisticalWeight;
					bool   isDelta;
				};

				template<unsigned int kDTreeStackDepth>
				struct DTreeWrapperT {
					using DTreeWrapper = DTreeWrapperT<kDTreeStackDepth>;
					using DTree        = DTreeT<kDTreeStackDepth>;
					using DTreeNode    = DTreeNodeT<kDTreeStackDepth>;
					template<DirectionalFilter dFilter>
					RTLIB_INLINE RTLIB_DEVICE      void  Record(const DTreeRecord& rec) noexcept {
						if (!rec.isDelta) {
							float irradiance = rec.radiance / rec.woPdf;
							building.RecordIrradiance<dFilter>(RTLib::Ext::CUDA::Math::dir_to_canonical(rec.direction), irradiance, rec.statisticalWeight);
						}
					}
					template<typename RNG>
					RTLIB_INLINE RTLIB_HOST_DEVICE auto  SampleAndPdf(RNG& rng, float& pdf_value)const noexcept -> float3 {
						return RTLib::Ext::CUDA::Math::canonical_to_dir(sampling.SampleAndPdf(rng, pdf_value));
					}
					template<typename RNG>
					RTLIB_INLINE RTLIB_HOST_DEVICE auto  Sample(RNG& rng)const noexcept -> float3 {
						return RTLib::Ext::CUDA::Math::canonical_to_dir(sampling.Sample(rng));
					}
					RTLIB_INLINE RTLIB_HOST_DEVICE auto  Pdf(const float3& dir)const noexcept -> float {
						return sampling.Pdf(RTLib::Ext::CUDA::Math::dir_to_canonical(dir));
					}
					DTree building;
					DTree sampling;
				};

				template<unsigned int kSTreeStackDepth,unsigned int kDTreeStackDepth>
				struct STreeNodeT {
					using STreeNode    = STreeNodeT<kSTreeStackDepth, kDTreeStackDepth>;
					using DTreeWrapper = DTreeWrapperT<kDTreeStackDepth>;
					using DTree        = DTreeT<kDTreeStackDepth>;
					using DTreeNode    = DTreeNodeT<kDTreeStackDepth>;

					RTLIB_INLINE RTLIB_HOST_DEVICE auto  GetNodeIdx(float3& p)const noexcept -> unsigned int {
						return children[GetChildIdx(p)];
					}
					RTLIB_INLINE RTLIB_HOST_DEVICE auto  GetChildIdx(float3& p)const noexcept -> int {
						float* p_A = reinterpret_cast<float*>(&p);
						p_A[axis] *= 2.0f;
						int    idx = p_A[axis];
						p_A[axis] -= static_cast<float>(idx);
						return idx;
					}
					RTLIB_INLINE RTLIB_HOST_DEVICE bool  IsLeaf()const noexcept {
						return dTree != nullptr;
					}
					RTLIB_INLINE RTLIB_HOST_DEVICE auto  GetDTreeWrapper(float3& p, float3& size, const STreeNode* nodes)const noexcept -> DTreeWrapper* {
						const STreeNode* cur = this;
						int              idx = cur->GetChildIdx(p);
						int            depth = 1;
						unsigned int  t_axis = this->axis;
						while (!cur->IsLeaf()) {
							reinterpret_cast<float*>(&size)[t_axis] /= 2.0f;
							cur = &nodes[cur->children[idx]];
							idx = cur->GetChildIdx(p);
							t_axis = (t_axis + 1) % 3;
							++depth;
						}
						return cur->dTree;
					}
					RTLIB_INLINE RTLIB_HOST_DEVICE auto  GetDTreeWrapper()const noexcept -> DTreeWrapper* {
						return dTree;
					}
					template<DirectionalFilter dFilter>
					RTLIB_INLINE RTLIB_DEVICE      void  Record(const float3& min1, const float3& max1, float3 min2, float3 size2, const DTreeRecord& rec, STreeNode* nodes) noexcept {
						struct StackNode {
							STreeNode* curNode;
							float3     min2;
							float3     size2;
						};
						StackNode stackNodes[kSTreeStackDepth] = {};
						int       stackNodeSize = 1;
						const int stackNodeCapacity = kSTreeStackDepth;
						stackNodes[0].curNode = this;
						stackNodes[0].min2 = min2;
						stackNodes[0].size2 = size2;
						while (stackNodeSize > 0) {
							auto  top = stackNodes[stackNodeSize - 1];
							stackNodeSize--;
							float w = ComputeOverlappingVolume(min1, max1, top.min2, top.min2 + top.size2);
							if (w > 0.0f) {
								if (top.curNode->IsLeaf()) {
									top.curNode->dTree->Record<dFilter>({ rec.direction,rec.radiance,rec.product,rec.woPdf,rec.bsdfPdf,rec.dTreePdf,rec.statisticalWeight * w,rec.isDelta });
								}
								else if (stackNodeSize < stackNodeCapacity) {
									float3 t_size2 = top.size2;
									int    t_axis = top.curNode->axis;
									reinterpret_cast<float*>(&t_size2)[t_axis] /= 2.0f;
									for (int i = 0; i < 2; ++i) {
										float3 t_min2 = top.min2;
										if (i & 1) {
											reinterpret_cast<float*>(&t_min2)[t_axis] += reinterpret_cast<float*>(&t_size2)[t_axis];
										}
										stackNodes[stackNodeSize + i].curNode = &nodes[top.curNode->children[i]];
										stackNodes[stackNodeSize + i].min2 = t_min2;
										stackNodes[stackNodeSize + i].size2 = t_size2;
									}
									stackNodeSize += 2;
								}
							}
						}
					}
					RTLIB_INLINE RTLIB_HOST_DEVICE static auto ComputeOverlappingVolume(const float3& min1, const float3& max1, const float3& min2, const float3& max2)noexcept->float {
						float lengths[3] = {
							fmaxf(fminf(max1.x,max2.x) - fmaxf(min1.x,min2.x),0.0f),
							fmaxf(fminf(max1.y,max2.y) - fmaxf(min1.y,min2.y),0.0f),
							fmaxf(fminf(max1.z,max2.z) - fmaxf(min1.z,min2.z),0.0f)
						};
						return lengths[0] * lengths[1] * lengths[2];
					}
					RTLIB_INLINE RTLIB_HOST_DEVICE static void MoveToLeft(float& p)noexcept {
						p *= 2.0f;
					}
					RTLIB_INLINE RTLIB_HOST_DEVICE static void MoveToRight(float& p)noexcept {
						p = 2.0f * p - 1.0f;
					}
					unsigned char axis;
					unsigned int  children[2];
					DTreeWrapper* dTree;
				};

				template<unsigned int kSTreeStackDepth, unsigned int kDTreeStackDepth>
				struct STreeT {
					using STree        = STreeT<kSTreeStackDepth, kDTreeStackDepth>;
					using STreeNode    = STreeNodeT<kSTreeStackDepth, kDTreeStackDepth>;
					using DTreeWrapper = DTreeWrapperT<kDTreeStackDepth>;
					using DTree        = DTreeT<kDTreeStackDepth>;
					using DTreeNode    = DTreeNodeT<kDTreeStackDepth>;

					RTLIB_INLINE RTLIB_HOST_DEVICE auto  GetDTreeWrapper(float3 p, float3& size)const noexcept -> DTreeWrapper* {
						size = aabbMax - aabbMin;
						p = p - aabbMin;
						p /= size;
						return nodes[0].GetDTreeWrapper(p, size, nodes);
					}
					RTLIB_INLINE RTLIB_HOST_DEVICE auto  GetDTreeWrapper(float3 p)const noexcept -> DTreeWrapper* {
						float3 size;
						return GetDTreeWrapper(p, size);
					}
					template<DirectionalFilter dFilter>
					RTLIB_INLINE RTLIB_DEVICE      void  Record(const float3& p, const float3& dVoxelSize, DTreeRecord rec)
					{
						float volume = dVoxelSize.x * dVoxelSize.y * dVoxelSize.z;
						rec.statisticalWeight /= volume;
						nodes[0].Record<dFilter>(p - dVoxelSize * 0.5f, p + dVoxelSize * 0.5f, aabbMin, aabbMax - aabbMin, rec, nodes);
					}
					STreeNode* nodes;
					float3     aabbMin;
					float3     aabbMax;
					float      fraction;
				};

				template<unsigned int kSTreeStackDepth, unsigned int kDTreeStackDepth>
				struct STreeNode2T {

					using STreeNode2   = STreeNode2T<kSTreeStackDepth, kDTreeStackDepth>;
					using DTreeWrapper = DTreeWrapperT<kDTreeStackDepth>;
					using DTree        = DTreeT<kDTreeStackDepth>;
					using DTreeNode    = DTreeNodeT<kDTreeStackDepth>;

					RTLIB_INLINE RTLIB_HOST_DEVICE auto  GetNodeIdx(float3& p)const noexcept -> unsigned int {
						return children[GetChildIdx(p)];
					}
					RTLIB_INLINE RTLIB_HOST_DEVICE auto  GetChildIdx(float3& p)const noexcept -> int {
						int i_x = 2.0f * p.x;
						int i_y = 2.0f * p.y;
						int i_z = 2.0f * p.z;
						p *= 2.0f;
						p -= make_float3((float)i_x, (float)i_y, (float)i_z);
						return i_x + (1 << 1) * i_y + (1 << 2) * i_z;
					}
					RTLIB_INLINE RTLIB_HOST_DEVICE bool  IsLeaf()const noexcept {
						return dTree != nullptr;
					}
					RTLIB_INLINE RTLIB_HOST_DEVICE auto  GetDTreeWrapper(float3& p, float3& size, const STreeNode2* nodes)const noexcept -> DTreeWrapper* {
						const STreeNode2* cur = this;
						int               idx = cur->GetChildIdx(p);
						while (!cur->IsLeaf()) {
							size /= 2.0f;
							cur = &nodes[cur->children[idx]];
							idx = cur->GetChildIdx(p);
						}
						return cur->dTree;
					}
					RTLIB_INLINE RTLIB_HOST_DEVICE auto  GetDTreeWrapper()const noexcept -> DTreeWrapper* {
						return dTree;
					}
					template<DirectionalFilter dFilter>
					RTLIB_INLINE RTLIB_DEVICE      void  Record(const float3& min1, const float3& max1, float3 min2, float3 size2, const DTreeRecord& rec, STreeNode2* nodes) noexcept {
						struct StackNode {
							STreeNode2* curNode;
							float3      min2;
							float3      size2;
						};
						StackNode stackNodes[7 * kSTreeStackDepth] = {};
						int       stackNodeSize = 1;
						const int stackNodeCapacity = 7 * kSTreeStackDepth;
						stackNodes[0].curNode = this;
						stackNodes[0].min2 = min2;
						stackNodes[0].size2 = size2;
						while (stackNodeSize > 0) {
							auto  top = stackNodes[stackNodeSize - 1];
							stackNodeSize--;
							float w = ComputeOverlappingVolume(min1, max1, top.min2, top.min2 + top.size2);
							if (w > 0.0f) {
								if (top.curNode->IsLeaf()) {
									top.curNode->dTree->Record<dFilter>({ rec.direction,rec.radiance,rec.product,rec.woPdf,rec.bsdfPdf,rec.dTreePdf,rec.statisticalWeight * w,rec.isDelta });
								}
								else if (stackNodeSize < stackNodeCapacity) {
									float3 t_size2 = top.size2;
									t_size2 /= 2.0f;
									for (int i = 0; i < 8; ++i) {
										float3 t_min2 = top.min2;
										if (i & 1) {
											t_min2.x += t_size2.x;
										}
										if (i & 2) {
											t_min2.y += t_size2.y;
										}

										if (i & 4) {
											t_min2.z += t_size2.z;
										}
										stackNodes[stackNodeSize + i].curNode = &nodes[top.curNode->children[i]];
										stackNodes[stackNodeSize + i].min2 = t_min2;
										stackNodes[stackNodeSize + i].size2 = t_size2;
									}
									stackNodeSize += 8;
								}
							}
						}
					}
					RTLIB_INLINE RTLIB_HOST_DEVICE static auto ComputeOverlappingVolume(const float3& min1, const float3& max1, const float3& min2, const float3& max2)noexcept->float {
						float lengths[3] = {
							fmaxf(fminf(max1.x,max2.x) - fmaxf(min1.x,min2.x),0.0f),
							fmaxf(fminf(max1.y,max2.y) - fmaxf(min1.y,min2.y),0.0f),
							fmaxf(fminf(max1.z,max2.z) - fmaxf(min1.z,min2.z),0.0f)
						};
						return lengths[0] * lengths[1] * lengths[2];
					}
					unsigned int  children[8];
					DTreeWrapper* dTree;
				};

				template<unsigned int kSTreeStackDepth, unsigned int kDTreeStackDepth>
				struct STree2T {
					using STree2       = STree2T<kSTreeStackDepth, kDTreeStackDepth>;
					using STreeNode2   = STreeNode2T<kSTreeStackDepth, kDTreeStackDepth>;
					using DTreeWrapper = DTreeWrapperT<kDTreeStackDepth>;
					using DTree        = DTreeT<kDTreeStackDepth>;
					using DTreeNode    = DTreeNodeT<kDTreeStackDepth>;

					RTLIB_INLINE RTLIB_HOST_DEVICE auto  GetDTreeWrapper(float3 p, float3& size)const noexcept -> DTreeWrapper* {
						size = aabbMax - aabbMin;
						p = p - aabbMin;
						p /= size;
						return nodes[0].GetDTreeWrapper(p, size, nodes);
					}
					RTLIB_INLINE RTLIB_HOST_DEVICE auto  GetDTreeWrapper(float3 p)const noexcept -> DTreeWrapper* {
						float3 size;
						return GetDTreeWrapper(p, size);
					}
					template<DirectionalFilter dFilter>
					RTLIB_INLINE RTLIB_DEVICE      void  Record(const float3& p, const float3& dVoxelSize, DTreeRecord rec)
					{
						float volume = dVoxelSize.x * dVoxelSize.y * dVoxelSize.z;
						rec.statisticalWeight /= volume;
						nodes[0].Record<dFilter>(p - dVoxelSize * 0.5f, p + dVoxelSize * 0.5f, aabbMin, aabbMax - aabbMin, rec, nodes);
					}
					STreeNode2* nodes;
					float3      aabbMin;
					float3      aabbMax;
					float       fraction;
				};

				template<unsigned int kDTreeStackDepth>
				struct TraceVertexT {
					using TraceVertex  = TraceVertexT<kDTreeStackDepth>;
					using DTreeWrapper = DTreeWrapperT<kDTreeStackDepth>;
					using DTree        = DTreeT<kDTreeStackDepth>;
					using DTreeNode    = DTreeNodeT<kDTreeStackDepth>;

					template<SpatialFilter sFilter>
					struct Impl;

					template<>
					struct Impl<SpatialFilter::SpatialFilterNearest>
					{
						template<DirectionalFilter dFilter, typename STreeType>
						RTLIB_INLINE RTLIB_HOST_DEVICE static void Record(TraceVertex& v, STreeType& sTree, const DTreeRecord& rec)
						{
							v.dTree->Record<dFilter>(rec);
						}
					};

					template<>
					struct Impl<SpatialFilter::SpatialFilterBox>
					{
						template<DirectionalFilter dFilter, typename STreeType>
						RTLIB_INLINE RTLIB_HOST_DEVICE static void Record(TraceVertex& v, STreeType& sTree, const DTreeRecord& rec)
						{
							sTree.Record<dFilter>(v.rayOrigin, v.dTreeVoxelSize, rec);
						}
					};

					DTreeWrapper* dTree;
					float3        dTreeVoxelSize;
					float3        rayOrigin;
					float3        rayDirection;
					float3        throughPut;
					float3        bsdfVal;
					float3        radiance;
					float         woPdf;
					float         bsdfPdf;
					float         dTreePdf;
					float         cosine;
					bool          isDelta;
					RTLIB_INLINE RTLIB_HOST_DEVICE void Record(const float3& r) noexcept {
						radiance += r;
					}
					template<SpatialFilter sFilter, DirectionalFilter dFilter, typename STreeType>
					RTLIB_INLINE RTLIB_HOST_DEVICE void Commit(STreeType& sTree, float statisticalWeight)noexcept
					{
						if (!dTree) {
							return;
						}
						bool isValidRadiance =
							(isfinite(radiance.x) && radiance.x >= 0.0f) &&
							(isfinite(radiance.y) && radiance.y >= 0.0f) &&
							(isfinite(radiance.z) && radiance.z >= 0.0f);
						bool isValidBsdfVal =
							(isfinite(bsdfVal.x) && bsdfVal.x >= 0.0f) &&
							(isfinite(bsdfVal.y) && bsdfVal.y >= 0.0f) &&
							(isfinite(bsdfVal.z) && bsdfVal.z >= 0.0f);
						if (woPdf <= 0.0f || !isValidRadiance || !isValidBsdfVal)
						{
							return;
						}
						auto localRadiance = make_float3(0.0f);
						if (throughPut.x * woPdf > 1e-4f) {
							localRadiance.x = radiance.x / throughPut.x;
						}
						if (throughPut.y * woPdf > 1e-4f) {
							localRadiance.y = radiance.y / throughPut.y;
						}
						if (throughPut.z * woPdf > 1e-4f) {
							localRadiance.z = radiance.z / throughPut.z;
						}

						localRadiance *= fabsf(cosine);
						/*printf("localRadiance=(%f,%f,%f)\n",localRadiance.x,localRadiance.y,localRadiance.z);*/
						float3 product = localRadiance * bsdfVal;
						float localRadianceAvg = (localRadiance.x + localRadiance.y + localRadiance.z) / 3.0f;
						float productAvg = (product.x + product.y + product.z) / 3.0f;
						DTreeRecord rec{ rayDirection,localRadianceAvg ,productAvg,woPdf,bsdfPdf,dTreePdf,statisticalWeight,isDelta };
						Impl<sFilter>::Record<dFilter>(*this, sTree, rec);
					}
				};
#ifndef __CUDACC__
				template<unsigned int kDTreeStackDepth>
				using  RTDTreeNodeT = DTreeNodeT<kDTreeStackDepth>;

				template<unsigned int kDTreeStackDepth>
				class  RTDTreeT {
					using RTDTree     = RTDTreeT<kDTreeStackDepth>;
					using RTDTreeNode = RTDTreeNodeT<kDTreeStackDepth>;

					using DTree       = DTreeT<kDTreeStackDepth>;
					using DTreeNode   = DTreeNodeT<kDTreeStackDepth>;
				public:
					RTDTreeT()noexcept {
						m_MaxDepth = 0;
						m_Nodes.emplace_back();
						m_Nodes.front().SetSumAll(0.0f);
						m_Sum = 0.0f;
						m_Area = 0.0f;
						m_StatisticalWeight = 0.0f;
					}
					void Reset(const RTDTree& prvDTree, int newMaxDepth, float subDivTh) {
						m_Area = 0.0f;
						m_Sum = 0.0f;
						m_StatisticalWeight = 0.0f;
						m_MaxDepth = 0;
						m_Nodes.clear();
						m_Nodes.emplace_back();
						struct StackNode {
							size_t         dstNodeIdx;
							//const RTDTree* dstDTree = this;
							size_t         srcNodeIdx;
							const RTDTree* srcDTree;
							int            depth;
							auto GetSrcNode()const -> const RTDTreeNode& {
								return srcDTree->Node(srcNodeIdx);
							}
						};
						std::stack<StackNode> stackNodes = {};
						stackNodes.push({ 0,0,&prvDTree,1 });
						const auto total = prvDTree.m_Sum;
						while (!stackNodes.empty())
						{
							StackNode sNode = stackNodes.top();
							stackNodes.pop();

							m_MaxDepth = std::max(m_MaxDepth, sNode.depth);

							for (int i = 0; i < 4; ++i) {
								//this
								const auto fraction = total > 0.0f ? (sNode.GetSrcNode().GetSum(i) / total) : std::pow(0.25f, sNode.depth);
								if (sNode.depth < newMaxDepth && fraction > subDivTh) {
									if (!sNode.GetSrcNode().IsLeaf(i)) {
										if (sNode.srcDTree != &prvDTree) {
											std::cout << "sNode.srcDTree != &prvDTree!\n";
										}
										//Not Leaf -> Copy Child
										stackNodes.push({ m_Nodes.size(), sNode.GetSrcNode().GetChild(i),&prvDTree,sNode.depth + 1 });
									}
									else {
										//    Leaf -> Copy Itself
										stackNodes.push({ m_Nodes.size(), m_Nodes.size()                , this    ,sNode.depth + 1 });
									}
									m_Nodes[sNode.dstNodeIdx].SetChild(i, static_cast<unsigned short>(m_Nodes.size()));
									m_Nodes.emplace_back();
									auto& backNode = m_Nodes.back();
									backNode.SetSumAll(sNode.GetSrcNode().GetSum(i) / 4.0f);
									if (m_Nodes.size() > std::numeric_limits<uint16_t>::max())
									{
										std::cout << "DTreeWrapper hit maximum count!\n";
										stackNodes = {};
										break;
									}
								}
							}
						}
						for (auto& node : m_Nodes)
						{
							node.SetSumAll(0.0f);
						}
					}
					void Build() {
						auto& root = m_Nodes.front();
						root.Build(m_Nodes);
						m_Area = root.GetArea(m_Nodes);
						float sum = 0.0f;
						for (int i = 0; i < 4; ++i) {
							sum += root.sums[i];
						}
						m_Sum = sum;
					}
					auto GetSum()const noexcept -> float {
						return m_Sum;
					}
					void SetSum(float val)noexcept {
						m_Sum = val;
					}
					void SetStatisticalWeight(float val)noexcept {
						m_StatisticalWeight = val;
					}
					auto GetStatisticalWeight()const noexcept -> float {
						return m_StatisticalWeight;
					}
					auto GetApproxMemoryFootPrint()const noexcept -> size_t {
						return sizeof(DTreeNode) * m_Nodes.size() + sizeof(DTree);
					}
					auto GetNumNodes()const noexcept -> size_t {
						return m_Nodes.size();
					}
					auto GetDepth()const noexcept -> int {
						return m_MaxDepth;
					}
					auto GetMean()const noexcept -> float {
						if (m_StatisticalWeight * m_Area <= 0.0f) { return 0.0f; }
						const float factor = 1.0f / (4.0f * RTLIB_M_PI * m_Area * m_StatisticalWeight);
						return factor * m_Sum;
					}
					auto GetArea()const noexcept -> float {
						return m_Area;
					}
					template<typename RNG>
					auto Sample(RNG& rng)const noexcept -> float3 {
						if (GetMean() <= 0.0f) {
							return RTLib::Ext::CUDA::Math::canonical_to_dir(RTLib::Ext::CUDA::Math::random_float2(rng));
						}
						return RTLib::Ext::CUDA::Math::canonical_to_dir(m_Nodes[0].Sample(rng, m_Nodes.data()));
					}
					auto Pdf(const float3& dir)const noexcept -> float {
						if (GetMean() <= 0.0f) {
							return 1.0f / (4.0f * RTLIB_M_PI);
						}
						float2 dir2 = RTLib::Ext::CUDA::Math::dir_to_canonical(dir);
						return m_Area * m_Nodes[0].Pdf(dir2, m_Nodes.data()) / (4.0f * RTLIB_M_PI);
					}
					void Dump(std::fstream& jsonFile)const noexcept {
						jsonFile << "{\n";
						jsonFile << "\"sum\"              : " << m_Sum << ",\n";
						jsonFile << "\"statisticalWeight\": " << m_StatisticalWeight << ",\n";
						jsonFile << "\"maxDepth\"         : " << m_MaxDepth << ",\n";
						jsonFile << "\"root\"             :  \n";
						m_Nodes[0].Dump(jsonFile, m_Nodes);
						jsonFile << "\n";
						jsonFile << "}";
					}
					auto Nodes()noexcept -> std::vector<RTDTreeNode>& {
						return m_Nodes;
					}
					auto Node(size_t idx)const noexcept -> const RTDTreeNode& {
						return m_Nodes[idx];
					}
					auto Node(size_t idx) noexcept -> RTDTreeNode& {
						return m_Nodes[idx];
					}
					void SetGpuHandle(const DTree& dTree)
					{
						m_Area = dTree.area;
						m_Sum = dTree.sum;
						m_StatisticalWeight = dTree.statisticalWeight;
					}
					auto GetGpuHandle()const noexcept ->DTree {
						auto dTree = DTree();
						dTree.area = m_Area;
						dTree.sum = m_Sum;
						dTree.statisticalWeight = m_StatisticalWeight;
						dTree.nodes = nullptr;
						return dTree;
					}
				private:
					std::vector<RTDTreeNode> m_Nodes;
					float                    m_Area;
					float                    m_Sum;
					float                    m_StatisticalWeight;
					int                      m_MaxDepth;
				};

				template<unsigned int kDTreeStackDepth>
				struct RTDTreeWrapperT {
					using RTDTreeWrapper = RTDTreeWrapperT<kDTreeStackDepth>;
					using RTDTree        = RTDTreeT<kDTreeStackDepth>;
					using RTDTreeNode    = RTDTreeNodeT<kDTreeStackDepth>;

					using DTreeWrapper = DTreeWrapperT<kDTreeStackDepth>;
					using DTree        = DTreeT<kDTreeStackDepth>;
					using DTreeNode    = DTreeNodeT<kDTreeStackDepth>;

					auto GetApproxMemoryFootPrint()const noexcept->size_t {
						return building.GetApproxMemoryFootPrint() + sampling.GetApproxMemoryFootPrint();
					}
					auto GetStatisticalWeightSampling()const noexcept -> float {
						return sampling.GetStatisticalWeight();
					}
					auto GetStatisticalWeightBuilding()const noexcept -> float {
						return building.GetStatisticalWeight();
					}
					void SetStatisticalWeightSampling(float val)noexcept {
						sampling.SetStatisticalWeight(val);
					}
					void SetStatisticalWeightBuilding(float val)noexcept {
						building.SetStatisticalWeight(val);
					}
					template<typename RNG>
					auto  Sample(RNG& rng)const noexcept -> float3 {
						return sampling.Sample(rng);
					}
					auto  Pdf(const float3& dir)const noexcept -> float {
						return sampling.Pdf(dir);
					}
					auto  GetNumNodes()const noexcept->size_t {
						return sampling.GetNumNodes();
					}
					auto  GetMean()const noexcept->float {
						return sampling.GetMean();
					}
					auto  GetArea()const noexcept->float {
						return sampling.GetArea();
					}
					auto  GetDepth()const noexcept -> int {
						return sampling.GetDepth();
					}
					void  Build() {
						//一層にする→うまくいく
						building.Build();
						sampling = building;
					}
					void  Reset(int newMaxDepth, float subDivTh) {
						//Buildingを削除し、samplingで得た新しい構造に変更
						building.Reset(sampling, newMaxDepth, subDivTh);
					}
					RTDTree    building;
					RTDTree    sampling;
				};

				template<unsigned int kSTreeStackDepth, unsigned int kDTreeStackDepth>
				struct RTSTreeNodeT {
					using RTSTreeNode    = RTSTreeNodeT<kSTreeStackDepth, kDTreeStackDepth>;
					using RTDTreeWrapper = RTDTreeWrapperT<kDTreeStackDepth>;
					using RTDTree        = RTDTreeT<kDTreeStackDepth>;
					using RTDTreeNode    = RTDTreeNodeT<kDTreeStackDepth>;

					using STreeNode    = STreeNodeT<kSTreeStackDepth, kDTreeStackDepth>;
					using DTreeWrapper = DTreeWrapperT<kDTreeStackDepth>;
					using DTree        = DTreeT<kDTreeStackDepth>;
					using DTreeNode    = DTreeNodeT<kDTreeStackDepth>;

					RTSTreeNodeT()noexcept : dTree(), isLeaf{ true }, axis{ 0 }, padding{ 0 }, children{}{}
					auto GetChildIdx(float3& p)const noexcept -> int {
						float* p_A = reinterpret_cast<float*>(&p);
						if (p_A[axis] < 0.5f) {
							p_A[axis] *= 2.0f;
							return 0;
						}
						else {
							p_A[axis] = 2.0f * p_A[axis] - 1.0f;
							return 1;
						}
					}
					auto GetNodeIdx(float3& p)const noexcept -> unsigned int {
						return children[GetChildIdx(p)];
					}
					auto GetDTree(float3 p, float3& size, const std::vector<RTSTreeNode>& nodes)const noexcept -> const RTDTreeWrapper* {
						const RTSTreeNode* cur = this;
						int   ndx = cur->GetNodeIdx(p);
						int   depth = 1;
						while (true) {
							if (cur->isLeaf) {
								return &cur->dTree;
							}
							reinterpret_cast<float*>(&size)[axis] /= 2.0f;
							cur = &nodes[ndx];
							ndx = cur->GetNodeIdx(p);
							depth++;
						}
						return nullptr;
					}
					auto GetDTreeWrapper()const noexcept -> const RTDTreeWrapper* {
						return &dTree;
					}
					auto GetDepth(const std::vector<RTSTreeNode>& nodes)const-> int {
						int result = 1;
						if (isLeaf) {
							return 1;
						}
						for (int i = 0; i < 2; ++i) {
							result = std::max(result, 1 + nodes[children[i]].GetDepth(nodes));
						}
						return result;
					}
					void Dump(std::fstream& jsonFile, size_t sTreeNodeIdx, const std::vector<RTSTreeNode>& nodes)const noexcept {
						jsonFile << "{\n";
						jsonFile << "\"isLeaf\"  : " << (this->isLeaf ? "true" : "false") << ",\n";
						jsonFile << "\"axis\"    : " << (int)this->axis << ",\n";
						if (!this->isLeaf) {
							jsonFile << "\"children\": [\n";
							nodes[this->children[0]].Dump(jsonFile, children[0], nodes);
							jsonFile << ",\n";
							nodes[this->children[1]].Dump(jsonFile, children[1], nodes);
							jsonFile << "\n";

							jsonFile << "]\n";
						}
						else {
							jsonFile << "\"dTree\": \"" << "dTree" << sTreeNodeIdx << "\"";
						}
						jsonFile << "}";
					}
					void SetGpuHandle(const STreeNode& sTreeNode)noexcept
					{
						isLeaf = sTreeNode.IsLeaf();
						axis = sTreeNode.axis;
						children[0] = sTreeNode.children[0];
						children[1] = sTreeNode.children[1];
					}
					auto GetGpuHandle()const noexcept -> STreeNode
					{
						STreeNode sTreeNode;
						sTreeNode.axis = axis;
						sTreeNode.children[0] = children[0];
						sTreeNode.children[1] = children[1];
						sTreeNode.dTree = nullptr;
						return sTreeNode;
					}

					RTDTreeWrapper           dTree;
					bool                     isLeaf;
					unsigned char            axis;
					unsigned short           padding;    //2^32���ő�
					unsigned int             children[2];//2^32
				};

				template<unsigned int kSTreeStackDepth, unsigned int kDTreeStackDepth>
				class  RTSTreeT {
				public:
					using RTSTree        = RTSTreeT<kSTreeStackDepth, kDTreeStackDepth>;
					using RTSTreeNode    = RTSTreeNodeT<kSTreeStackDepth, kDTreeStackDepth>;
					using RTDTreeWrapper = RTDTreeWrapperT<kDTreeStackDepth>;
					using RTDTree        = RTDTreeT<kDTreeStackDepth>;
					using RTDTreeNode    = RTDTreeNodeT<kDTreeStackDepth>;

					using STree        = STreeT<kSTreeStackDepth, kDTreeStackDepth>;
					using STreeNode    = STreeNodeT<kSTreeStackDepth, kDTreeStackDepth>;
					using DTreeWrapper = DTreeWrapperT<kDTreeStackDepth>;
					using DTree        = DTreeT<kDTreeStackDepth>;
					using DTreeNode    = DTreeNodeT<kDTreeStackDepth>;
				public:
					RTSTreeT(const float3& aabbMin, const float3& aabbMax)noexcept {
						this->Clear();
						auto size = aabbMax - aabbMin;
						auto maxSize = std::max(std::max(size.x, size.y), size.z);
						m_AabbMin = aabbMin;
						m_AabbMax = aabbMin + make_float3(maxSize);
					}
					void Clear()noexcept {
						m_Nodes.clear();
						m_Nodes.emplace_back();
					}
					void SubDivideAll() {
						int nNodes = m_Nodes.size();
						for (size_t i = 0; i < nNodes; ++i)
						{
							if (m_Nodes[i].isLeaf) {
								SubDivide(i, m_Nodes);
							}
						}
					}
					void SubDivide(int nodeIdx, std::vector<RTSTreeNode>& nodes)
					{
						size_t curNodeIdx = nodes.size();
						nodes.resize(curNodeIdx + 2);
						auto& cur = nodes[nodeIdx];
						for (int i = 0; i < 2; ++i)
						{
							uint32_t idx = curNodeIdx + i;
							cur.children[i] = idx;
							nodes[idx].axis = (cur.axis + 1) % 3;
							nodes[idx].isLeaf = true;
							nodes[idx].dTree = cur.dTree;
							nodes[idx].dTree.building.SetStatisticalWeight(cur.dTree.building.GetStatisticalWeight() / 2.0f);
						}
						cur.isLeaf = false;
						cur.dTree = {};
					}
					auto GetDTree(float3 p, float3& size)const noexcept -> const RTDTreeWrapper* {
						size = m_AabbMax - m_AabbMin;
						p = p - m_AabbMin;
						p /= size;
						return m_Nodes[0].GetDTree(p, size, m_Nodes);
					}
					auto GetDTree(const float3& p)const noexcept ->const RTDTreeWrapper* {
						float3 size;
						return GetDTree(p, size);
					}
					auto GetDepth()const -> int {
						return m_Nodes[0].GetDepth(m_Nodes);
					}
					auto Node(size_t idx)const noexcept -> const RTSTreeNode& {
						return m_Nodes[idx];
					}
					auto Node(size_t idx) noexcept -> RTSTreeNode& {
						return m_Nodes[idx];
					}
					auto GetNumNodes()const noexcept -> size_t {
						return m_Nodes.size();
					}
					bool ShallSplit(const RTSTreeNode& node, int depth, size_t samplesRequired)const noexcept
					{
						//std::cout << node.dTree.GetStatisticalWeight() << "vs " << samplesRequired << std::endl;
						return m_Nodes.size() < std::numeric_limits<uint32_t>::max() - 1 && node.dTree.building.GetStatisticalWeight() > samplesRequired;
					}
					void Refine(size_t sTreeTh, int maxMB) {
						if (maxMB >= 0) {
							size_t approxMemoryFootPrint = 0;
							for (const auto& node : m_Nodes)
							{
								approxMemoryFootPrint += node.GetDTreeWrapper()->GetApproxMemoryFootPrint();
							}
							if (approxMemoryFootPrint / 1000000 >= maxMB) {
								return;
							}
						}

						struct StackNode {
							size_t index;
							int    depth;
						};
						std::stack<StackNode> nodeIndices = {};
						nodeIndices.push({ 0, 1 });
						while (!nodeIndices.empty())
						{
							StackNode sNode = nodeIndices.top();
							nodeIndices.pop();

							if (m_Nodes[sNode.index].isLeaf) {
								if (ShallSplit(m_Nodes[sNode.index], sNode.depth, sTreeTh)) {
									SubDivide((int)sNode.index, m_Nodes);
								}
							}

							if (!m_Nodes[sNode.index].isLeaf) {
								const auto& node = m_Nodes[sNode.index];
								for (int i = 0; i < 2; ++i) {
									nodeIndices.push({ node.children[i],sNode.depth + 1 });
								}
							}
						}
					}
					auto GetAabbMin()const -> float3 {
						return m_AabbMin;
					}
					auto GetAabbMax()const -> float3 {
						return m_AabbMax;
					}
					void Dump(std::fstream& jsonFile)const noexcept {
						jsonFile << "{\n";
						jsonFile << "\"aabbMin\" : [" << m_AabbMin.x << ", " << m_AabbMin.y << ", " << m_AabbMin.z << "],\n";
						jsonFile << "\"aabbMax\" : [" << m_AabbMax.x << ", " << m_AabbMax.y << ", " << m_AabbMax.z << "],\n";
						jsonFile << "\"root\"    : \n";
						m_Nodes[0].Dump(jsonFile, 0, m_Nodes);
						jsonFile << "\n";
						jsonFile << "}\n";
					}
				private:
					std::vector<RTSTreeNode> m_Nodes;
					float3                   m_AabbMin;
					float3                   m_AabbMax;
				};

				template<unsigned int kSTreeStackDepth, unsigned int kDTreeStackDepth>
				class  RTSTreeWrapperT {
				public:
					using RTSTreeWrapper = RTSTreeWrapperT<kSTreeStackDepth, kDTreeStackDepth>;
					using RTSTree        = RTSTreeT<kSTreeStackDepth, kDTreeStackDepth>;
					using RTSTreeNode    = RTSTreeNodeT<kSTreeStackDepth, kDTreeStackDepth>;
					using RTDTreeWrapper = RTDTreeWrapperT<kDTreeStackDepth>;
					using RTDTree        = RTDTreeT<kDTreeStackDepth>;
					using RTDTreeNode    = RTDTreeNodeT<kDTreeStackDepth>;

					using STree        = STreeT<kSTreeStackDepth, kDTreeStackDepth>;
					using STreeNode    = STreeNodeT<kSTreeStackDepth, kDTreeStackDepth>;
					using DTreeWrapper = DTreeWrapperT<kDTreeStackDepth>;
					using DTree        = DTreeT<kDTreeStackDepth>;
					using DTreeNode    = DTreeNodeT<kDTreeStackDepth>;

					using CUDABuffer    = RTLib::Ext::CUDA::CUDABuffer;
					using CUDABufferPtr = std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>;
				public:
					RTSTreeWrapperT(RTLib::Ext::CUDA::CUDAContext* context, const float3& aabbMin, const float3& aabbMax, unsigned int maxDTreeDepth = 20, float fraction = 0.3f)noexcept :m_CpuSTree{ aabbMin,aabbMax }, m_Context{ context }, m_MaxDTreeDepth{ maxDTreeDepth }, m_Fraction{ fraction }{}
					void Upload(CUDA::CUDAStream* stream = nullptr)noexcept {
						bool isNeededSync = false;
						//Uploadは両方必要
						const size_t gpuSTreeNodeCnt = m_CpuSTree.GetNumNodes();
						size_t gpuDTreeCnt = 0;
						size_t gpuDTreeNodeCntBuilding = 0;
						size_t gpuDTreeNodeCntSampling = 0;
						for (size_t i = 0; i < gpuSTreeNodeCnt; ++i) {
							if (m_CpuSTree.Node(i).isLeaf) {
								gpuDTreeCnt++;
								gpuDTreeNodeCntBuilding += m_CpuSTree.Node(i).dTree.building.GetNumNodes();
								gpuDTreeNodeCntSampling += m_CpuSTree.Node(i).dTree.sampling.GetNumNodes();
							}
						}
						//CPU Upload Memory
						std::vector<STreeNode>    sTreeNodes(gpuSTreeNodeCnt);
						std::vector<DTreeWrapper> dTreeWrappers(gpuDTreeCnt);
						std::vector<DTreeNode>    dTreeNodesBuilding(gpuDTreeNodeCntBuilding);
						std::vector<DTreeNode>    dTreeNodesSampling(gpuDTreeNodeCntSampling);
						//GPU Upload Memory
						{
							auto desc = RTLib::Ext::CUDA::CUDABufferCreateDesc();
							desc.sizeInBytes = sizeof(sTreeNodes[0]) * std::size(sTreeNodes);
							desc.pData       = nullptr;

							if (m_GpuSTreeNodes) {
								if (m_GpuSTreeNodes->GetSizeInBytes() != desc.sizeInBytes)
								{
									if (!isNeededSync) { 
										if (stream) {
											stream->Synchronize();
										}
									}
									m_GpuSTreeNodes->Destroy();
									m_GpuSTreeNodes = CUDABufferPtr(m_Context->CreateBuffer(desc));
									isNeededSync = true;
								}
							}
							else {
								if (!isNeededSync) {
									if (stream) {
										stream->Synchronize();
									}
								}
								m_GpuSTreeNodes = CUDABufferPtr(m_Context->CreateBuffer(desc));
								isNeededSync = true;
							}
						}
						{
							auto desc = RTLib::Ext::CUDA::CUDABufferCreateDesc();
							desc.sizeInBytes   = sizeof(dTreeWrappers[0]) * std::size(dTreeWrappers);
							desc.pData         = nullptr;

							if (m_GpuDTreeWrappers) {
								if (m_GpuDTreeWrappers->GetSizeInBytes() != desc.sizeInBytes)
								{
									if (!isNeededSync) {
										if (stream) {
											stream->Synchronize();
										}
									}
									m_GpuDTreeWrappers->Destroy();
									m_GpuDTreeWrappers = CUDABufferPtr(m_Context->CreateBuffer(desc));
									isNeededSync = true;
								}
							}
							else {
								if (!isNeededSync) {
									if (stream) {
										stream->Synchronize();
									}
								}
								m_GpuDTreeWrappers = CUDABufferPtr(m_Context->CreateBuffer(desc));
								isNeededSync = true;
							}
						}
						{
							auto desc = RTLib::Ext::CUDA::CUDABufferCreateDesc();
							desc.sizeInBytes        = sizeof(dTreeNodesBuilding[0]) * std::size(dTreeNodesBuilding);
							desc.pData              = nullptr;

							if (m_GpuDTreeNodesBuilding) {
								if (m_GpuDTreeNodesBuilding->GetSizeInBytes() != desc.sizeInBytes)
								{
									if (!isNeededSync) {
										if (stream) {
											stream->Synchronize();
										}
									}
									m_GpuDTreeNodesBuilding->Destroy();
									m_GpuDTreeNodesBuilding = CUDABufferPtr(m_Context->CreateBuffer(desc));
									isNeededSync = true;
								}
							}
							else {
								if (!isNeededSync) {
									if (stream) {
										stream->Synchronize();
									}
								}
								m_GpuDTreeNodesBuilding = CUDABufferPtr(m_Context->CreateBuffer(desc));
								isNeededSync = true;
							}
						}
						{
							auto desc = RTLib::Ext::CUDA::CUDABufferCreateDesc();
							desc.sizeInBytes        = sizeof(dTreeNodesSampling[0]) * std::size(dTreeNodesSampling);
							desc.pData              = nullptr;
							m_GpuDTreeNodesSampling = CUDABufferPtr(m_Context->CreateBuffer(desc));

							if (m_GpuDTreeNodesSampling) {
								if (m_GpuDTreeNodesSampling->GetSizeInBytes() != desc.sizeInBytes)
								{
									if (!isNeededSync) {
										if (stream) {
											stream->Synchronize();
										}
									}
									m_GpuDTreeNodesSampling->Destroy();
									m_GpuDTreeNodesSampling = CUDABufferPtr(m_Context->CreateBuffer(desc));
									isNeededSync = true;
								}
							}
							else {
								if (!isNeededSync) {
									if (stream) {
										stream->Synchronize();
									}
								}
								m_GpuDTreeNodesSampling = CUDABufferPtr(m_Context->CreateBuffer(desc));
								isNeededSync = true;
							}
						}
						{
							size_t dTreeIndex = 0;
							size_t dTreeNodeOffsetBuilding = 0;
							size_t dTreeNodeOffsetSampling = 0;
							for (size_t i = 0; i < gpuSTreeNodeCnt; ++i) {
								sTreeNodes[i] = m_CpuSTree.Node(i).GetGpuHandle();
								if (m_CpuSTree.Node(i).isLeaf) {
									//DTREE
									sTreeNodes[i].dTree = RTLib::Ext::CUDA::CUDANatives::GetGpuAddress<DTreeWrapper>(m_GpuDTreeWrappers.get()) + dTreeIndex;
									//Optimizer
									//BUILDING
									dTreeWrappers[dTreeIndex].building       = m_CpuSTree.Node(i).dTree.building.GetGpuHandle();
									dTreeWrappers[dTreeIndex].building.nodes = RTLib::Ext::CUDA::CUDANatives::GetGpuAddress<DTreeNode>(m_GpuDTreeNodesBuilding.get()) + dTreeNodeOffsetBuilding;
									for (size_t j = 0; j < m_CpuSTree.Node(i).dTree.building.GetNumNodes(); ++j) {
										//SUM
										dTreeNodesBuilding[dTreeNodeOffsetBuilding + j] = m_CpuSTree.Node(i).dTree.building.Node(j);
									}
									dTreeNodeOffsetBuilding += m_CpuSTree.Node(i).dTree.building.GetNumNodes();
									//SAMPLING
									dTreeWrappers[dTreeIndex].sampling       = m_CpuSTree.Node(i).dTree.sampling.GetGpuHandle();
									dTreeWrappers[dTreeIndex].sampling.nodes = RTLib::Ext::CUDA::CUDANatives::GetGpuAddress<DTreeNode>(m_GpuDTreeNodesSampling.get()) + dTreeNodeOffsetSampling;
									for (size_t j = 0; j < m_CpuSTree.Node(i).dTree.sampling.GetNumNodes(); ++j) {
										//SUMS
										dTreeNodesSampling[dTreeNodeOffsetSampling + j] = m_CpuSTree.Node(i).dTree.sampling.Node(j);
									}
									dTreeNodeOffsetSampling += m_CpuSTree.Node(i).dTree.sampling.GetNumNodes();
									dTreeIndex++;
								}
								else {
									sTreeNodes[i].dTree = nullptr;
								}
							}
						}
						//Upload
						if (!stream) {
							auto memoryBufferCopy = RTLib::Ext::CUDA::CUDAMemoryBufferCopy();
							memoryBufferCopy.size      = m_GpuSTreeNodes->GetSizeInBytes();
							memoryBufferCopy.srcData   = sTreeNodes.data();
							memoryBufferCopy.dstOffset = 0;
							RTLIB_CORE_ASSERT_IF_FAILED(m_Context->CopyMemoryToBuffer(m_GpuSTreeNodes.get(), { memoryBufferCopy }));
							memoryBufferCopy.size      = m_GpuDTreeWrappers->GetSizeInBytes();
							memoryBufferCopy.srcData   = dTreeWrappers.data();
							memoryBufferCopy.dstOffset = 0;
							RTLIB_CORE_ASSERT_IF_FAILED(m_Context->CopyMemoryToBuffer(m_GpuDTreeWrappers.get(), { memoryBufferCopy }));
							memoryBufferCopy.size      = m_GpuDTreeNodesBuilding->GetSizeInBytes();
							memoryBufferCopy.srcData   = dTreeNodesBuilding.data();
							memoryBufferCopy.dstOffset = 0;
							RTLIB_CORE_ASSERT_IF_FAILED(m_Context->CopyMemoryToBuffer(m_GpuDTreeNodesBuilding.get(), { memoryBufferCopy }));
							memoryBufferCopy.size      = m_GpuDTreeNodesSampling->GetSizeInBytes();
							memoryBufferCopy.srcData   = dTreeNodesSampling.data();
							memoryBufferCopy.dstOffset = 0;
							RTLIB_CORE_ASSERT_IF_FAILED(m_Context->CopyMemoryToBuffer(m_GpuDTreeNodesSampling.get(), { memoryBufferCopy }));
						}
						else {
							auto memoryBufferCopy = RTLib::Ext::CUDA::CUDAMemoryBufferCopy();
							memoryBufferCopy.size = m_GpuSTreeNodes->GetSizeInBytes();
							memoryBufferCopy.srcData = sTreeNodes.data();
							memoryBufferCopy.dstOffset = 0;
							RTLIB_CORE_ASSERT_IF_FAILED(stream->CopyMemoryToBuffer(m_GpuSTreeNodes.get(), { memoryBufferCopy }));
							memoryBufferCopy.size = m_GpuDTreeWrappers->GetSizeInBytes();
							memoryBufferCopy.srcData = dTreeWrappers.data();
							memoryBufferCopy.dstOffset = 0;
							RTLIB_CORE_ASSERT_IF_FAILED(stream->CopyMemoryToBuffer(m_GpuDTreeWrappers.get(), { memoryBufferCopy }));
							memoryBufferCopy.size = m_GpuDTreeNodesBuilding->GetSizeInBytes();
							memoryBufferCopy.srcData = dTreeNodesBuilding.data();
							memoryBufferCopy.dstOffset = 0;
							RTLIB_CORE_ASSERT_IF_FAILED(stream->CopyMemoryToBuffer(m_GpuDTreeNodesBuilding.get(), { memoryBufferCopy }));
							memoryBufferCopy.size = m_GpuDTreeNodesSampling->GetSizeInBytes();
							memoryBufferCopy.srcData = dTreeNodesSampling.data();
							memoryBufferCopy.dstOffset = 0;
							RTLIB_CORE_ASSERT_IF_FAILED(stream->CopyMemoryToBuffer(m_GpuDTreeNodesSampling.get(), { memoryBufferCopy }));
						}
#ifndef NDEBUG
						std::cout << "Upload(Info)\n";
						std::cout << "GpuSTreeNodes          : " << m_GpuSTreeNodes->GetSizeInBytes()         / (1024.0f * 1024.0f) << "MB\n";
						std::cout << "GpuDTreeNodes(Building): " << m_GpuDTreeNodesBuilding->GetSizeInBytes() / (1024.0f * 1024.0f) << "MB\n";
						std::cout << "GpuDTreeNodes(Sampling): " << m_GpuDTreeNodesSampling->GetSizeInBytes() / (1024.0f * 1024.0f) << "MB\n";
#endif
					}
					void Download(CUDA::CUDAStream* stream = nullptr) noexcept {
						//ダウンロードが必要なのはBuildingだけ
						const size_t gpuSTreeNodeCnt = m_CpuSTree.GetNumNodes();
						size_t gpuDTreeCnt = 0;
						size_t gpuDTreeNodeCntBuilding = 0;
						for (size_t i = 0; i < gpuSTreeNodeCnt; ++i) {
							if (m_CpuSTree.Node(i).isLeaf) {
								gpuDTreeCnt++;
								gpuDTreeNodeCntBuilding += m_CpuSTree.Node(i).dTree.building.GetNumNodes();
							}
						}
						std::vector<DTreeWrapper> dTreeWrappers(gpuDTreeCnt);
						std::vector<DTreeNode>    dTreeNodesBuilding(gpuDTreeNodeCntBuilding);
						if(!stream) {
							auto bufferMemoryCopy      = RTLib::Ext::CUDA::CUDABufferMemoryCopy();
							bufferMemoryCopy.size      = m_GpuDTreeWrappers->GetSizeInBytes();
							bufferMemoryCopy.dstData   = dTreeWrappers.data();
							bufferMemoryCopy.srcOffset = 0;
							RTLIB_CORE_ASSERT_IF_FAILED(m_Context->CopyBufferToMemory(m_GpuDTreeWrappers.get(), { bufferMemoryCopy }));
							bufferMemoryCopy.size      = m_GpuDTreeNodesBuilding->GetSizeInBytes();
							bufferMemoryCopy.dstData   = dTreeNodesBuilding.data();
							bufferMemoryCopy.srcOffset = 0;
							RTLIB_CORE_ASSERT_IF_FAILED(m_Context->CopyBufferToMemory(m_GpuDTreeNodesBuilding.get(), { bufferMemoryCopy }));
						}
						else {
							auto bufferMemoryCopy = RTLib::Ext::CUDA::CUDABufferMemoryCopy();
							bufferMemoryCopy.size = m_GpuDTreeWrappers->GetSizeInBytes();
							bufferMemoryCopy.dstData = dTreeWrappers.data();
							bufferMemoryCopy.srcOffset = 0;
							RTLIB_CORE_ASSERT_IF_FAILED(stream->CopyBufferToMemory(m_GpuDTreeWrappers.get(), { bufferMemoryCopy }));
							bufferMemoryCopy.size = m_GpuDTreeNodesBuilding->GetSizeInBytes();
							bufferMemoryCopy.dstData = dTreeNodesBuilding.data();
							bufferMemoryCopy.srcOffset = 0;
							RTLIB_CORE_ASSERT_IF_FAILED(stream->CopyBufferToMemory(m_GpuDTreeNodesBuilding.get(), { bufferMemoryCopy }));
						}
						{
							size_t cpuDTreeIndex = 0;
							size_t cpuDTreeNodeOffsetBuilding = 0;
							for (size_t i = 0; i < gpuSTreeNodeCnt; ++i) {
								if (m_CpuSTree.Node(i).isLeaf) {
									m_CpuSTree.Node(i).dTree.building.SetGpuHandle(dTreeWrappers[cpuDTreeIndex].building);
									for (size_t j = 0; j < m_CpuSTree.Node(i).dTree.building.GetNumNodes(); ++j) {
										//SUMS
										m_CpuSTree.Node(i).dTree.building.Node(j) = dTreeNodesBuilding[cpuDTreeNodeOffsetBuilding + j];
									}
									cpuDTreeNodeOffsetBuilding += m_CpuSTree.Node(i).dTree.building.GetNumNodes();
									cpuDTreeIndex++;
								}
							}
						}
					}
					void Destroy() {
						if (m_GpuSTreeNodes) {
							m_GpuSTreeNodes->Destroy();
							m_GpuSTreeNodes.reset();
						}
						if (m_GpuDTreeWrappers) {
							m_GpuDTreeWrappers->Destroy();
							m_GpuDTreeWrappers.reset();
						}
						if (m_GpuDTreeNodesBuilding) {
							m_GpuDTreeNodesBuilding->Destroy();
							m_GpuDTreeNodesBuilding.reset();
						}
						if (m_GpuDTreeNodesSampling) {
							m_GpuDTreeNodesSampling->Destroy();
							m_GpuDTreeNodesSampling.reset();
						}
					}
					void Clear() {
						m_CpuSTree = RTSTree(m_CpuSTree.GetAabbMin(), m_CpuSTree.GetAabbMax());
					}
					auto GetGpuHandle()const noexcept -> STree {
						STree sTree;
						sTree.aabbMax  = m_CpuSTree.GetAabbMax();
						sTree.aabbMin  = m_CpuSTree.GetAabbMin();
						sTree.nodes    = RTLib::Ext::CUDA::CUDANatives::GetGpuAddress<STreeNode>(m_GpuSTreeNodes.get());
						sTree.fraction = m_Fraction;
						return sTree;
					}
					void Reset(int iter, int samplePerPasses) {
						if (iter <= 0) {
							return;
						}
						size_t sTreeTh = std::sqrt(std::pow(2.0, iter) * samplePerPasses / 4.0f) * 4000;
						m_CpuSTree.Refine(sTreeTh, 2000);
						for (int i = 0; i < m_CpuSTree.GetNumNodes(); ++i) {
							if (m_CpuSTree.Node(i).isLeaf) {
								m_CpuSTree.Node(i).dTree.Reset(m_MaxDTreeDepth, 0.01);
							}
						}
					}
					void Build() {
						size_t bestIdx = 0;
						for (int i = 0; i < m_CpuSTree.GetNumNodes(); ++i) {
							if (m_CpuSTree.Node(i).isLeaf) {
								m_CpuSTree.Node(i).dTree.Build();
							}
						}
						int   maxDepth = 0;
						int   minDepth = std::numeric_limits<int>::max();
						float avgDepth = 0.0f;
						float maxAvgRadiance = 0.0f;
						float minAvgRadiance = std::numeric_limits<float>::max();
						float avgAvgRadiance = 0.0f;
						size_t maxNodes = 0;
						size_t minNodes = std::numeric_limits<size_t>::max();
						float avgNodes = 0.0f;
						float maxStatisticalWeight = 0;
						float minStatisticalWeight = std::numeric_limits<float>::max();
						float avgStatisticalWeight = 0;

						int nPoints = 0;
						int nPointsNodes = 0;
						bool isSaved = false;
						for (int i = 0; i < m_CpuSTree.GetNumNodes(); ++i) {
							if (m_CpuSTree.Node(i).isLeaf) {
								auto& dTree = m_CpuSTree.Node(i).dTree;
								//printf("Area = %f\n", dTree.sampling.GetArea());
								const int depth = dTree.GetDepth();
								maxDepth = std::max<int>(maxDepth, depth);
								minDepth = std::min<int>(minDepth, depth);
								avgDepth += depth;

								const float avgRadiance = dTree.GetMean();
								maxAvgRadiance = std::max<float>(maxAvgRadiance, avgRadiance);
								minAvgRadiance = std::min<float>(minAvgRadiance, avgRadiance);
								avgAvgRadiance += avgRadiance;

								if (dTree.GetNumNodes() >= 1) {

									const size_t numNodes = dTree.GetNumNodes();
									maxNodes = std::max<size_t>(maxNodes, numNodes);
									minNodes = std::min<size_t>(minNodes, numNodes);
									avgNodes += numNodes;
									++nPointsNodes;
								}

								const auto statisticalWeight = dTree.GetStatisticalWeightSampling();
								maxStatisticalWeight = std::max<float>(maxStatisticalWeight, statisticalWeight);
								minStatisticalWeight = std::min<float>(minStatisticalWeight, statisticalWeight);
								avgStatisticalWeight += statisticalWeight;

								++nPoints;
							}
						}

						if (nPoints > 0) {
							avgDepth /= nPoints;
							avgAvgRadiance /= nPoints;
							if (nPointsNodes) {
								avgNodes /= nPointsNodes;
							}
							avgStatisticalWeight /= nPoints;
						}
#if 0
						std::cout << "SDTree Build Statistics\n";
						std::cout << "Depth(STree):      " << m_CpuSTree.GetDepth() << std::endl;
						std::cout << "Depth(DTree):      " << minDepth << "," << avgDepth << "," << maxDepth << std::endl;
						std::cout << "Node count:        " << minNodes << "," << avgNodes << "," << maxNodes << std::endl;
						std::cout << "Mean Radiance:     " << minAvgRadiance << "," << avgAvgRadiance << "," << maxAvgRadiance << std::endl;
						std::cout << "statisticalWeight: " << minStatisticalWeight << "," << avgStatisticalWeight << "," << maxStatisticalWeight << std::endl;
#endif
					}
					auto GetMemoryFootPrint()const noexcept -> size_t
					{
						return GetSTreeMemoryFootPrint() + GetDTreeMemoryFootPrint() + (m_GpuDTreeWrappers ? m_GpuDTreeWrappers->GetSizeInBytes() : 0);
					}
					auto GetSTreeMemoryFootPrint()const noexcept -> size_t
					{
						return sizeof(STree) + (m_GpuSTreeNodes ? m_GpuSTreeNodes->GetSizeInBytes() : 0);
					}
					auto GetDTreeMemoryFootPrint()const noexcept -> size_t
					{
						return GetDTreeMemoryFootPrintBuilding() + GetDTreeMemoryFootPrintSampling() + (m_GpuDTreeWrappers ? m_GpuDTreeWrappers->GetSizeInBytes() : 0);
					}
					auto GetDTreeMemoryFootPrintBuilding()const noexcept -> size_t
					{
						return m_GpuDTreeNodesBuilding ? m_GpuDTreeNodesBuilding->GetSizeInBytes() : 0;
					}
					auto GetDTreeMemoryFootPrintSampling()const noexcept -> size_t
					{
						return m_GpuDTreeNodesSampling ? m_GpuDTreeNodesSampling->GetSizeInBytes() : 0;
					}
					void Dump(std::string filename) {
						std::fstream jsonFile(filename, std::ios::binary | std::ios::out);
						jsonFile << "{\n";
						jsonFile << "\"STree\":\n";
						m_CpuSTree.Dump(jsonFile);
						jsonFile << "}\n";
						jsonFile.close();
					}
				private:
					RTLib::Ext::CUDA::CUDAContext* m_Context = nullptr;
					RTSTree         m_CpuSTree;
					CUDABufferPtr   m_GpuSTreeNodes         = {};//����
					CUDABufferPtr   m_GpuDTreeWrappers      = {};//���L
					CUDABufferPtr   m_GpuDTreeNodesBuilding = {};//������
					CUDABufferPtr   m_GpuDTreeNodesSampling = {};//������
					unsigned int    m_MaxDTreeDepth         = 0;
					float           m_Fraction              = 0.0f;

				};

				template<unsigned int kSTreeStackDepth, unsigned int kDTreeStackDepth>
				class  RTSTreeControllerT
				{
				public:
					using RTSTreeController = RTSTreeControllerT<kSTreeStackDepth, kDTreeStackDepth>;
					using RTSTreeWrapper    = RTSTreeWrapperT<kSTreeStackDepth, kDTreeStackDepth>;
					using RTSTree           = RTSTreeT<kSTreeStackDepth, kDTreeStackDepth>;
					using RTSTreeNode       = RTSTreeNodeT<kSTreeStackDepth, kDTreeStackDepth>;
					using RTDTreeWrapper    = RTDTreeWrapperT<kDTreeStackDepth>;
					using RTDTree           = RTDTreeT<kDTreeStackDepth>;
					using RTDTreeNode       = RTDTreeNodeT<kDTreeStackDepth>;

					using STree        = STreeT<kSTreeStackDepth, kDTreeStackDepth>;
					using STreeNode    = STreeNodeT<kSTreeStackDepth, kDTreeStackDepth>;
					using DTreeWrapper = DTreeWrapperT<kDTreeStackDepth>;
					using DTree        = DTreeT<kDTreeStackDepth>;
					using DTreeNode    = DTreeNodeT<kDTreeStackDepth>;

					enum TraceState
					{
						TraceStateRecord          = 0,
						TraceStateRecordAndSample = 1,
						TraceStateSample          = 2,
					};

					RTSTreeControllerT(RTSTreeWrapper* sTree, 
						unsigned int sampleForBudget  /*ALL SAMPLES FOR TRACE*/,
						unsigned int iterationForBuilt/*ITERATION FOR BUILT*/=0,
						float         ratioForBudget  /*RATIO FOR RECORDING TREE*/=0.5f,
						unsigned int samplePerLaunch  /*SAMPLES PER LAUNCH*/ = 1
					)noexcept
						:m_STree{ sTree }, m_SampleForBudget{ sampleForBudget }, m_SamplePerLaunch{ samplePerLaunch }, m_IterationForBuilt{ iterationForBuilt }, m_RatioForBudget{ ratioForBudget }{}

					void SetSampleForBudget(unsigned int sampleForBudget)noexcept{	m_SampleForBudget = sampleForBudget;}

					void Start() {
						m_TraceStart = true;
					}

					void BegTrace(CUDA::CUDAStream* stream = nullptr) {
						if (!m_STree) {
							return;
						}
						if ( m_TraceStart) {

							m_SamplePerAll = 0;
							m_SamplePerTmp = 0;
							m_CurIteration = 0;

							m_TraceStart     = false;
							m_TraceExecuting = true;

							m_SampleForRemain = ((m_SampleForBudget - 1 + m_SamplePerLaunch) / m_SamplePerLaunch) * m_SamplePerLaunch;
							m_SampleForPass = 0;

							m_TraceState = TraceStateRecord;

							m_STree->Destroy();
							m_STree->Clear();
							m_STree->Upload(stream);
						}
						if (!m_TraceExecuting) {
							return;
						}

						if (m_SamplePerTmp == 0)
						{
							m_SampleForRemain -= m_SampleForPass;
							m_SampleForPass    = std::min<uint32_t>(m_SampleForRemain, (1 << m_CurIteration) * m_SamplePerLaunch);
							if ((m_SampleForRemain - m_SampleForPass < 2 * m_SampleForPass) ||
								(m_SamplePerAll   >= m_RatioForBudget * static_cast<float>(m_SampleForBudget))) {
								std::cout << "Final: this->m_SamplePerAll=" << m_SamplePerAll << std::endl;
								m_SampleForPass = m_SampleForRemain;
							}
							std::cout << "SampleForPass: " << m_SampleForPass << "vs SampleForRemain" << m_SampleForRemain << std::endl;
							if (m_SampleForRemain > m_SampleForPass) {
								m_STree->Download(stream);
								m_STree->Reset(m_CurIteration, m_SampleForPass);
								m_STree->Upload(stream);
							}
						}
						if (m_CurIteration > m_IterationForBuilt) {
							if (m_SampleForRemain > m_SampleForPass) {
								m_TraceState = TraceStateRecordAndSample;
							}
							else {
								m_TraceState = TraceStateSample;
							}
						}
						else {
							m_TraceState = TraceStateRecord;
						}
#ifndef NDEBUG
						std::cout << "CurIteration: " << m_CurIteration << " SamplePerTmp: " << m_SamplePerTmp << std::endl;
#endif

					}

					void EndTrace(CUDA::CUDAStream* stream = nullptr) {
						if (!m_STree || !m_TraceExecuting) {
							return;
						}
						m_SamplePerAll += m_SamplePerLaunch;
						m_SamplePerTmp += m_SamplePerLaunch;

						if (m_SamplePerTmp >= m_SampleForPass)
						{
							m_STree->Download(stream);
							m_STree->Build();
							m_STree->Upload(stream);

							m_CurIteration++;
							m_SamplePerTmp = 0;
						}
						if (m_SamplePerAll > m_SampleForBudget) {
							m_TraceExecuting = false;
							m_SamplePerAll   = 0;
						}
					}

					auto GetGpuSTree()const noexcept -> STree
					{
						return m_STree->GetGpuHandle();
					}

					auto GetState()const noexcept -> TraceState {
						return m_TraceState;
					}
				private:
					RTSTreeWrapper* m_STree             = nullptr;
					unsigned int    m_SampleForBudget   = 0;
					unsigned int    m_SamplePerLaunch   = 0;
					unsigned int    m_IterationForBuilt = 0;
					float           m_RatioForBudget    = 0.0f;

					bool            m_TraceStart      = false;
					bool            m_TraceExecuting  = false;
					unsigned int    m_SamplePerAll    = 0;
					unsigned int    m_SamplePerTmp    = 0;
					unsigned int    m_SampleForRemain = 0;
					unsigned int    m_SampleForPass   = 0;
					TraceState      m_TraceState      = TraceStateRecord;
					unsigned int    m_CurIteration    = 0;
				};

				template<unsigned int kSTreeStackDepth, unsigned int kDTreeStackDepth>
				struct RTSTreeNode2T {
					using RTSTreeNode2   = RTSTreeNode2T<kSTreeStackDepth, kDTreeStackDepth>;
					using RTDTreeWrapper = RTDTreeWrapperT<kDTreeStackDepth>;
					using RTDTree        = RTDTreeT<kDTreeStackDepth>;
					using RTDTreeNode    = RTDTreeNodeT<kDTreeStackDepth>;

					using STreeNode2   = STreeNode2T<kSTreeStackDepth, kDTreeStackDepth>;
					using DTreeWrapper = DTreeWrapperT<kDTreeStackDepth>;
					using DTree        = DTreeT<kDTreeStackDepth>;
					using DTreeNode    = DTreeNodeT<kDTreeStackDepth>;

					RTSTreeNode2T()noexcept : dTree(), isLeaf{ true }, children{}{}
					auto GetChildIdx(float3& p)const noexcept -> int {
						int idx = 0;
						if (p.x >= 0.5f) {
							idx |= (1 << 0);
							p.x *= 2.0f;
							p.x -= 1.0f;
						}
						else {
							p.x *= 2.0f;
						}
						if (p.y >= 0.5f) {
							idx |= (1 << 1);
							p.y *= 2.0f;
							p.y -= 1.0f;
						}
						else {
							p.y *= 2.0f;
						}
						if (p.z >= 0.5f) {
							idx |= (1 << 2);
							p.z *= 2.0f;
							p.z -= 1.0f;
						}
						else {
							p.z *= 2.0f;
						}
						return idx;
					}
					auto GetNodeIdx(float3& p)const noexcept -> unsigned int {
						return children[GetChildIdx(p)];
					}
					auto GetDTree(float3 p, float3& size, const std::vector<RTSTreeNode2>& nodes)const noexcept -> const RTDTreeWrapper* {
						const RTSTreeNode2* cur = this;
						int   ndx = cur->GetNodeIdx(p);
						int   depth = 1;
						while (true) {
							if (cur->isLeaf) {
								return &cur->dTree;
							}
							size /= 2.0f;
							cur = &nodes[ndx];
							ndx = cur->GetNodeIdx(p);
							depth++;
						}
						return nullptr;
					}
					auto GetDTreeWrapper()const noexcept -> const RTDTreeWrapper* {
						return &dTree;
					}
					auto GetDepth(const std::vector<RTSTreeNode2>& nodes)const-> int {
						int result = 1;
						if (isLeaf) {
							return 1;
						}
						for (int i = 0; i < 2; ++i) {
							result = std::max(result, 1 + nodes[children[i]].GetDepth(nodes));
						}
						return result;
					}
					void Dump(std::fstream& jsonFile, size_t sTreeNodeIdx, const std::vector<RTSTreeNode2>& nodes)const noexcept {
					}
					void SetGpuHandle(const STreeNode2& sTreeNode)noexcept
					{
						isLeaf = sTreeNode.IsLeaf();
						children[0] = sTreeNode.children[0];
						children[1] = sTreeNode.children[1];
						children[2] = sTreeNode.children[2];
						children[3] = sTreeNode.children[3];
						children[4] = sTreeNode.children[4];
						children[5] = sTreeNode.children[5];
						children[6] = sTreeNode.children[6];
						children[7] = sTreeNode.children[7];
					}
					auto GetGpuHandle()const noexcept -> STreeNode2
					{
						STreeNode2 sTreeNode;
						sTreeNode.children[0] = children[0];
						sTreeNode.children[1] = children[1];
						sTreeNode.children[2] = children[2];
						sTreeNode.children[3] = children[3];
						sTreeNode.children[4] = children[4];
						sTreeNode.children[5] = children[5];
						sTreeNode.children[6] = children[6];
						sTreeNode.children[7] = children[7];
						sTreeNode.dTree = nullptr;
						return sTreeNode;
					}

					RTDTreeWrapper           dTree;
					bool                     isLeaf;
					unsigned int             children[8];
				};

				template<unsigned int kSTreeStackDepth, unsigned int kDTreeStackDepth>
				class  RTSTree2T {
				public:
					using RTSTree2       = RTSTree2T<kSTreeStackDepth, kDTreeStackDepth>;
					using RTSTreeNode2   = RTSTreeNode2T<kSTreeStackDepth, kDTreeStackDepth>;
					using RTDTreeWrapper = RTDTreeWrapperT<kDTreeStackDepth>;
					using RTDTree        = RTDTreeT<kDTreeStackDepth>;
					using RTDTreeNode    = RTDTreeNodeT<kDTreeStackDepth>;

					using STree2       = STree2T<kSTreeStackDepth, kDTreeStackDepth>;
					using STreeNode2   = STreeNode2T<kSTreeStackDepth, kDTreeStackDepth>;
					using DTreeWrapper = DTreeWrapperT<kDTreeStackDepth>;
					using DTree        = DTreeT<kDTreeStackDepth>;
					using DTreeNode    = DTreeNodeT<kDTreeStackDepth>;

					using CUDABuffer    = RTLib::Ext::CUDA::CUDABuffer;
					using CUDABufferPtr = std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>;
				public:
					RTSTree2T(const float3& aabbMin, const float3& aabbMax)noexcept {
						this->Clear();
						auto size = aabbMax - aabbMin;
						auto maxSize = std::max(std::max(size.x, size.y), size.z);
						m_AabbMin = aabbMin;
						m_AabbMax = aabbMin + make_float3(maxSize);
					}
					void Clear()noexcept {
						m_Nodes.clear();
						m_Nodes.emplace_back();
					}
					void SubDivideAll() {
						int nNodes = m_Nodes.size();
						for (size_t i = 0; i < nNodes; ++i)
						{
							if (m_Nodes[i].isLeaf) {
								SubDivide(i, m_Nodes);
							}
						}
					}
					void SubDivide(int nodeIdx, std::vector<RTSTreeNode2>& nodes)
					{
						size_t curNodeIdx = nodes.size();
						nodes.resize(curNodeIdx + 8);
						auto& cur = nodes[nodeIdx];
						for (int i = 0; i < 8; ++i)
						{
							uint32_t idx = curNodeIdx + i;
							cur.children[i] = idx;
							nodes[idx].isLeaf = true;
							nodes[idx].dTree = cur.dTree;
							nodes[idx].dTree.building.SetStatisticalWeight(cur.dTree.building.GetStatisticalWeight() / 8.0f);
						}
						cur.isLeaf = false;
						cur.dTree = {};
					}
					auto GetDTree(float3 p, float3& size)const noexcept -> const RTDTreeWrapper* {
						size = m_AabbMax - m_AabbMin;
						p = p - m_AabbMin;
						p /= size;
						return m_Nodes[0].GetDTree(p, size, m_Nodes);
					}
					auto GetDTree(const float3& p)const noexcept ->const RTDTreeWrapper* {
						float3 size;
						return GetDTree(p, size);
					}
					auto GetDepth()const -> int {
						return m_Nodes[0].GetDepth(m_Nodes);
					}
					auto Node(size_t idx)const noexcept -> const RTSTreeNode2& {
						return m_Nodes[idx];
					}
					auto Node(size_t idx) noexcept -> RTSTreeNode2& {
						return m_Nodes[idx];
					}
					auto GetNumNodes()const noexcept -> size_t {
						return m_Nodes.size();
					}
					bool ShallSplit(const RTSTreeNode2& node, int depth, size_t samplesRequired)const noexcept
					{
						//std::cout << node.dTree.GetStatisticalWeight() << "vs " << samplesRequired << std::endl;
						return m_Nodes.size() < std::numeric_limits<uint32_t>::max() - 1 && node.dTree.building.GetStatisticalWeight() > samplesRequired;
					}
					void Refine(size_t sTreeTh, int maxMB) {
						if (maxMB >= 0) {
							size_t approxMemoryFootPrint = 0;
							for (const auto& node : m_Nodes)
							{
								approxMemoryFootPrint += node.GetDTreeWrapper()->GetApproxMemoryFootPrint();
							}
							if (approxMemoryFootPrint / 1000000 >= maxMB) {
								return;
							}
						}

						struct StackNode {
							size_t index;
							int    depth;
						};
						std::stack<StackNode> nodeIndices = {};
						nodeIndices.push({ 0, 1 });
						while (!nodeIndices.empty())
						{
							StackNode sNode = nodeIndices.top();
							nodeIndices.pop();

							if (m_Nodes[sNode.index].isLeaf) {
								if (ShallSplit(m_Nodes[sNode.index], sNode.depth, sTreeTh)) {
									SubDivide((int)sNode.index, m_Nodes);
								}
							}

							if (!m_Nodes[sNode.index].isLeaf) {
								const auto& node = m_Nodes[sNode.index];
								for (int i = 0; i < 8; ++i) {
									nodeIndices.push({ node.children[i],sNode.depth + 1 });
								}
							}
						}
					}
					auto GetAabbMin()const -> float3 {
						return m_AabbMin;
					}
					auto GetAabbMax()const -> float3 {
						return m_AabbMax;
					}
					void Dump(std::fstream& jsonFile)const noexcept {
						jsonFile << "{\n";
						jsonFile << "\"aabbMin\" : [" << m_AabbMin.x << ", " << m_AabbMin.y << ", " << m_AabbMin.z << "],\n";
						jsonFile << "\"aabbMax\" : [" << m_AabbMax.x << ", " << m_AabbMax.y << ", " << m_AabbMax.z << "],\n";
						jsonFile << "\"root\"    : \n";
						m_Nodes[0].Dump(jsonFile, 0, m_Nodes);
						jsonFile << "\n";
						jsonFile << "}\n";
					}
				private:
					std::vector<RTSTreeNode2> m_Nodes;
					float3                    m_AabbMin;
					float3                    m_AabbMax;
				};

				template<unsigned int kSTreeStackDepth, unsigned int kDTreeStackDepth>
				class  RTSTreeWrapper2T {
				public:
					using RTSTreeWrapper2= RTSTreeWrapper2T<kSTreeStackDepth, kDTreeStackDepth>;
					using RTSTree2       = RTSTree2T<kSTreeStackDepth, kDTreeStackDepth>;
					using RTSTreeNode2   = RTSTreeNode2T<kSTreeStackDepth, kDTreeStackDepth>;
					using RTDTreeWrapper = RTDTreeWrapperT<kDTreeStackDepth>;
					using RTDTree        = RTDTreeT<kDTreeStackDepth>;
					using RTDTreeNode    = RTDTreeNodeT<kDTreeStackDepth>;

					using STree2       = STree2T<kSTreeStackDepth, kDTreeStackDepth>;
					using STreeNode2   = STreeNode2T<kSTreeStackDepth, kDTreeStackDepth>;
					using DTreeWrapper = DTreeWrapperT<kDTreeStackDepth>;
					using DTree        = DTreeT<kDTreeStackDepth>;
					using DTreeNode    = DTreeNodeT<kDTreeStackDepth>;

					using CUDABuffer    = RTLib::Ext::CUDA::CUDABuffer;
					using CUDABufferPtr = std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>;
				public:
					RTSTreeWrapper2T(const float3& aabbMin, const float3& aabbMax, unsigned int maxDTreeDepth = 20)noexcept :m_CpuSTree{ aabbMin,aabbMax }, m_MaxDTreeDepth{ maxDTreeDepth }{}
					void Upload()noexcept {
						//Uploadは両方必要
						const size_t gpuSTreeNodeCnt = m_CpuSTree.GetNumNodes();
						size_t gpuDTreeCnt = 0;
						size_t gpuDTreeNodeCntBuilding = 0;
						size_t gpuDTreeNodeCntSampling = 0;
						for (size_t i = 0; i < gpuSTreeNodeCnt; ++i) {
							if (m_CpuSTree.Node(i).isLeaf) {
								gpuDTreeCnt++;
								gpuDTreeNodeCntBuilding += m_CpuSTree.Node(i).dTree.building.GetNumNodes();
								gpuDTreeNodeCntSampling += m_CpuSTree.Node(i).dTree.sampling.GetNumNodes();
							}
						}
						//CPU Upload Memory
						std::vector<STreeNode2>   sTreeNodes(gpuSTreeNodeCnt);
						std::vector<DTreeWrapper> dTreeWrappers(gpuDTreeCnt);
						std::vector<DTreeNode>    dTreeNodesBuilding(gpuDTreeNodeCntBuilding);
						std::vector<DTreeNode>    dTreeNodesSampling(gpuDTreeNodeCntSampling);
						//GPU Upload Memory
						m_GpuSTreeNodes.resize(sTreeNodes.size());
						m_GpuDTreeWrappers.resize(dTreeWrappers.size());
						m_GpuDTreeNodesBuilding.resize(dTreeNodesBuilding.size());
						m_GpuDTreeNodesSampling.resize(dTreeNodesSampling.size());
						{
							size_t dTreeIndex = 0;
							size_t dTreeNodeOffsetBuilding = 0;
							size_t dTreeNodeOffsetSampling = 0;
							for (size_t i = 0; i < gpuSTreeNodeCnt; ++i) {
								sTreeNodes[i] = m_CpuSTree.Node(i).GetGpuHandle();
								if (m_CpuSTree.Node(i).isLeaf) {
									//DTREE
									sTreeNodes[i].dTree = m_GpuDTreeWrappers.getDevicePtr() + dTreeIndex;
									//BUILDING
									dTreeWrappers[dTreeIndex].building = m_CpuSTree.Node(i).dTree.building.GetGpuHandle();
									dTreeWrappers[dTreeIndex].building.nodes = m_GpuDTreeNodesBuilding.getDevicePtr() + dTreeNodeOffsetBuilding;
									for (size_t j = 0; j < m_CpuSTree.Node(i).dTree.building.GetNumNodes(); ++j) {
										//SUM
										dTreeNodesBuilding[dTreeNodeOffsetBuilding + j] = m_CpuSTree.Node(i).dTree.building.Node(j);
									}
									dTreeNodeOffsetBuilding += m_CpuSTree.Node(i).dTree.building.GetNumNodes();
									//SAMPLING
									dTreeWrappers[dTreeIndex].sampling = m_CpuSTree.Node(i).dTree.sampling.GetGpuHandle();
									dTreeWrappers[dTreeIndex].sampling.nodes = m_GpuDTreeNodesSampling.getDevicePtr() + dTreeNodeOffsetSampling;
									for (size_t j = 0; j < m_CpuSTree.Node(i).dTree.sampling.GetNumNodes(); ++j) {
										//SUMS
										dTreeNodesSampling[dTreeNodeOffsetSampling + j] = m_CpuSTree.Node(i).dTree.sampling.Node(j);
									}
									dTreeNodeOffsetSampling += m_CpuSTree.Node(i).dTree.sampling.GetNumNodes();
									dTreeIndex++;
								}
								else {
									sTreeNodes[i].dTree = nullptr;
								}
							}
						}
						//Upload
						m_GpuSTreeNodes.upload(sTreeNodes);
						m_GpuDTreeWrappers.upload(dTreeWrappers);
						m_GpuDTreeNodesBuilding.upload(dTreeNodesBuilding);
						m_GpuDTreeNodesSampling.upload(dTreeNodesSampling);
#ifndef NDEBUG
						std::cout << "Upload(Info)\n";
						std::cout << "GpuSTreeNodes          : " << m_GpuSTreeNodes.getSizeInBytes() / (1024.0f * 1024.0f) << "MB\n";
						std::cout << "GpuDTreeNodes(Building): " << m_GpuDTreeNodesBuilding.getSizeInBytes() / (1024.0f * 1024.0f) << "MB\n";
						std::cout << "GpuDTreeNodes(Sampling): " << m_GpuDTreeNodesSampling.getSizeInBytes() / (1024.0f * 1024.0f) << "MB\n";
#endif
					}
					void Download() noexcept {
						//ダウンロードが必要なのはBuildingだけ
						const size_t gpuSTreeNodeCnt = m_CpuSTree.GetNumNodes();
						size_t gpuDTreeCnt = 0;
						size_t gpuDTreeNodeCntBuilding = 0;
						for (size_t i = 0; i < gpuSTreeNodeCnt; ++i) {
							if (m_CpuSTree.Node(i).isLeaf) {
								gpuDTreeCnt++;
								gpuDTreeNodeCntBuilding += m_CpuSTree.Node(i).dTree.building.GetNumNodes();
							}
						}
						std::vector<DTreeWrapper> dTreeWrappers(gpuDTreeCnt);
						std::vector<DTreeNode>    dTreeNodesBuilding(gpuDTreeNodeCntBuilding);
						m_GpuDTreeWrappers.download(dTreeWrappers);
						m_GpuDTreeNodesBuilding.download(dTreeNodesBuilding);
						{
							size_t cpuDTreeIndex = 0;
							size_t cpuDTreeNodeOffsetBuilding = 0;
							for (size_t i = 0; i < gpuSTreeNodeCnt; ++i) {
								if (m_CpuSTree.Node(i).isLeaf) {
									m_CpuSTree.Node(i).dTree.building.SetGpuHandle(dTreeWrappers[cpuDTreeIndex].building);
									for (size_t j = 0; j < m_CpuSTree.Node(i).dTree.building.GetNumNodes(); ++j) {
										//SUMS
										m_CpuSTree.Node(i).dTree.building.Node(j) = dTreeNodesBuilding[cpuDTreeNodeOffsetBuilding + j];
									}
									cpuDTreeNodeOffsetBuilding += m_CpuSTree.Node(i).dTree.building.GetNumNodes();
									cpuDTreeIndex++;
								}
							}
						}
					}
					void Clear() {
						m_CpuSTree = RTSTree2(m_CpuSTree.GetAabbMin(), m_CpuSTree.GetAabbMax());
					}
					auto GetGpuHandle()const noexcept -> STree2 {
						STree2 sTree;
						sTree.aabbMax  = m_CpuSTree.GetAabbMax();
						sTree.aabbMin  = m_CpuSTree.GetAabbMin();
						sTree.nodes    = m_GpuSTreeNodes.getDevicePtr();
						sTree.fraction = 0.5f;
						return sTree;
					}
					void Reset(int iter, int samplePerPasses) {
						if (iter <= 0) {
							return;
						}
						size_t sTreeTh = std::sqrt(std::pow(2.0, iter) * samplePerPasses / 4.0f) * 4000;
						m_CpuSTree.Refine(sTreeTh, 4000);
						for (int i = 0; i < m_CpuSTree.GetNumNodes(); ++i) {
							if (m_CpuSTree.Node(i).isLeaf) {
								m_CpuSTree.Node(i).dTree.Reset(m_MaxDTreeDepth, 0.01);
							}
						}
					}
					void Build() {
						size_t bestIdx = 0;
						for (int i = 0; i < m_CpuSTree.GetNumNodes(); ++i) {
							if (m_CpuSTree.Node(i).isLeaf) {
								m_CpuSTree.Node(i).dTree.Build();
							}
						}
						int   maxDepth = 0;
						int   minDepth = std::numeric_limits<int>::max();
						float avgDepth = 0.0f;
						float maxAvgRadiance = 0.0f;
						float minAvgRadiance = std::numeric_limits<float>::max();
						float avgAvgRadiance = 0.0f;
						size_t maxNodes = 0;
						size_t minNodes = std::numeric_limits<size_t>::max();
						float avgNodes = 0.0f;
						float maxStatisticalWeight = 0;
						float minStatisticalWeight = std::numeric_limits<float>::max();
						float avgStatisticalWeight = 0;

						int nPoints = 0;
						int nPointsNodes = 0;
						bool isSaved = false;
						for (int i = 0; i < m_CpuSTree.GetNumNodes(); ++i) {
							if (m_CpuSTree.Node(i).isLeaf) {
								auto& dTree = m_CpuSTree.Node(i).dTree;
								//printf("Area = %f\n", dTree.sampling.GetArea());
								const int depth = dTree.GetDepth();
								maxDepth = std::max<int>(maxDepth, depth);
								minDepth = std::min<int>(minDepth, depth);
								avgDepth += depth;

								const float avgRadiance = dTree.GetMean();
								maxAvgRadiance = std::max<float>(maxAvgRadiance, avgRadiance);
								minAvgRadiance = std::min<float>(minAvgRadiance, avgRadiance);
								avgAvgRadiance += avgRadiance;

								if (dTree.GetNumNodes() >= 1) {

									const size_t numNodes = dTree.GetNumNodes();
									maxNodes = std::max<size_t>(maxNodes, numNodes);
									minNodes = std::min<size_t>(minNodes, numNodes);
									avgNodes += numNodes;
									++nPointsNodes;
								}

								const auto statisticalWeight = dTree.GetStatisticalWeightSampling();
								maxStatisticalWeight = std::max<float>(maxStatisticalWeight, statisticalWeight);
								minStatisticalWeight = std::min<float>(minStatisticalWeight, statisticalWeight);
								avgStatisticalWeight += statisticalWeight;

								++nPoints;
							}
						}

						if (nPoints > 0) {
							avgDepth /= nPoints;
							avgAvgRadiance /= nPoints;
							if (nPointsNodes) {
								avgNodes /= nPointsNodes;
							}
							avgStatisticalWeight /= nPoints;
						}
#if 0
						std::cout << "SDTree Build Statistics\n";
						std::cout << "Depth(STree):      " << m_CpuSTree.GetDepth() << std::endl;
						std::cout << "Depth(DTree):      " << minDepth << "," << avgDepth << "," << maxDepth << std::endl;
						std::cout << "Node count:        " << minNodes << "," << avgNodes << "," << maxNodes << std::endl;
						std::cout << "Mean Radiance:     " << minAvgRadiance << "," << avgAvgRadiance << "," << maxAvgRadiance << std::endl;
						std::cout << "statisticalWeight: " << minStatisticalWeight << "," << avgStatisticalWeight << "," << maxStatisticalWeight << std::endl;
#endif
					}
					void Dump(std::string filename) {
						std::fstream jsonFile(filename, std::ios::binary | std::ios::out);
						jsonFile << "{\n";
						jsonFile << "\"STree\":\n";
						m_CpuSTree.Dump(jsonFile);
						jsonFile << "}\n";
						jsonFile.close();
					}
				private:
					RTSTree2        m_CpuSTree;
					CUDABufferPtr   m_GpuSTreeNodes = {};//����
					CUDABufferPtr   m_GpuDTreeWrappers = {};//���L
					CUDABufferPtr   m_GpuDTreeNodesBuilding = {};//������
					CUDABufferPtr   m_GpuDTreeNodesSampling = {};//������
					unsigned int    m_MaxDTreeDepth = 0;

				};

				
#endif
				template<unsigned int kSTreeStackDepth, unsigned int kDTreeStackDepth>
				struct PathGuidingTraits
				{
#ifndef __CUDACC__
					using RTSTreeWrapper2 = RTSTreeWrapper2T<kSTreeStackDepth, kDTreeStackDepth>;
					using RTSTree2        = RTSTree2T<kSTreeStackDepth, kDTreeStackDepth>;
					using RTSTreeNode2    = RTSTreeNode2T<kSTreeStackDepth, kDTreeStackDepth>;

					using RTSTreeController = RTSTreeControllerT<kSTreeStackDepth, kDTreeStackDepth>;
					using RTSTreeWrapper    = RTSTreeWrapperT<kSTreeStackDepth, kDTreeStackDepth>;
					using RTSTree           = RTSTreeT<kSTreeStackDepth, kDTreeStackDepth>;
					using RTSTreeNode       = RTSTreeNodeT<kSTreeStackDepth, kDTreeStackDepth>;

					using RTDTreeWrapper = RTDTreeWrapperT<kDTreeStackDepth>;
					using RTDTree        = RTDTreeT<kDTreeStackDepth>;
					using RTDTreeNode    = RTDTreeNodeT<kDTreeStackDepth>;
#endif
					using STree     = STreeT<kSTreeStackDepth, kDTreeStackDepth>;
					using STreeNode = STreeNodeT<kSTreeStackDepth, kDTreeStackDepth>;

					using STree2     = STree2T<kSTreeStackDepth, kDTreeStackDepth>;
					using STreeNode2 = STreeNode2T<kSTreeStackDepth, kDTreeStackDepth>;

					using DTreeWrapper = DTreeWrapperT<kDTreeStackDepth>;
					using DTree        = DTreeT<kDTreeStackDepth>;
					using DTreeNode    = DTreeNodeT<kDTreeStackDepth>;

					using TraceVertex = TraceVertexT<kDTreeStackDepth>;
				};
            }
        }
    }
}
#endif