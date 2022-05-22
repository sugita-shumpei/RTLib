#ifndef RTLIB_EXT_OPX7_OPX7_ACCELERATION_STRUCTURE_BUILD_INPUT_H
#define RTLIB_EXT_OPX7_OPX7_ACCELERATION_STRUCTURE_BUILD_INPUT_H
#include <RTLib/Core/BaseObject.h>
#include <RTLib/Ext/CUDA/CUDABuffer.h>
#include <RTLib/Ext/OPX7/OPX7Common.h>
#include <RTLib/Ext/OPX7/UuidDefinitions.h>
#include <memory>
namespace RTLib {
	namespace Ext
	{
		namespace OPX7
		{
			class OPX7Context;
			class OPX7AccelerationStructureBuildInput : public RTLib::Core::BaseObject
			{
			private:
				RTLIB_CORE_TYPE_OBJECT_DECLARE_DERIVED_METHOD(OPX7AccelerationStructureBuildInput, Core::BaseObject, RTLIB_TYPE_UUID_RTLIB_EXT_OPX7_OPX7_ACCELERATION_STRUCTURE_BUILD_INPUT);
			public:
				static auto New(OPX7Context* context)->OPX7AccelerationStructureBuildInput*;
				virtual ~OPX7AccelerationStructureBuildInput()noexcept;
				virtual void Destroy()noexcept;
			private:
				OPX7AccelerationStructureBuildInput()noexcept;
				auto GetOptixBuildInput()const noexcept ->OptixBuildInput;
			private:
				struct Impl;
				std::unique_ptr<Impl> m_Impl;
			};
			class OPX7AccelerationStructureBuildInputTriangleArray : public OPX7AccelerationStructureBuildInput
			{
			private:
				RTLIB_CORE_TYPE_OBJECT_DECLARE_DERIVED_METHOD(OPX7AccelerationStructureBuildInputTriangleArray, OPX7AccelerationStructureBuildInput, RTLIB_TYPE_UUID_RTLIB_EXT_OPX7_OPX7_ACCELERATION_STRUCTURE_BUILD_INPUT_TRIANGLE_ARRAY);
			public:
				static auto New(OPX7Context* context)->OPX7AccelerationStructureBuildInputTriangleArray*;
				virtual ~OPX7AccelerationStructureBuildInputTriangleArray()noexcept;
				virtual void Destroy()noexcept;
			private:
				OPX7AccelerationStructureBuildInputTriangleArray()noexcept;
				auto GetOptixBuildInput()const noexcept ->OptixBuildInput;
			private:
				struct Impl;
				std::unique_ptr<Impl> m_Impl;
			};
			class OPX7AccelerationStructureBuildInputCurveArray : public OPX7AccelerationStructureBuildInput
			{
			private:
				RTLIB_CORE_TYPE_OBJECT_DECLARE_DERIVED_METHOD(OPX7AccelerationStructureBuildInputCurveArray, OPX7AccelerationStructureBuildInput, RTLIB_TYPE_UUID_RTLIB_EXT_OPX7_OPX7_ACCELERATION_STRUCTURE_BUILD_INPUT_CURVE_ARRAY);
			public:
				static auto New(OPX7Context* context)->OPX7AccelerationStructureBuildInputCurveArray*;
				virtual ~OPX7AccelerationStructureBuildInputCurveArray()noexcept;
				virtual void Destroy()noexcept;
			private:
				OPX7AccelerationStructureBuildInputCurveArray()noexcept;
				auto GetOptixBuildInput()const noexcept ->OptixBuildInput;
			private:
				struct Impl;
				std::unique_ptr<Impl> m_Impl;
			};
			class OPX7AccelerationStructureBuildInputCustomPrimitiveArray : public OPX7AccelerationStructureBuildInput
			{
			private:
				RTLIB_CORE_TYPE_OBJECT_DECLARE_DERIVED_METHOD(OPX7AccelerationStructureBuildInputCustomPrimitiveArray, OPX7AccelerationStructureBuildInput, RTLIB_TYPE_UUID_RTLIB_EXT_OPX7_OPX7_ACCELERATION_STRUCTURE_BUILD_INPUT_CUSTOM_PRIMITIVE_ARRAY);
			public:
				static auto New(OPX7Context* context)->OPX7AccelerationStructureBuildInputCustomPrimitiveArray*;
				virtual ~OPX7AccelerationStructureBuildInputCustomPrimitiveArray()noexcept;
				virtual void Destroy()noexcept;
			private:
				OPX7AccelerationStructureBuildInputCustomPrimitiveArray()noexcept;
				auto GetOptixBuildInput()const noexcept ->OptixBuildInput;
			private:
				struct Impl;
				std::unique_ptr<Impl> m_Impl;
			};
			class OPX7AccelerationStructureBuildInputInstanceArray : public OPX7AccelerationStructureBuildInput
			{
			private:
				RTLIB_CORE_TYPE_OBJECT_DECLARE_DERIVED_METHOD(OPX7AccelerationStructureBuildInputInstanceArray, OPX7AccelerationStructureBuildInput, RTLIB_TYPE_UUID_RTLIB_EXT_OPX7_OPX7_ACCELERATION_STRUCTURE_BUILD_INPUT_INSTANCE_ARRAY);
			public:
				static auto New(OPX7Context* context)->OPX7AccelerationStructureBuildInputCustomPrimitiveArray*;
				virtual ~OPX7AccelerationStructureBuildInputInstanceArray()noexcept;
				virtual void Destroy()noexcept;
			private:
				OPX7AccelerationStructureBuildInputInstanceArray()noexcept;
				auto GetOptixBuildInput()const noexcept ->OptixBuildInput;
			private:
				struct Impl;
				std::unique_ptr<Impl> m_Impl;
			};
		}
	}
}
#endif