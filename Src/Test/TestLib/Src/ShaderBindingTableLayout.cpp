#include <TestLib/ShaderBindingTableLayout.h>

TestLib::ShaderBindingTableLayout::ShaderBindingTableLayout(ShaderBindingTableLayoutDesc desc) noexcept
	:m_SbtStride{ desc.sbtStride }
{
	// 効率的な解決法: 一旦 InstanceASを全て数えてしまう
	// INSTANCE AS --> INSTANCE --> GEOMETRY AS
	//             --> INSTANCE --> INSTANCE AS
	desc.normalize();

	m_GeometryAccelerationStructures.reserve(desc.geometryAccelerationStructures.size());
	for (auto& geometryAS : desc.geometryAccelerationStructures)
	{
		m_GeometryAccelerationStructures.emplace_back(this, geometryAS);
	}

	m_InstanceAccelerationSturctureOrArrays.reserve(desc.instanceAccelerationStructureOrArrays.size());
	for (auto& instanceASOrArray : desc.instanceAccelerationStructureOrArrays)
	{
		m_InstanceAccelerationSturctureOrArrays.emplace_back(this, instanceASOrArray);
	}


	{
		std::unordered_set<unsigned int> finishSet;
		std::queue<unsigned int>   waitQueue;
		for (auto i = 0; i < m_InstanceAccelerationSturctureOrArrays.size(); ++i)
		{
			waitQueue.push(i);
		}

		while (!waitQueue.empty())
		{
			auto instanceASIndex = waitQueue.front();
			waitQueue.pop();
			bool instanceASOfFinishInstance = true;
			auto& instanceAS = m_InstanceAccelerationSturctureOrArrays[instanceASIndex];
			for (auto& instance : instanceAS.instances)
			{
				if (instance.type == AccelerationStructureType::eInstance) {
					if (finishSet.count(instance.baseIndex) == 0) {
						instanceASOfFinishInstance = false;
						break;
					}
				}
			}
			if (!instanceASOfFinishInstance) {
				waitQueue.push(instanceASIndex);
				continue;
			}
			unsigned int sbtOffset = 0;
			for (auto& instance : instanceAS.instances)
			{
				instance.sbtOffsetInternalParent = sbtOffset;
				sbtOffset += instance.get_sbt_count();
			}
			instanceAS.sbtCount = sbtOffset;
			finishSet.insert(instanceASIndex);
		}

		std::stack<uint32_t > stack;
		for (auto i = 0; i < m_InstanceAccelerationSturctureOrArrays[0].instances.size(); ++i)
		{
			auto& instance = m_InstanceAccelerationSturctureOrArrays[0].instances[i];
			if (instance.type == AccelerationStructureType::eInstance) {
				stack.push(0);

			}
		}

		if (stack.empty()) {
			return;
		}

		while (!stack.empty())
		{
			auto instanceASIndex = stack.top();
			stack.pop();

			auto& instanceAS = m_InstanceAccelerationSturctureOrArrays[instanceASIndex];
			for (auto& instance : instanceAS.instances)
			{
				instance.sbtOffsetExternalParent = instanceAS.sbtOffset;
				if (instance.type == AccelerationStructureType::eInstance)
				{
					auto& nextInstanceAS = m_InstanceAccelerationSturctureOrArrays[instance.baseIndex];
					nextInstanceAS.sbtOffset = instance.get_sbt_offset();
					stack.push(instance.baseIndex);
				}
			}

		}

		// INSTANCE_AS(5,0,0)--> INSTANCE0(3,0,0)--> GEOMETRY_AS(3)
		//                       INSTANCE1(2,3,0)--> GEOMETRY_AS(2)
		// INSTANCE_AS(7,0,0)--> INSTANCE0(2,0,0)--> GEOMETRY_AS(2)
		//                       INSTANCE1(5,2,0)--> INSTANCE_AS(5,0,2)
		// INSTANCE_AS(7,0,1)--> INSTANCE0(2,0,1)--> GEOMETRY_AS(2)
		//                       INSTANCE1(5,2,1)--> INSTANCE_AS(5,0,3)--> INSTANCE0(3,0,3)--> GEOMETRY_AS(3)
		//                                                                 INSTANCE1(2,3,3)--> GEOMETRY_AS(3)

	}

	// 0 --> 0 --> 0 --> 0
	//   --> 1 --> 0 --> 0
}