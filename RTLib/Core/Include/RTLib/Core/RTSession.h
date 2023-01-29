#ifndef RTLIB_CORE_RT_SESSION_H
#define RTLIB_CORE_RT_SESSION_H
#include <string>
namespace RTLib
{
	namespace Core {
		class RTSession
		{
		public:
			RTSession() noexcept;
			virtual ~RTSession() noexcept;

			bool Init();
			void Exec();
			void Free();

			void    AddSession(RTSession* session) noexcept;
			void RemoveSession(RTSession* session) noexcept;

			void    AddActor(class RTActor* actor) noexcept;
			void RemoveActor(class RTActor* actor) noexcept;

			void    AddSubSystem(class RTSubSystem* subSystem) noexcept;
			void RemoveSubSystem(class RTSubSystem* subSystem) noexcept;

			template<typename RTSubSystemType, bool Cond = std::is_base_of_v<RTSubSystem,RTSubSystemType>>
			auto GetSubSystem(std::string name)noexcept -> RTSubSystemType*
			{
				return static_cast<RTSubSystemType*>(GetSubSystem(name, RTSubSystemType::ID));
			}

			auto GetName() const noexcept ->   std::string;
			void SetName(const std::string& name) noexcept;
		protected:
			virtual bool OnInit();
			virtual void OnExec();
			virtual void OnFree();
		private:;
			auto GetActor(std::string name, uint64_t actor_id)noexcept -> class RTActor*;
			auto GetSubSystem(std::string name, uint64_t subsystem_id)noexcept -> class RTSubSystem*;
		private:
			std::string m_Name;
		};
	}
}
#endif
