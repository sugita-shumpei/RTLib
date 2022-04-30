#ifndef RTLIB_CORE_CPU_BUFFER_H
#define RTLIB_CORE_CPU_BUFFER_H
#include <RTLib/Core/Buffer.h>
#include <vector>
#include <memory>
namespace RTLib{
    namespace Core{
        //TCpuBuffer
        template<typename T>
        class TCpuBufferInternalData : public IBufferInternalData{
        public:
            virtual ~TCpuBufferInternalData()noexcept { }
            auto Size  ()const noexcept -> size_t { return sizeof(T) * m_Data.size(); }
            auto Stride()const noexcept -> size_t { return sizeof(T) ; }
            auto Data  ()const noexcept -> const std::vector<T>& {
                return m_Data;
            }
            auto Data  ()      noexcept ->       std::vector<T>& {
                return m_Data;
            }
        private:
            std::vector<T> m_Data;
        };
        template<typename T>
        class TCpuBuffer;
        template<typename T>
        class TCpuBufferAddressasbleView:public IBufferAddressableView {
        public:
            TCpuBufferAddressasbleView(TCpuBuffer<T>* buffer)noexcept : m_Buffer{ buffer } {}
            virtual ~TCpuBufferAddressasbleView()noexcept {}
            virtual auto Buffer()noexcept -> IBuffer* override { return m_Buffer; }
            virtual auto Address() -> uintptr_t override {
                if (!m_Buffer) { return 0; }
                auto internalData = static_cast<const TCpuBufferInternalData<T>*>(m_Buffer->InternalData());
                return reinterpret_cast<uintptr_t>(internalData->Data().data());
            }
        private:
            TCpuBuffer<T>* m_Buffer = nullptr;
        };
        template<typename T>
        class TCpuBufferMappableView : public IBufferMappableView{
        public:
            TCpuBufferMappableView(TCpuBuffer<T>* buffer)noexcept : m_Buffer{buffer}{}
            virtual ~TCpuBufferMappableView()noexcept { m_MappedMemory.reset(); }
            virtual auto Buffer()noexcept -> IBuffer* override { return m_Buffer; }
            virtual bool Map  (BufferMapAccess access, void** pMappedData, size_t size, size_t offset = 0) override {
                if (!pMappedData || !m_Buffer || m_MappedMemory){

                    return false;
                }
                if (offset % sizeof(T) != 0){

                    return false;
                }
                if (size % sizeof(T) != 0){

                    return false;
                }
                if (m_Buffer->Size() < offset + size ){

                    return false;
                }
                auto internalData = static_cast<TCpuBufferInternalData<T>*>(m_Buffer->InternalData());
                m_MappedMemory = std::unique_ptr<char[]>(new char[size]);
                m_Offset       = offset;
                m_Size         = size;
                m_Access       = access;
                *pMappedData   = m_MappedMemory.get();
                if (access == BufferMapAccess::eReadOnly || access == BufferMapAccess::eReadWrite) {
                    std::memcpy(*pMappedData, reinterpret_cast<const char*>(internalData->Data().data()) + offset, size);
                }
                return true;
            }
            virtual void Unmap() override {
                if (!m_Buffer||!m_MappedMemory){
                    return;
                }
                if (m_Access == BufferMapAccess::eWriteOnly || m_Access == BufferMapAccess::eReadWrite) {
                    auto internalData = static_cast<TCpuBufferInternalData<T>*>(m_Buffer->InternalData());
                    std::memcpy(reinterpret_cast<char*>(internalData->Data().data()) + m_Offset, m_MappedMemory.get(), m_Size);
                }
                m_MappedMemory.reset();
                m_Offset = 0;
                m_Size   = 0;
                m_Access = BufferMapAccess::eReadOnly;
            }
        private:
            TCpuBuffer<T>*          m_Buffer       = nullptr;
            std::unique_ptr<char[]> m_MappedMemory = {};
            size_t                  m_Offset       = 0;
            size_t                  m_Size         = 0;
            BufferMapAccess         m_Access       = BufferMapAccess::eReadOnly;
        };
        template<typename T>
        class TCpuBufferResizableView: public IBufferResizableView{
        public:
            TCpuBufferResizableView(TCpuBuffer<T>* buffer)noexcept : m_Buffer{buffer}{}
            virtual ~TCpuBufferResizableView()noexcept {}
            virtual auto Buffer()noexcept -> IBuffer* override { return m_Buffer; }
            virtual bool Resize(size_t newSizeInBytes) override {
                if (newSizeInBytes % sizeof(T)!=0){
                    return false;
                }
                if ( !m_Buffer ){
                    return false;
                }
                auto internalData = static_cast<TCpuBufferInternalData<T>*>(m_Buffer->InternalData());
                internalData->Data().resize(newSizeInBytes/sizeof(T));
                return true;
            }
        private:
            TCpuBuffer<T>*          m_Buffer       = nullptr;
        };
        template<typename T>
        class TCpuBufferFactory : public IBufferFactory {
        public:
            static auto New()noexcept -> TCpuBufferFactory* { return new TCpuBufferFactory<T>(); }
            virtual ~TCpuBufferFactory()noexcept {}
            // IBufferFactory ‚ð‰î‚µ‚ÄŒp³‚³‚ê‚Ü‚µ‚½
            virtual auto NewBuffer(size_t sizeInBytes, const void* pInitialData = nullptr) const -> IBuffer* override
            {
                if ((sizeInBytes % sizeof(T) != 0) || sizeInBytes == 0) {
                    return nullptr;
                }
                return new TCpuBuffer<T>(sizeInBytes, pInitialData);
            }
        private:
            TCpuBufferFactory()noexcept :IBufferFactory() {}
        };
        template<typename T>
        class TCpuBuffer : public IBuffer{
        public:
            using Factory = TCpuBufferFactory<T>;
            virtual ~TCpuBuffer()noexcept{
                m_InternalData.reset();
            }
            virtual auto Size()  const noexcept  -> size_t { return m_InternalData->Size  ();}
            virtual auto Stride()const noexcept  -> size_t { return m_InternalData->Stride();}
            virtual auto Addressable()noexcept -> IBufferAddressableView* override { return new TCpuBufferAddressasbleView<T>(this); }
            virtual auto Mappable   ()noexcept -> IBufferMappableView   * override { return new TCpuBufferMappableView <T>(this);}
            virtual auto Resizable  ()noexcept -> IBufferResizableView  * override { return new TCpuBufferResizableView<T>(this);}
        protected:
            friend class TCpuBufferFactory <T>;
            friend class TCpuBufferMappableView <T>;
            friend class TCpuBufferResizableView<T>;
            friend class TCpuBufferAddressasbleView<T>;
            TCpuBuffer(size_t sizeInBytes, const void* pInitialData)noexcept :m_InternalData{ new TCpuBufferInternalData<T>() } {
                m_InternalData->Data() = std::vector<T>(sizeInBytes/sizeof(T));
                if (pInitialData) {
                    std::memcpy(m_InternalData->Data().data(), pInitialData, sizeInBytes);
                }
            }
            TCpuBuffer(const std::vector<T>& data)noexcept :m_InternalData{new TCpuBufferInternalData<T> ()} {
                m_InternalData->Data() = data;
            }
            virtual auto InternalData ()noexcept -> IBufferInternalData* override { return m_InternalData.get();}
            virtual auto InternalData ()const noexcept ->const IBufferInternalData* { return  m_InternalData.get();}
        private:
            std::unique_ptr<TCpuBufferInternalData<T>> m_InternalData;
        };
    }
}
#endif
