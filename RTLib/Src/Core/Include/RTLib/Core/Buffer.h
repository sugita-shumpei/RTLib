#ifndef RTLIB_CORE_BUFFER_H
#define RTLIB_CORE_BUFFER_H
#include <cstdint>
namespace RTLib{
    namespace Core{
        class IBuffer;
        enum class BufferMapAccess {
            eReadOnly,
            eWriteOnly,
            eReadWrite
        };
        //VIEW
        class IBufferMappableView {
        public:
            virtual ~IBufferMappableView()noexcept {}
            virtual auto Buffer()noexcept -> IBuffer* = 0;
            virtual bool Map  (BufferMapAccess access, void** pMappedData, size_t size, size_t offset = 0) = 0;
            virtual void Unmap() = 0;
        };
        class IBufferResizableView{
        public:
            virtual ~IBufferResizableView()noexcept {}
            virtual auto Buffer()noexcept -> IBuffer* = 0;
            virtual bool Resize(size_t newSizeInBytes) = 0;
        };
        class IBufferAddressableView {
        public:
            virtual ~IBufferAddressableView()noexcept {}
            virtual auto Buffer()noexcept -> IBuffer* = 0;
            virtual auto Address()  -> uintptr_t = 0;
        };
        //INTERNAL
        class IBufferInternalData {
        public:
            virtual ~IBufferInternalData()noexcept {}
        };
        //BUFFER FACTORY
        class IBuffer;
        class IBufferFactory {
        public:
            virtual auto NewBuffer(size_t sizeInBytes, const void* pInitialData = nullptr)const -> IBuffer* = 0;
            virtual ~IBufferFactory()noexcept{}
        };
        //BUFFER
        class IBuffer {
        public:
            using Factory = IBufferFactory;
            virtual ~IBuffer()noexcept{}
            virtual auto Size()  const noexcept  -> size_t = 0;
            virtual auto Stride()const noexcept  -> size_t = 0;
            //View
            virtual auto Addressable()noexcept -> IBufferAddressableView* { return nullptr;}
            virtual auto Mappable   ()noexcept -> IBufferMappableView *   { return nullptr;}
            virtual auto Resizable  ()noexcept -> IBufferResizableView*   { return nullptr;}
        protected:
            //INTERNAL DATA
            virtual auto InternalData ()noexcept -> IBufferInternalData*            { return nullptr;}
            virtual auto InternalData ()const noexcept ->const IBufferInternalData* { return nullptr;}
        };
    }
}
#endif
