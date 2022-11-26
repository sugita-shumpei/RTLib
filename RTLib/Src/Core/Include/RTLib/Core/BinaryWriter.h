#ifndef RTLIB_CORE_BINARY_WRITER_H
#define RTLIB_CORE_BINARY_WRITER_H
#include <string>
#include <vector>
namespace RTLib {
	namespace Core {
		bool SavePngImage(std::string path, int width, int height, const std::vector<unsigned char>& pixels);
		bool SaveExrImage(std::string path, int width, int height, const std::vector<float>& pixels);

	}
}
#endif
