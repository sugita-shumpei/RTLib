#ifndef RTLIB_CORE_BINARY_WRITER_H
#define RTLIB_CORE_BINARY_WRITER_H
#include <string>
#include <vector>
namespace RTLib {
	namespace Core {
		bool SavePngImage(std::string path, int width, int height, const std::vector<unsigned char>& pixels);
		bool SaveBmpImage(std::string path, int width, int height, const std::vector<unsigned char>& pixels);
		bool SaveTgaImage(std::string path, int width, int height, const std::vector<unsigned char>& pixels);
		bool SaveJpgImage(std::string path, int width, int height, int quality, const std::vector<unsigned char>& pixels);
		bool SaveExrImage(std::string path, int width, int height, const std::vector<float>&         pixels);
		bool SaveHdrImage(std::string path, int width, int height, const std::vector<float>&         pixels);
	}
}
#endif
