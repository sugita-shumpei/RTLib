#define STB_IMAGE_WRITE_IMPLEMENTATION
#define TINYEXR_IMPLEMENTATION
#include <RTLib/Core/BinaryWriter.h>
#include <tinyexr.h>
#include <stb_image_write.h>
bool RTLib::Core::SavePngImage(std::string path, int width, int height, const std::vector<unsigned char>& pixels)
{
    if (width * height * 4 != pixels.size()) { return false; }
    return stbi_write_png(path.c_str(), width, height, 4, pixels.data(), width * 4);
}

bool RTLib::Core::SaveBmpImage(std::string path, int width, int height, const std::vector<unsigned char>& pixels)
{
    if (width * height * 4 != pixels.size()) { return false; }
    return stbi_write_bmp(path.c_str(), width, height, 4, pixels.data());
}

bool RTLib::Core::SaveTgaImage(std::string path, int width, int height, const std::vector<unsigned char>& pixels)
{
    if (width * height * 4 != pixels.size()) { return false; }
    return stbi_write_tga(path.c_str(), width, height, 4, pixels.data());
}

bool RTLib::Core::SaveJpgImage(std::string path, int width, int height, int quality, const std::vector<unsigned char>& pixels)
{
    if (width * height * 4 != pixels.size()) { return false; }
    return stbi_write_jpg(path.c_str(), width, height, 4, pixels.data(), quality);
}

bool RTLib::Core::SaveExrImage(std::string path, int width, int height, const std::vector<float>& pixels)
{
    EXRHeader header;
    InitEXRHeader(&header);

    EXRImage image;
    InitEXRImage(&image);

    image.num_channels = 3;

    std::vector<float> images[3];
    images[0].resize(width * height);
    images[1].resize(width * height);
    images[2].resize(width * height);

    // Split RGBRGBRGB... into R, G and B layer
    for (int i = 0; i < width * height; i++) {
        images[0][i] = pixels[3 * i + 0];
        images[1][i] = pixels[3 * i + 1];
        images[2][i] = pixels[3 * i + 2];
    }

    float* image_ptr[3];
    image_ptr[0] = &(images[2].at(0)); // B
    image_ptr[1] = &(images[1].at(0)); // G
    image_ptr[2] = &(images[0].at(0)); // R

    image.images = (unsigned char**)image_ptr;
    image.width = width;
    image.height = height;

    header.num_channels = 3;
    header.channels = (EXRChannelInfo*)malloc(sizeof(EXRChannelInfo) * header.num_channels);
    // Must be (A)BGR order, since most of EXR viewers expect this channel order.
    strncpy(header.channels[0].name, "B", 255); header.channels[0].name[strlen("B")] = '\0';
    strncpy(header.channels[1].name, "G", 255); header.channels[1].name[strlen("G")] = '\0';
    strncpy(header.channels[2].name, "R", 255); header.channels[2].name[strlen("R")] = '\0';

    header.pixel_types = (int*)malloc(sizeof(int) * header.num_channels);
    header.requested_pixel_types = (int*)malloc(sizeof(int) * header.num_channels);
    for (int i = 0; i < header.num_channels; i++) {
        header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT; // pixel type of input image
        header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_HALF; // pixel type of output image to be stored in .EXR
    }

    const char* err = NULL; // or nullptr in C++11 or later.
    int ret = SaveEXRImageToFile(&image, &header, path.c_str(), &err);
    if (ret != TINYEXR_SUCCESS) {
        fprintf(stderr, "Save EXR err: %s\n", err);
        FreeEXRErrorMessage(err); // free's buffer for an error message
        return ret;
    }
    //printf("Saved exr file. [ %s ] \n", path.c_str());

    free(header.channels);
    free(header.pixel_types);
    free(header.requested_pixel_types);
    return true;
}

bool RTLib::Core::SaveHdrImage(std::string path, int width, int height, const std::vector<float>& pixels)
{
    return stbi_write_hdr(path.c_str(),width,height, pixels.size()/(width*height), pixels.data());
}
