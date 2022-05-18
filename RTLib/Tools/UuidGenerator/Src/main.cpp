#include <uuid.h>
#include <iostream>
#include <random>
#include <string>
#include <fstream>
int main(int argc, const char* argv[])
{
	int count = 1;
	if (argc > 2) {
		if (std::string(argv[1]) == "-c") {
			count = std::stoi(std::string(argv[2]));
		}
	}
	bool saveFile = false;
	auto fname = std::string();
	if (argc > 4) {
		if (std::string(argv[3]) == "-o") {
			fname = std::string(argv[4]);
			saveFile = true;
		}
	}
	std::random_device rd;
	auto seed_data = std::array<int, std::mt19937::state_size> {};
	std::generate(std::begin(seed_data), std::end(seed_data), std::ref(rd));
	std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
	std::mt19937 generator(seq);
	uuids::uuid_random_generator gen{ generator };
    std::cout << "OK!\n";
	if (!saveFile) {
		for (int j = 0; j < count; ++j) {
			uuids::uuid const id = gen();
			auto id_values = id.as_bytes();
			for (auto i = 0; i < 15; ++i) {
				std::cout << std::hex << "0x" << (uint32_t)(id_values[i]) << "-";
			}
			std::cout << std::hex << "0x" << (uint32_t)(id_values[15]) << ",\n";
		}
	}
	else {
		std::ofstream file(fname);
		if (file.is_open()) {
			std::cout << fname << std::endl;
			for (int j = 0; j < count; ++j) {
				uuids::uuid const id = gen();
				auto id_values = id.as_bytes();
				for (auto i = 0; i < 15; ++i) {
					file << std::hex << "0x" << (uint32_t)(id_values[i]) << "-";
				}
				file << std::hex << "0x" << (uint32_t)(id_values[15]) << ",\n";
			}
		}
		file.close();
	}
	return 0;
}