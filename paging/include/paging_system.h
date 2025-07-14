#pragma once

#include <string>
#include <vector>

#include "compressor.h"

class PagingSystem {
public:
    PagingSystem();
    void page_out(const std::string& module_name, const std::vector<uint8_t>& data);
    std::vector<uint8_t> page_in(const std::string& module_name);

private:
    Compressor compressor;
    std::string get_page_path(const std::string& module_name);
};
