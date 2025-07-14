#include "paging_system.h"
#include <fstream>
#include <iostream>

PagingSystem::PagingSystem() {}

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

void PagingSystem::page_out(const std::string& module_name, const std::vector<uint8_t>& data) {
    std::vector<uint8_t> compressed_data = compressor.compress(data);
    std::string page_path = get_page_path(module_name);
#ifdef _WIN32
    HANDLE file_handle = CreateFile(page_path.c_str(), GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    DWORD bytes_written;
    WriteFile(file_handle, compressed_data.data(), compressed_data.size(), &bytes_written, NULL);
    CloseHandle(file_handle);
#else
    int fd = open(page_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
    write(fd, compressed_data.data(), compressed_data.size());
    close(fd);
#endif
}

std::vector<uint8_t> PagingSystem::page_in(const std::string& module_name) {
    std::string page_path = get_page_path(module_name);
#ifdef _WIN32
    HANDLE file_handle = CreateFile(page_path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file_handle == INVALID_HANDLE_VALUE) {
        return {};
    }
    DWORD file_size = GetFileSize(file_handle, NULL);
    std::vector<uint8_t> compressed_data(file_size);
    DWORD bytes_read;
    ReadFile(file_handle, compressed_data.data(), file_size, &bytes_read, NULL);
    CloseHandle(file_handle);
#else
    int fd = open(page_path.c_str(), O_RDONLY);
    if (fd == -1) {
        return {};
    }
    struct stat st;
    fstat(fd, &st);
    std::vector<uint8_t> compressed_data(st.st_size);
    read(fd, compressed_data.data(), st.st_size);
    close(fd);
#endif
    return compressor.decompress(compressed_data);
}

std::string PagingSystem::get_page_path(const std::string& module_name) {
    return "/tmp/" + module_name + ".page";
}
