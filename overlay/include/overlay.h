#pragma once

#include <string>

class Overlay {
public:
    Overlay();
    void show_window(const std::string& title, const std::string& content);
    void show_notification(const std::string& text);
    std::string get_input();
    void run_plugin(const std::string& plugin_name);
};
