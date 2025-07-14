#include "overlay.h"
#include <imgui.h>

Overlay::Overlay() {}

void Overlay::show_window(const std::string& title, const std::string& content) {
    ImGui::Begin(title.c_str());
    ImGui::Text(content.c_str());
    ImGui::End();
}

void Overlay::show_notification(const std::string& text) {
    // This is a placeholder for showing a notification.
    // A real implementation would use a library like libnotify
    // or a platform-specific API.
    std::cout << "Notification: " << text << std::endl;
}

std::string Overlay::get_input() {
    // This is a placeholder for getting input.
    // A real implementation would capture keyboard, canvas, or voice input.
    return "user_input";
}

void Overlay::run_plugin(const std::string& plugin_name) {
    // This is a placeholder for running a plugin.
    // A real implementation would load and execute a plugin.
    std::cout << "Running plugin: " << plugin_name << std::endl;
}
