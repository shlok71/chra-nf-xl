#pragma once

#include "bhv.h"
#include <string>
#include <vector>
#include <map>
#include <functional>

enum class TaskType {
    TEXT,
    OCR,
    CANVAS,
    RETRIEVAL
};

class SGR {
public:
    SGR();
    TaskType route(const BHV& input);
    void register_module(TaskType task_type, std::function<void()> load_function);

private:
    std::map<TaskType, BHV> task_vectors;
    std::map<TaskType, std::function<void()>> module_loaders;
};
