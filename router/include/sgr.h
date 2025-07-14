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

#include <json/json.h>
#include "paging_system.h"

class SGR {
public:
    SGR(const std::string& registry_path);
    TaskType route(const BHV& input);

private:
    std::map<TaskType, BHV> task_vectors;
    Json::Value module_registry;
    PagingSystem paging_system;
    void load_module(TaskType task_type);
};
