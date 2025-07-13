#include "sgr.h"
#include <limits>

SGR::SGR() {
    // In a real system, these task vectors would be learned.
    task_vectors[TaskType::TEXT] = BHV::encode("text");
    task_vectors[TaskType::OCR] = BHV::encode("ocr");
    task_vectors[TaskType::CANVAS] = BHV::encode("canvas");
    task_vectors[TaskType::RETRIEVAL] = BHV::encode("retrieval");
}

TaskType SGR::route(const BHV& input) {
    int min_dist = std::numeric_limits<int>::max();
    TaskType best_task = TaskType::TEXT;

    for (auto const& [task_type, task_vector] : task_vectors) {
        int dist = BHV::hamming_distance(input, task_vector);
        if (dist < min_dist) {
            min_dist = dist;
            best_task = task_type;
        }
    }

    // Lazy load the module
    if (module_loaders.count(best_task)) {
        module_loaders[best_task]();
    }

    return best_task;
}

void SGR::register_module(TaskType task_type, std::function<void()> load_function) {
    module_loaders[task_type] = load_function;
}
