#include <iostream>
#include <string>
#include "sgr.h"
#include "bhv.h"
#include "reasoning_engine.h"
#include "nca_canvas.h"
#include "associative_memory.h"
#include "voice_input.h"
#include "voice_output.h"
#include "overlay.h"

#include "profiler.h"

int main(int argc, char* argv[]) {
    Profiler profiler;
    profiler.start_profiling();

    if (argc != 3) {
        std::cout << "Usage: neuroforge.exe <task> <input>" << std::endl;
        return 1;
    }

    std::string task = argv[1];
    std::string input = argv[2];

    SGR sgr("module_registry.json");
    BHV input_bhv = BHV::encode(input);
    TaskType task_type = sgr.route(input_bhv);

    std::cout << "Task type: " << static_cast<int>(task_type) << std::endl;

    return 0;
}
