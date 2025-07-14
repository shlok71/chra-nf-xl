#pragma once

#include "bhv.h"
#include <vector>
#include <string>
#include <map>
#include <functional>
#include <thread>

class SubAgent {
public:
    SubAgent(const std::string& name);
    virtual void run() = 0;
    BHV get_state();

protected:
    std::string name;
    BHV state;
};

class Agent {
public:
    Agent(const std::string& name);
    void add_sub_agent(SubAgent* sub_agent);
    void run_serial();
    void run_parallel();

private:
    std::string name;
    std::vector<SubAgent*> sub_agents;
};

class AgentFramework {
public:
    AgentFramework();
    void add_agent(Agent* agent);
    void run();

private:
    std::vector<Agent*> agents;
};
