#include "agent_framework.h"
#include <iostream>

SubAgent::SubAgent(const std::string& name) : name(name) {}

BHV SubAgent::get_state() {
    return state;
}

Agent::Agent(const std::string& name) : name(name) {}

void Agent::add_sub_agent(SubAgent* sub_agent) {
    sub_agents.push_back(sub_agent);
}

void Agent::run_serial() {
    for (auto& sub_agent : sub_agents) {
        sub_agent->run();
    }
}

void Agent::run_parallel() {
    std::vector<std::thread> threads;
    for (auto& sub_agent : sub_agents) {
        threads.emplace_back(&SubAgent::run, sub_agent);
    }
    for (auto& thread : threads) {
        thread.join();
    }
}

AgentFramework::AgentFramework() {}

void AgentFramework::add_agent(Agent* agent) {
    agents.push_back(agent);
}

void AgentFramework::run() {
    for (auto& agent : agents) {
        agent->run_parallel();
    }
}
