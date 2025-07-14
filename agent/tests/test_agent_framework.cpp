#include <gtest/gtest.h>
#include "agent_framework.h"
#include <iostream>

class TestSubAgent : public SubAgent {
public:
    TestSubAgent(const std::string& name) : SubAgent(name) {}
    void run() override {
        std::cout << "Running sub-agent: " << name << std::endl;
        state = BHV::encode(name);
    }
};

TEST(AgentFrameworkTest, Run) {
    AgentFramework framework;
    Agent agent("Test Agent");
    TestSubAgent sub_agent1("Sub-agent 1");
    TestSubAgent sub_agent2("Sub-agent 2");
    agent.add_sub_agent(&sub_agent1);
    agent.add_sub_agent(&sub_agent2);
    framework.add_agent(&agent);
    framework.run();
    ASSERT_NE(sub_agent1.get_state().data[0], 0);
    ASSERT_NE(sub_agent2.get_state().data[0], 0);
}
