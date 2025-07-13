#pragma once

#include "bhv.h"
#include <vector>
#include <map>

class AssociativeMemory {
public:
    AssociativeMemory();
    void insert(const BHV& key, const BHV& value);
    BHV query(const BHV& key);
    void update(const BHV& key, const BHV& value);

private:
    std::map<BHV, BHV> memory;
};
