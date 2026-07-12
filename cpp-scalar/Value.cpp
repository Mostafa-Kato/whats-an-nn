//
// Created by mkato on 6/30/2026.
//

#include <cmath>
#include <utility>
#include "Value.h"

#include <set>
#include <stack>

using std::shared_ptr;
using std::vector;
using std::function;
using std::set;

Value::Value(double data, vector<shared_ptr<Value>> children, std::string op)
    : data(data), grad(0), back([](){}), op(std::move(op)), children(std::move(children)) {}

ValuePtr operator+(const ValuePtr& a, const ValuePtr& b) {
    auto result = std::make_shared<Value>(a->data + b->data, std::vector<ValuePtr>{a, b}, "+");

    Value* raw_res = result.get();
    result->back = [a, b, raw_res]() {
        a->grad += raw_res->grad;
        b->grad += raw_res->grad;
    };
    return result;
}

ValuePtr operator*(const ValuePtr& a, const ValuePtr& b) {
    auto result = std::make_shared<Value>(a->data * b->data, std::vector<ValuePtr>{a, b}, "*");

    Value* raw_res = result.get();
    result->back = [a, b, raw_res]() {
        a->grad += b->data * raw_res->grad;
        b->grad += a->data * raw_res->grad;
    };
    return result;
}

ValuePtr operator-(const ValuePtr& a, const ValuePtr& b) {
    auto result = std::make_shared<Value>(a->data - b->data, std::vector<ValuePtr>{a, b}, "-");

    Value* raw_res = result.get();
    result->back = [a, b, raw_res]() {
        a->grad += raw_res->grad;
        b->grad -= raw_res->grad;
    };
    return result;
}

ValuePtr operator/(const ValuePtr& a, const ValuePtr& b) {
    return a * pow(b, -1.0);
}

ValuePtr exp(const ValuePtr& a) {
    auto result = std::make_shared<Value>(std::exp(a->data), std::vector<ValuePtr>{a}, "exp");

    Value* raw_res = result.get();
    result->back = [a, raw_res]() {
        a->grad += raw_res->data * raw_res->grad;
    };
    return result;
}

ValuePtr pow(const ValuePtr& a, double power) {
    auto result = std::make_shared<Value>(std::pow(a->data, power), std::vector<ValuePtr>{a}, "**");

    Value* raw_res = result.get();
    result->back = [a, power, raw_res]() {
        a->grad += power * std::pow(a->data, power - 1) * raw_res->grad;
    };
    return result;
}

ValuePtr log(const ValuePtr& a) {
    auto result = std::make_shared<Value>(std::log(a->data), std::vector<ValuePtr>{a}, "log");

    Value* raw_res = result.get();
    result->back = [a, raw_res]() {
        a->grad += (1.0 / a->data) * raw_res->grad;
    };
    return result;
}

ValuePtr relu(const ValuePtr& a) {
    auto result = std::make_shared<Value>(std::max(0.0, a->data), std::vector<ValuePtr>{a}, "relu");

    Value* raw_res = result.get();
    result->back = [a, raw_res]() {
        if (raw_res->data > 0) {
            a->grad += raw_res->grad;
        }
    };
    return result;
}

void Value::backward() {
    set<ValuePtr> visited;
    vector<ValuePtr> order;

    std::stack<std::pair<ValuePtr, bool>> to_visit;
    to_visit.push({shared_from_this(),false});

    while (!to_visit.empty()) {
        auto [n, children_processed] = to_visit.top();
        to_visit.pop();

        if (visited.count(n)) continue;

        if (children_processed) {
            visited.insert(n);
            order.push_back(n);
        } else {
            to_visit.push({n,true});
            for (auto& child : n->children) {
                if (!visited.count(child)) {
                    to_visit.push({child,false});
                }
            }
        }
    }

    this->grad = 1;

    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        (*it)->back();
    }

    for (auto& node : order) {
        node->children.clear();
        node->back = [](){};
    }
}
