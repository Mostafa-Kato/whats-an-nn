#ifndef WHATS_AN_NN_VALUE_H
#define WHATS_AN_NN_VALUE_H
#include <functional>
#include <memory>
#include <vector>
#include <string>

using std::shared_ptr;
using std::vector;
using std::function;

class Value : public std::enable_shared_from_this<Value>{
public:
    double data;
    double grad;
    function<void()> back;
    std::string op;
    vector<shared_ptr<Value>> children;

    Value(double data, vector<shared_ptr<Value>> children = {}, std::string op = "");

    void backward();
};

using ValuePtr = shared_ptr<Value>;
ValuePtr operator+(const ValuePtr& a, const ValuePtr& b);
ValuePtr operator*(const ValuePtr& a, const ValuePtr& b);
ValuePtr operator-(const ValuePtr& a, const ValuePtr& b);
ValuePtr operator/(const ValuePtr& a, const ValuePtr& b);
ValuePtr pow(const ValuePtr& a, double power);
ValuePtr relu(const ValuePtr& a);
ValuePtr exp(const ValuePtr& a);
ValuePtr log(const ValuePtr& a);

#endif