#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <random>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <CL/sycl.hpp>

// 自定义一个表示城市的点类
class Point {
public:
    float x, y;
};

// 设定评估函数
float evaluate(std::vector<int>& sol, std::vector<Point>& cities) {
    float dist = 0.0;
    for (int i = 0; i < sol.size() - 1; ++i) {
        float dx = cities[sol[i + 1]].x - cities[sol[i]].x;
        float dy = cities[sol[i + 1]].y - cities[sol[i]].y;
        dist += std::sqrt(dx * dx + dy * dy);
    }
    // 添加从终点城市回到起点城市的距离
    float dx = cities[sol[0]].x - cities[sol[sol.size() - 1]].x;
    float dy = cities[sol[0]].y - cities[sol[sol.size() - 1]].y;
    dist += std::sqrt(dx * dx + dy * dy);
    return dist;
}

auto rng = std::default_random_engine{ std::random_device()() };

int main() {
    // 初始化DPC++队列
    sycl::queue q(sycl::default_selector{});

    auto start = std::chrono::high_resolution_clock::now();

    const int NumCities = 1002;
    const float TempInitial = 10000.0;
    const float TempFinal = 0.001;
    const int Iterations = 1000;

    std::vector<Point> cities(NumCities);
    std::vector<int> sol(NumCities);

    // 初始化城市坐标
    for (auto& city : cities) {
        city.x = ((float)rand()) / RAND_MAX;
        city.y = ((float)rand()) / RAND_MAX;
    }

    // 将sol初始化为城市的初始序列
    std::iota(sol.begin(), sol.end(), 0);

    // 初始解的评估距离
    float currentDist = evaluate(sol, cities);
    std::cout << "Initial solution distance: " << currentDist << "\n";

    // 开始退火过程
    float T = TempInitial;
    int iter = 0;

    while (T > TempFinal) {
        for (int k = 0; k < Iterations; ++k) {
            // 将解队列的两个随机位置城市交换，可得到新解
            std::uniform_int_distribution<int> uni(0, NumCities - 1);
            int i = uni(rng), j = uni(rng);
            std::swap(sol[i], sol[j]);

            // 计算新解的评估距离
            auto newDist = evaluate(sol, cities);

            // 如果新解更好或在一定概率下接受较差解
            auto delta = newDist - currentDist;
            if (delta < 0.0 || std::exp(-delta / T) >((float)rand()) / RAND_MAX) {
                currentDist = newDist;
            }
            else {
                // 如果新解不被接受，则撤销交换操作
                std::swap(sol[i], sol[j]);
            }
            iter++;
        }

        T *= 0.99;
    }

    std::cout << "Final solution distance: " << currentDist << "\n";
    std::cout << "Solution: ";
    for (auto& s : sol)
        std::cout << s << " ";
    std::cout << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Running time: " << duration << " ms" << std::endl;

    return 0;
}