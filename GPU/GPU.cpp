#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <random>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <CL/sycl.hpp>

// �Զ���һ����ʾ���еĵ���
class Point {
public:
    float x, y;
};

// �趨��������
float evaluate(std::vector<int>& sol, std::vector<Point>& cities) {
    float dist = 0.0;
    for (int i = 0; i < sol.size() - 1; ++i) {
        float dx = cities[sol[i + 1]].x - cities[sol[i]].x;
        float dy = cities[sol[i + 1]].y - cities[sol[i]].y;
        dist += std::sqrt(dx * dx + dy * dy);
    }
    // ��Ӵ��յ���лص������еľ���
    float dx = cities[sol[0]].x - cities[sol[sol.size() - 1]].x;
    float dy = cities[sol[0]].y - cities[sol[sol.size() - 1]].y;
    dist += std::sqrt(dx * dx + dy * dy);
    return dist;
}

auto rng = std::default_random_engine{ std::random_device()() };

int main() {
    // ��ʼ��DPC++����
    sycl::queue q(sycl::default_selector{});

    auto start = std::chrono::high_resolution_clock::now();

    const int NumCities = 1002;
    const float TempInitial = 10000.0;
    const float TempFinal = 0.001;
    const int Iterations = 1000;

    std::vector<Point> cities(NumCities);
    std::vector<int> sol(NumCities);

    // ��ʼ����������
    for (auto& city : cities) {
        city.x = ((float)rand()) / RAND_MAX;
        city.y = ((float)rand()) / RAND_MAX;
    }

    // ��sol��ʼ��Ϊ���еĳ�ʼ����
    std::iota(sol.begin(), sol.end(), 0);

    // ��ʼ�����������
    float currentDist = evaluate(sol, cities);
    std::cout << "Initial solution distance: " << currentDist << "\n";

    // ��ʼ�˻����
    float T = TempInitial;
    int iter = 0;

    while (T > TempFinal) {
        for (int k = 0; k < Iterations; ++k) {
            // ������е��������λ�ó��н������ɵõ��½�
            std::uniform_int_distribution<int> uni(0, NumCities - 1);
            int i = uni(rng), j = uni(rng);
            std::swap(sol[i], sol[j]);

            // �����½����������
            auto newDist = evaluate(sol, cities);

            // ����½���û���һ�������½��ܽϲ��
            auto delta = newDist - currentDist;
            if (delta < 0.0 || std::exp(-delta / T) >((float)rand()) / RAND_MAX) {
                currentDist = newDist;
            }
            else {
                // ����½ⲻ�����ܣ�������������
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