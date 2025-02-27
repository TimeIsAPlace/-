#include <iostream>  
#include <sstream>  
#include <fstream>  
#include <string>
#include <cstring>
#include <iterator>  
#include <algorithm>  
#include <climits>  
#include <cmath>  
#include <cstdlib>  
#include <vector>
#include <stdio.h>
#include <time.h>
#include <nmmintrin.h>
#include <immintrin.h>
#include <windows.h>

using namespace std;

const int nCities = 3038;           //城市数量  
const double SPEED = 0.99;        //退火速度  
const int INITIAL_TEMP = 10000;    //初始温度  
const int L = 1000;       //Markov 链的长度  

struct unit                             //一个解  
{
    float length;                      //代价，总长度  
    int path[nCities];                  //路径  
    bool operator < (const struct unit& other) const
    {
        return length < other.length;
    }
    bool operator == (const struct unit& other) const
    {
        int i = 0;
        __m128 v1, v2, ones;
        float temp[4];
        for (i = 0; i <= nCities - 4; i += 4)
        {
            v1 = _mm_loadu_ps(&path[i]);
            v2 = _mm_loadu_ps(&other.path[i]);
            ones = _mm_xor_ps(v1, v2);
            _mm_storeu_ps(&temp[0], ones);
            if (temp[0] == 1 || temp[1] == 1 || temp[2] == 1 || temp[3] == 1)
                return false;
        }
        while (i < nCities)
        {
            if (path[i] != other.path[i])
                return false;
            i++;
        }
        return true;
    }
};
unit bestone = { INT_MAX, {0} };         //最优解  

float point_table[nCities][nCities];  //distance  
float length_table[nCities][nCities];  //distance  
vector <unit> tabu_table;
const int tabu_size = 10;

class saTSP
{
public:
    void init_dis(char a[]);                  // create matrix to storage the Distance each city
    void SA_TSP();
    bool tabu_Search(unit& p);
    void CalCulate_length(unit& p);  //计算长度  
    void print(unit& p);             //打印一个解  
    void getNewSolution(unit& p);    // 从邻域中获去一个新解  
    bool Accept(unit& bestone, unit& temp, double t);//新解以Metropolis 准则接受  
    bool is_in_tuba(unit& p);
};

//stl 中 generate 的辅助函数对象  
class GenbyOne {
public:
    GenbyOne(int _seed = -1) : seed(_seed) {}
    int operator() () { return seed += 1; }
private:
    int seed;
};

void saTSP::SA_TSP()
{
    srand((int)time(0));
    int i = 0;
    double r = SPEED;
    double t = INITIAL_TEMP;
    const double t_min = 0.1; //温度下限，若温度达到t_min ，则停止搜索  

    //choose an initial solution ~  
    unit temp;
    generate(temp.path, temp.path + nCities, GenbyOne(0));
    random_shuffle(temp.path, temp.path + nCities);
    CalCulate_length(temp);
    memcpy(&bestone, &temp, sizeof(temp));

    // while the stop criterion is not yet satisfied do  
    while (t > t_min)
    {
        if (tabu_Search(temp))
            continue;
        for (i = 0; i < L; i++)
        {
            getNewSolution(temp);
            if (Accept(bestone, temp, t))
            {
                memcpy(&bestone, &temp, sizeof(unit));
                tabu_table.push_back(bestone);
                if (tabu_table.size() > tabu_size)
                    tabu_table.erase(tabu_table.begin());
            }
            else
            {
                memcpy(&temp, &bestone, sizeof(unit));
            }
        }
        t *= r; //退火 
    }
    return;
}

bool saTSP::tabu_Search(unit& p)
{
    unit temp;
    unit best_neighbour = { INT_MAX, {0} };
    int i;
    memcpy(&temp, &p, sizeof(p));
    for (i = 0; i < L; i++)
    {
        getNewSolution(temp);
        if (temp.length < best_neighbour.length)
        {
            memcpy(&best_neighbour, &temp, sizeof(unit));
        }
        else
        {
            memcpy(&temp, &best_neighbour, sizeof(unit));
        }
    }
    if (best_neighbour.length < bestone.length && !(is_in_tuba(best_neighbour)))
    {
        tabu_table.push_back(best_neighbour);
        memcpy(&bestone, &best_neighbour, sizeof(unit));
        if (tabu_table.size() > tabu_size)
            tabu_table.erase(tabu_table.begin());
        return true;
    }
    return false;
}

bool saTSP::is_in_tuba(unit& p)
{
    auto it = tabu_table.begin();
    for (; it != tabu_table.end(); it++)
    {
        if (*it == p)
            return true;
    }
    return false;
}

bool saTSP::Accept(unit& bestone, unit& temp, double t)
{
    if (is_in_tuba(temp))
        return false;
    if (bestone.length > temp.length) //获得更短的路径  
    {
        return true;
    }
    else
    {
        if ((int)(exp((bestone.length - temp.length) / t) * 100) > (rand() % 101))
        {
            return true;
        }
    }
    return false;
}

void saTSP::getNewSolution(unit& p)
{
    int i = rand() % nCities;
    int j = rand() % nCities;
    if (i > j)
    {
        int t = i;
        i = j;
        j = t;
    }
    else if (i == j)
    {
        return;
    }

    int choose = rand() % 3;
    if (choose == 0)
    {//交换  
        int temp = p.path[i];
        p.path[i] = p.path[j];
        p.path[j] = temp;
    }
    else if (choose == 1)
    {//置逆  
        reverse(p.path + i, p.path + j);
    }
    else
    {//移位  
        if (j + 1 == nCities) //边界处不处理  
        {
            return;
        }
        rotate(p.path + i, p.path + j, p.path + j + 1);
    }
    CalCulate_length(p);
}

void saTSP::init_dis(char a[]) // create matrix to storage the Distance each city  
{
    a = strcat(a, ".tsp");
    FILE* fp;
    char str[10000];  //暂存读取的字符串
    int i = 1, j = 0, m = 0;  //i控制从城市坐标文件第几行读取，j控制只读坐标值，不读城市编号,m为城市坐标赋值下标
    fp = fopen(a, "r");
    while (i < 7)
    {
        fgets(str, 255, fp);
        i++;
    }
    while (!feof(fp))
    {
        fscanf(fp, "%s\n", &str);
        if (j % 3 == 1) {
            point_table[m][0] = atoi(str);
        }
        else if (j % 3 == 2) {
            point_table[m][1] = atoi(str);
            m++;
        }
        j++;
    }
    fclose(fp);
    for (i = 0; i < nCities; i++)
    {
        for (j = 0; j < nCities; j++)
        {
            if (i == j)
                length_table[i][j] = 0;
            else
            {
                length_table[i][j] = float(sqrt(pow(point_table[i][0] - point_table[j][0], 2) + pow(point_table[i][1] - point_table[j][1], 2)));
            }
        }
    }
}

void saTSP::CalCulate_length(unit& p)
{
    int j = 0;
    p.length = 0;
    float temp[4] = { 0 };
    __m128 v1, v2;
    for (j = 1; j <= nCities - 4; j += 4)
    {
        v1 = _mm_set_ps(length_table[p.path[j - 1]][p.path[j]], length_table[p.path[j]][p.path[j + 1]], length_table[p.path[j + 1]][p.path[j + 2]], length_table[p.path[j + 2]][p.path[j + 3]]);
        v2 = _mm_loadu_ps(&temp[0]);
        v2 = _mm_add_ps(v1, v2);
        _mm_storeu_ps(&temp[0], v2);
    }
    p.length += (temp[0] + temp[1] + temp[2] + temp[3]);
    while (j < nCities)
    {
        p.length += length_table[p.path[j - 1]][p.path[j]];
        j++;
    }
    p.length += length_table[p.path[nCities - 1]][p.path[0]];
}

void saTSP::print(unit& p)
{
    int i;
    cout << "代价是：" << p.length << endl;
    cout << "路径是：";
    for (i = 0; i < nCities; i++)
    {
        cout << p.path[i] << " ";
    }
    cout << endl;
}

int main()
{
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    saTSP sa;
    char a[20] = "pcb3038";
    sa.init_dis(a);
    sa.SA_TSP();
    sa.CalCulate_length(bestone);
    sa.print(bestone);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "用时:" << (tail - head) * 1000.0 / freq << "ms" << endl;
    return 0;
}