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
#include <omp.h>

using namespace std;

const int nCities = 1002;           //��������  
const double SPEED = 0.99;        //�˻��ٶ�  
const int INITIAL_TEMP = 10000;    //��ʼ�¶�  
const int L = 1000;       //Markov ���ĳ���  

struct unit                             //һ����  
{
    float length;                      //���ۣ��ܳ���  
    int path[nCities];                  //·��  
    bool operator < (const struct unit& other) const
    {
        return length < other.length;
    }
    bool operator == (const struct unit& other) const
    {
        for (int i = 0; i < nCities; i++)
        {
            if (path[i] != other.path[i])
                return false;
        }
        return true;
    }
};
unit bestone = { INT_MAX, {0} };         //���Ž�  

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
    void CalCulate_length(unit& p);  //���㳤��  
    void print(unit& p);             //��ӡһ����  
    void getNewSolution(unit& p);    // �������л�ȥһ���½�  
    bool Accept(unit& bestone, unit& temp, double t);//�½���Metropolis ׼�����  
    bool is_in_tuba(unit& p);
};

//stl �� generate �ĸ�����������  
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
    const double t_min = 0.001; //�¶����ޣ����¶ȴﵽt_min ����ֹͣ����  

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
#pragma omp parallel for num_threads(4)
        for (i = 0; i < L; i++)
        {
#pragma omp critical
            {
                getNewSolution(temp);
                if (Accept(bestone, temp, t))
                {
                    memcpy(&bestone, &temp, sizeof(unit));
                }
                else
                {
                    memcpy(&temp, &bestone, sizeof(unit));
                }
            }
        }
        t *= r; //�˻� 
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
    if (bestone.length > temp.length) //��ø��̵�·��  
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
    {//����  
        int temp = p.path[i];
        p.path[i] = p.path[j];
        p.path[j] = temp;
    }
    else if (choose == 1)
    {//����  
        reverse(p.path + i, p.path + j);
    }
    else
    {//��λ  
        if (j + 1 == nCities) //�߽紦������  
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
    char str[10000];  //�ݴ��ȡ���ַ���
    int i = 1, j = 0, m = 0;  //i���ƴӳ��������ļ��ڼ��ж�ȡ��j����ֻ������ֵ���������б��,mΪ�������긳ֵ�±�
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
                length_table[i][j] = sqrt(pow(point_table[i][0] - point_table[j][0], 2) + pow(point_table[i][1] - point_table[j][1], 2));
            }
        }
    }
}

void saTSP::CalCulate_length(unit& p)
{
    int j = 0;
    p.length = 0;
#pragma omp simd
    for (j = 1; j < nCities; j++)
    {
        p.length += length_table[p.path[j - 1]][p.path[j]];
    }
    p.length += length_table[p.path[nCities - 1]][p.path[0]];
}

void saTSP::print(unit& p)
{
    int i;
    cout << "�����ǣ�" << p.length << endl;
    cout << "·���ǣ�";
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
    char a[20] = "pr1002";
    sa.init_dis(a);
    sa.SA_TSP();
    sa.CalCulate_length(bestone);
    sa.print(bestone);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "��ʱ:" << (tail - head) * 1000.0 / freq << "ms" << endl;
    return 0;
}