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
#include <stdio.h>
#include <time.h>
#include <nmmintrin.h>
#include <immintrin.h>
#include <windows.h>

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
};
unit bestone = { INT_MAX, {0} };         //���Ž�  

float point_table[nCities][nCities];  //distance  
float length_table[nCities][nCities];  //distance  

class saTSP
{
public:
    void init_dis(char a[]);                  // create matrix to storage the Distance each city
    void SA_TSP();
    void CalCulate_length(unit& p);  //���㳤��  
    void print(unit& p);             //��ӡһ����  
    void getNewSolution(unit& p);    // �������л�ȥһ���½�  
    bool Accept(unit& bestone, unit& temp, double t);//�½���Metropolis ׼�����  
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
        for (i = 0; i < L; i++)
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
        t *= r; //�˻�  
    }
    return;
}

bool saTSP::Accept(unit& bestone, unit& temp, double t)
{
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
    float temp[8] = { 0 };
    __m256 v1, v2;
    for (j = 1; j <= nCities - 8; j += 8)
    {
        v1 = _mm256_set_ps(length_table[p.path[j - 1]][p.path[j]], length_table[p.path[j]][p.path[j + 1]], length_table[p.path[j + 1]][p.path[j + 2]], length_table[p.path[j + 2]][p.path[j + 3]],
            length_table[p.path[j + 3]][p.path[j + 4]], length_table[p.path[j + 4]][p.path[j + 5]], length_table[p.path[j + 5]][p.path[j + 6]], length_table[p.path[j + 6]][p.path[j + 7]]);
        v2 = _mm256_loadu_ps(&temp[0]);
        v2 = _mm256_add_ps(v1, v2);
        _mm256_storeu_ps(&temp[0], v2);
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