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

using namespace std;

const int nCities = 48;           //��������  
const double SPEED = 0.98;        //�˻��ٶ�  
const int INITIAL_TEMP = 1000;    //��ʼ�¶�  
const int L = 5 * nCities;       //Markov ���ĳ���  
const int sizepop = 1000;


int find(int gene[], int start, int end, int x)// ��gene�����в���x�����λ�ò�����
{
    for (int i = start; i <= end; i++) {
        if (gene[i] == x) {
            return i;
        }
    }
    return -1;
}

struct unit                             //һ����  
{
    double length;                      //���ۣ��ܳ���  
    int path[nCities];                  //·��  
    bool operator < (const struct unit& other) const
    {
        return length < other.length;
    }
    void exchange(unit& p)
    {
        unit other;
        memcpy(&other, &p, sizeof(unit));
        int i, j = 0, k = 0, repeat;
        int pos = rand() % (nCities - 4);	// ���ѡ�񽻲�λ�ã�����Ҫ����3��4���������Խ���λ��ֻ����[1��n-4]�ڣ�
        int num = 3 + rand() % 2;	// ���ѡ�񽻲�Ļ���������СΪ3�����Ϊ4

        int* segment1 = new int[nCities];	// ���ڼ�¼������ǰȾɫ���ϵĻ���
        int* segment2 = new int[nCities];	// ���ڼ�¼��������һȾɫ���ϵĻ���
        for (i = 0; i < nCities; i++) {
            if (i >= pos && i < pos + num) {
                segment1[i] = other.path[i];
                segment2[i] = path[i];
            }
            else {
                segment1[i] = other.path[i];
                segment2[i] = path[i];
            }
        }

        int* mapping1 = new int[4];	// ��ǰȾɫ���м�ε�ӳ��
        int* mapping2 = new int[4];	// ��һȾɫ���м�ε�ӳ��
        for (i = 0; i < 4; i++) {
            mapping1[i] = -1;	// ��ֵȫ��Ϊ-1
            mapping2[i] = -1;
        }
        for (i = pos; i < pos + num; i++) {
            repeat = find(segment1, pos, pos + num - 1, path[i]);
            if (repeat == -1) {
                mapping1[j++] = path[i];
            }
            repeat = find(segment2, pos, pos + num - 1, other.path[i]);
            if (repeat == -1) {
                mapping2[k++] = other.path[i];
            }
        }
        j = k = 0;
        for (i = pos; i < pos + num; i++) {// ���ظ��Ļ����滻Ϊӳ���еĻ���
            repeat = find(path, 0, pos - 1, segment1[i]);
            if (repeat != -1) {
                segment1[repeat] = mapping1[j++];
            }
            repeat = find(path, pos + num, nCities - 1, segment1[i]);
            if (repeat != -1) {
                segment1[repeat] = mapping1[j++];
            }
            repeat = find(other.path, 0, pos - 1, segment2[i]);
            if (repeat != -1) {
                segment2[repeat] = mapping2[k++];
            }
            repeat = find(other.path, pos + num, nCities - 1, segment2[i]);
            if (repeat != -1) {
                segment2[repeat] = mapping2[k++];
            }
        }
        for (i = 0; i < nCities; i++) {
            path[i] = segment1[i];	// �����ĸ�Ⱦɫ��
            other.path[i] = segment2[i];// ��������һȾɫ��
        }
        delete[]segment1;
        delete[]segment2;
        delete[]mapping1;
        delete[]mapping2;
    }
    void variation()
    {
        for (int i = 0; i < rand() % (nCities - 1); i++)
        {
            int pos = rand() % (nCities - 1);	// ���ѡ�����λ��
            int temp = path[pos];	// ����ѡ�еĻ���ͺ���һλ���򽻻�
            path[pos] = path[pos + 1];
            path[pos + 1] = temp;
        }
    }
};
unit bestone = { INT_MAX, {0} };         //���Ž�  
vector<unit> population;
double point_table[nCities][nCities];  //distance  
double length_table[nCities][nCities];  //distance  

class saTSP
{
public:
    void init_dis(char a[]);                  // create matrix to storage the Distance each city
    void SA_TSP();
    void CalCulate_length(unit& p);  //���㳤��  
    void print(unit& p);             //��ӡһ����  
    bool Accept(unit& bestone, unit& temp, double t);//�½���Metropolis ׼����� 
    void cross_and_selection();
    void variation_and_selection();
private:
    double t;
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
    t = INITIAL_TEMP;
    const double t_min = 0.001; //�¶����ޣ����¶ȴﵽt_min ����ֹͣ����  

    //choose an initial solution ~  
    for (int i = 0; i < sizepop; i++)
    {
        unit temp;
        generate(temp.path, temp.path + nCities, GenbyOne(0));
        random_shuffle(temp.path, temp.path + nCities);
        CalCulate_length(temp);
        population.push_back(temp);
    }

    // while the stop criterion is not yet satisfied do  
    while (t > t_min)
    {
        for (int i = 0; i < L; i++)
        {
            cross_and_selection();
            variation_and_selection();
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

void saTSP::cross_and_selection()
{
    for (int i = 0; i < sizepop - 1; i += 2) {
        unit temp;
        memcpy(&temp, &population[i], sizeof(unit));
        population[i].exchange(population[i + 1]);
        CalCulate_length(population[i]);
        if (!Accept(temp, population[i], t))
        {
            memcpy(&population[i], &temp, sizeof(unit));
        }
    }
    //random_shuffle(population.begin(), population.end());
}

void saTSP::variation_and_selection()
{
    int probability = rand() % 100;	// �������Ϊ1%
    if (probability == 1) {
        int x = rand() % sizepop;	// ���ѡ��һ��Ⱦɫ�����
        unit temp;
        memcpy(&temp, &population[x], sizeof(unit));
        population[x].variation();
        CalCulate_length(population[x]);
        if (!Accept(temp, population[x], t))
        {
            memcpy(&population[x], &temp, sizeof(unit));
        }
    }
}

int main()
{
    saTSP sa;
    char a[20] = "att48";
    sa.init_dis(a);
    sa.SA_TSP();
    for (int i = 0; i < sizepop; i++)
    {
        if (population[i].length < bestone.length)
            memcpy(&bestone, &population[i], sizeof(unit));
    }
    sa.CalCulate_length(bestone);
    sa.print(bestone);
    return 0;
}