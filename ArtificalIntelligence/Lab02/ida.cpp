// Not opt! https://people.cs.nctu.edu.tw/~huanpo/AIandSearching/IDAstar.cpp
// Not done!
// Other optimization-> https://pdfs.semanticscholar.org/b2fb/9a3e8bbb62e07810b01d00746cc65f3d446e.pdf

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
using namespace std;

const int MAXN = 4, MAXSTEP = 70, MAXP = 15; //Maximum length of map size, Maximum Step, Maximum number of puzzle grids
int A[MAXN][MAXN];
const int dr[4] = {1, 0, -1, 0},
          dc[4] = {0, 1, 0, -1},
          opp[4] = {2, 3, 0, 1}; //delta row, delta column, opposite direction of (dr[i], dc[i])
//For example, (dr[0], dr[0]) is down, which means its opposite is up , then opp[0] is 2. (which is up)
int tr[MAXP], tc[MAXP], bound; //A row recording, A column recording
char rec[MAXSTEP];             //recording answer
const char name[4] = {'D', 'R', 'U', 'L'};
bool pass;

int heuristic(int A[][MAXN]) //we use Manhattan Distance as heuristic function
{
    int ret = 0;
    for (int i = 0; i < MAXN; i++)
        for (int j = 0; j < MAXN; j++)
            if (A[i][j] < 15)
                ret += abs(i - tr[A[i][j]]) + abs(j - tc[A[i][j]]);
    return ret;
}

bool valid(int r, int c) //check if a position is out of boundary
{
    return r >= 0 && r < MAXN && c >= 0 && c < MAXN;
}

int IDA(int depth, int r, int c, int cost, int pre) //Depth of this function, row of the blank, column of the blank, actual cost, previous opposite direction (avoiding stupid moves)
{
    if (pass)
        return bound;
    if (cost == 0) //you find an answer. output it
    {
        printf("%d\n", depth);
        for (int i = 0; i < depth; i++)
            printf("%c", rec[i]);
        printf("\n");
        pass = true;
        return bound;
    }
    int f = cost + heuristic(A);
    if (f > bound)
        return f;

    int minF = 0x3f3f3f3f;
    for (int i = 0; i < 4; i++) //check all possible direction
        if (i != pre)           //don't go for a stupid move. for example, go left after you go right
        {
            int nr = r + dr[i], nc = c + dc[i], oCost, nCost, num;

            if (valid(nr, nc))
            {
                //update new heuristic cost, new cost = actual cost + heuristic cost
                num = A[nr][nc];
                A[r][c] = num, A[nr][nc] = 15;
                rec[depth] = name[i];
                int t = IDA(depth + 1, nr, nc, cost + 1, opp[i]);
                A[r][c] = 15, A[nr][nc] = num;
                if (pass)
                    return bound;
                if (t < minF)
                    minF = t;
            }
        }
    return minF;
}

int main()
{
    for (int i = 0; i < MAXP; i++)
        tr[i] = i / 4, tc[i] = i % 4;
    pass = false;
    int sr, sc;
    for (int i = 0; i < MAXN; i++)
        for (int j = 0; j < MAXN; j++)
        {
            scanf("%d", &A[i][j]);
            if (A[i][j])
                A[i][j]--;
            else
                A[i][j] = 15, sr = i, sc = j;
        }
    printf("Begin to solve!\n");
    int cost = heuristic(A);
    bound = cost;

    while (!pass)
    {
        bound = IDA(0, sr, sc, cost, -1);
    }
    if (bound == 0x3f3f3f3f)
        printf("No solution!\n");
    return 0;
}