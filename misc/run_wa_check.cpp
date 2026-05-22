#include <iostream>
#include <vector>
#include <map>
#include <queue>
#include <cstring>
using namespace std;

typedef pair<long long, long long> pll;

const int MAXN = 6005;
const int INF = 1000000000;

struct Edge {
    int to, cap, rev;
};

vector<Edge> graph[MAXN];
int level[MAXN], iter_[MAXN];

void add_edge(int from, int to, int cap) {
    Edge e1, e2;
    e1.to = to; e1.cap = cap; e1.rev = (int)graph[to].size();
    e2.to = from; e2.cap = 0; e2.rev = (int)graph[from].size();
    graph[from].push_back(e1);
    graph[to].push_back(e2);
}

bool bfs(int s, int t) {
    memset(level, -1, sizeof(level));
    queue<int> q;
    level[s] = 0;
    q.push(s);
    while (!q.empty()) {
        int v = q.front(); q.pop();
        for (int i = 0; i < (int)graph[v].size(); i++) {
            Edge &e = graph[v][i];
            if (e.cap > 0 && level[e.to] < 0) {
                level[e.to] = level[v] + 1;
                q.push(e.to);
            }
        }
    }
    return level[t] >= 0;
}

int dfs(int v, int t, int f) {
    if (v == t) return f;
    for (int &i = iter_[v]; i < (int)graph[v].size(); i++) {
        Edge &e = graph[v][i];
        if (e.cap > 0 && level[v] < level[e.to]) {
            int d = dfs(e.to, t, min(f, e.cap));
            if (d > 0) {
                e.cap -= d;
                graph[e.to][e.rev].cap += d;
                return d;
            }
        }
    }
    return 0;
}

int max_flow(int s, int t) {
    int flow = 0;
    while (bfs(s, t)) {
        memset(iter_, 0, sizeof(iter_));
        int d;
        while ((d = dfs(s, t, INF)) > 0)
            flow += d;
    }
    return flow;
}

int main() {
    int n;
    while (cin >> n) {
        map<pll, int> nodeId;
        int cnt = 0;
        
        auto getId = [&](long long x, long long y) {
            pll p = make_pair(x, y);
            if (nodeId.find(p) == nodeId.end())
                nodeId[p] = cnt++;
            return nodeId[p];
        };

        vector<pair<int,int> > boards;
        for (int i = 0; i < n; i++) {
            long long x1, y1, x2, y2;
            cin >> x1 >> y1 >> x2 >> y2;
            int u = getId(x1, y1);
            int v = getId(x2, y2);
            boards.push_back(make_pair(u, v));
        }

        long long sx, sy, ex, ey;
        cin >> sx >> sy >> ex >> ey;
        int src = getId(sx, sy);
        int snk = getId(ex, ey);

        for (int i = 0; i < 2 * cnt + 5; i++) {
            graph[i].clear();
        }

        for (int i = 0; i < cnt; i++) {
            int cap = (i == src || i == snk) ? INF : 1;
            add_edge(2 * i, 2 * i + 1, cap);
        }

        for (int i = 0; i < (int)boards.size(); i++) {
            int u = boards[i].first, v = boards[i].second;
            add_edge(2 * u + 1, 2 * v, INF);
            add_edge(2 * v + 1, 2 * u, INF);
        }

        cout << max_flow(2 * src, 2 * snk + 1) << endl;
    }
    return 0;
}
