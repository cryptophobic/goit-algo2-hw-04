"""
Edmonds–Karp для логістичної мережі з JSON-конфігом.

JSON-схема (мінімум):
{
  "terminals": ["Термінал 1", "Термінал 2"],
  "warehouses": ["Склад 1", "Склад 2", "Склад 3", "Склад 4"],
  "stores": ["Магазин 1", "...", "Магазин 14"],
  "edges": {
    "terminal_to_warehouse": [{"from":"Термінал 1","to":"Склад 1","cap":25}, ...],
    "warehouse_to_store":    [{"from":"Склад 1","to":"Магазин 1","cap":15}, ...]
  }
}
"""

from collections import deque, defaultdict
from dataclasses import dataclass
import argparse, json, csv, sys
from typing import Dict, List, Tuple

# ---------- Дефолтний сценарій із завдання ----------
DEFAULT = {
    "terminals": ["Термінал 1", "Термінал 2"],
    "warehouses": ["Склад 1", "Склад 2", "Склад 3", "Склад 4"],
    "stores": [f"Магазин {i}" for i in range(1, 15)],
    "edges": {
        "terminal_to_warehouse": [
            {"from":"Термінал 1","to":"Склад 1","cap":25},
            {"from":"Термінал 1","to":"Склад 2","cap":20},
            {"from":"Термінал 1","to":"Склад 3","cap":15},
            {"from":"Термінал 2","to":"Склад 3","cap":15},
            {"from":"Термінал 2","to":"Склад 4","cap":30},
            {"from":"Термінал 2","to":"Склад 2","cap":10},
        ],
        "warehouse_to_store": [
            {"from":"Склад 1","to":"Магазин 1","cap":15},
            {"from":"Склад 1","to":"Магазин 2","cap":10},
            {"from":"Склад 1","to":"Магазин 3","cap":20},
            {"from":"Склад 2","to":"Магазин 4","cap":15},
            {"from":"Склад 2","to":"Магазин 5","cap":10},
            {"from":"Склад 2","to":"Магазин 6","cap":25},
            {"from":"Склад 3","to":"Магазин 7","cap":20},
            {"from":"Склад 3","to":"Магазин 8","cap":15},
            {"from":"Склад 3","to":"Магазин 9","cap":10},
            {"from":"Склад 4","to":"Магазин 10","cap":20},
            {"from":"Склад 4","to":"Магазин 11","cap":10},
            {"from":"Склад 4","to":"Магазин 12","cap":15},
            {"from":"Склад 4","to":"Магазин 13","cap":5},
            {"from":"Склад 4","to":"Магазин 14","cap":10},
        ]
    }
}

# ---------- Утиліти ----------
@dataclass
class Network:
    nodes: List[str]
    idx: Dict[str, int]
    C: List[List[int]]
    terminals: List[str]
    warehouses: List[str]
    stores: List[str]

def _add(C, idx, u, v, c):
    C[idx[u]][idx[v]] += int(c)

def build_network_from_json(cfg: dict) -> Network:
    # базова валідація
    need_keys = ["terminals","warehouses","stores","edges"]
    for k in need_keys:
        if k not in cfg:
            raise ValueError(f"У конфігу відсутній ключ: '{k}'")
    for k in ["terminal_to_warehouse","warehouse_to_store"]:
        if k not in cfg["edges"]:
            raise ValueError(f"У 'edges' відсутній ключ: '{k}'")

    terminals = list(cfg["terminals"])
    warehouses = list(cfg["warehouses"])
    stores = list(cfg["stores"])
    nodes = ["S"] + terminals + warehouses + stores + ["T"]
    idx = {n:i for i,n in enumerate(nodes)}
    n = len(nodes)
    C = [[0]*n for _ in range(n)]

    # Встановимо "великий" капасіті на S->термінали та Магазин->T
    total_store_cap = sum(int(e["cap"]) for e in cfg["edges"]["warehouse_to_store"])
    BIG = max(1, total_store_cap)
    for t in terminals:
        _add(C, idx, "S", t, BIG)
    for s in stores:
        _add(C, idx, s, "T", BIG)

    # Термінал->Склад
    def ensure(name, pool, kind):
        if name not in pool:
            raise ValueError(f"'{name}' не знайдено серед {kind}")
    for e in cfg["edges"]["terminal_to_warehouse"]:
        u, v, cap = e["from"], e["to"], e["cap"]
        ensure(u, set(terminals), "терміналів")
        ensure(v, set(warehouses), "складів")
        _add(C, idx, u, v, cap)

    # Склад->Магазин
    for e in cfg["edges"]["warehouse_to_store"]:
        u, v, cap = e["from"], e["to"], e["cap"]
        ensure(u, set(warehouses), "складів")
        ensure(v, set(stores), "магазинів")
        _add(C, idx, u, v, cap)

    return Network(nodes, idx, C, terminals, warehouses, stores)

# ---------- Edmonds–Karp ----------
def edmonds_karp(C, s, t, names=None, explain=False):
    n = len(C)
    F = [[0]*n for _ in range(n)]
    maxflow = 0
    iters = 0

    while True:
        iters += 1
        parent = [-1]*n
        parent[s] = s
        m = [0]*n
        m[s] = float('inf')
        q = deque([s])

        while q:
            u = q.popleft()
            for v in range(n):
                if C[u][v] - F[u][v] > 0 and parent[v] == -1:
                    parent[v] = u
                    m[v] = min(m[u], C[u][v] - F[u][v])
                    if v == t:
                        q.clear()
                        break
                    q.append(v)

        if parent[t] == -1:
            iters -= 1  # остання безрезультатна спроба
            break

        inc = m[t]
        maxflow += inc

        # пояснення: шлях + приріст
        if explain and names:
            path = []
            v = t
            while v != s:
                path.append(v)
                v = parent[v]
            path.append(s)
            path.reverse()
            print(f"[EK] path {iters:02d}: " + " -> ".join(names[i] for i in path) + f" | +{int(inc)}")

        # аугментація
        v = t
        while v != s:
            u = parent[v]
            F[u][v] += inc
            F[v][u] -= inc
            v = u

    if explain:
        print(f"[EK] total iterations: {iters}")
    return int(maxflow), F

# ---------- Аналіз/пояснення ----------
def residual(C, F):
    n = len(C)
    R = [[0]*n for _ in range(n)]
    for u in range(n):
        for v in range(n):
            R[u][v] = C[u][v] - F[u][v]
    return R

def min_cut_edges(net: Network, F):
    C, nodes, idx = net.C, net.nodes, net.idx
    R = residual(C, F)
    n = len(nodes)
    vis = [False]*n
    q = deque([idx["S"]]); vis[idx["S"]] = True
    while q:
        u = q.popleft()
        for v in range(n):
            if R[u][v] > 0 and not vis[v]:
                vis[v] = True; q.append(v)
    S_side = {nodes[i] for i,ok in enumerate(vis) if ok}
    T_side = set(nodes) - S_side
    cut = []
    for u in S_side:
        for v in T_side:
            if C[idx[u]][idx[v]] > 0:
                cut.append((u,v,C[idx[u]][idx[v]]))
    return cut

def edge_flows(net: Network, F) -> Dict[Tuple[str,str],int]:
    flows = {}
    for u_name,u in net.idx.items():
        for v_name,v in net.idx.items():
            c = net.C[u][v]
            if c > 0 and F[u][v] > 0:
                flows[(u_name,v_name)] = int(F[u][v])
    return flows

def decompose_T_to_S(flows, terminals, warehouses, stores):
    # збір частин потоку
    tw = defaultdict(int); ws = defaultdict(int); sT = defaultdict(int)
    for (u,v),f in flows.items():
        if u in terminals and v in warehouses: tw[(u,v)] += f
        elif u in warehouses and v in stores: ws[(u,v)] += f
        elif u in stores and v == "T": sT[u] += f

    W_in = defaultdict(lambda: defaultdict(int))
    for (t,w),f in tw.items(): W_in[w][t]+=f
    W_out = defaultdict(lambda: defaultdict(int))
    for (w,s),f in ws.items(): W_out[w][s]+=f

    result = defaultdict(int)
    for w in warehouses:
        tin = dict(W_in[w]); tout = dict(W_out[w])
        if not tin or not tout: continue
        tkeys = list(tin.keys()); skeys = list(tout.keys())
        ti = si = 0
        while sum(tin.values())>0 and sum(tout.values())>0:
            t = tkeys[ti % len(tkeys)]
            s = skeys[si % len(skeys)]
            if tin[t]==0: ti+=1; continue
            if tout[s]==0: si+=1; continue
            d = min(tin[t], tout[s])
            tin[t]-=d; tout[s]-=d; result[(t,s)]+=d
            if tout[s]==0: si+=1
            if tin[t]==0: ti+=1
    return result

def print_table(ts):
    rows = sorted(ts.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    hdr = ["Термінал","Магазин","Фактичний потік (од.)"]
    line = "-"*len(" | ".join(hdr))
    print("\nТаблиця: Потоки між терміналами та магазинами")
    print(line); print(" | ".join(hdr)); print(line)
    for (t,s),v in rows:
        print(f"{t:10s} | {s:12s} | {v:6d}")
    print(line)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Максимальний потік (Edmonds–Karp) для логістики з JSON-конфігом")
    ap.add_argument("--config", help="JSON з описом мережі")
    ap.add_argument("--csv", help="зберегти таблицю Термінал→Магазин у CSV")
    ap.add_argument("--dump-default", help="записати дефолтну мережу у JSON і вийти")
    ap.add_argument("--explain", action="store_true", help="друкувати кроки (шляхи BFS та прирости)")
    args = ap.parse_args()

    if args.dump_default:
        with open(args.dump_default, "w", encoding="utf-8") as f:
            json.dump(DEFAULT, f, ensure_ascii=False, indent=2)
        print(f"✅ Збережено шаблон у {args.dump_default}")
        return

    cfg = DEFAULT
    if args.config:
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as e:
            print(f"❌ Не вдалося прочитати конфіг: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        net = build_network_from_json(cfg)
    except Exception as e:
        print(f"❌ Помилка валідації мережі: {e}", file=sys.stderr)
        sys.exit(1)

    s, t = net.idx["S"], net.idx["T"]
    maxflow, F = edmonds_karp(net.C, s, t, net.nodes, explain=args.explain)
    flows = edge_flows(net, F)
    ts = decompose_T_to_S(flows, net.terminals, net.warehouses, net.stores)

    print("\n=== РЕЗУЛЬТАТИ ===")
    print(f"Максимальний потік: {maxflow} од.")

    cut = min_cut_edges(net, F)
    print("\nМінімальний розріз (вузькі місця):")
    for u,v,c in cut:
        tag = " (критичне Т→Ск)" if (u in net.terminals and v in net.warehouses) else ""
        print(f"  {u} -> {v}: {c}{tag}")

    print_table(ts)

    # агрегати
    per_t = defaultdict(int)
    per_s = defaultdict(int)
    for (tname, sname), v in ts.items():
        per_t[tname]+=v; per_s[sname]+=v

    print("Відправлено терміналами:")
    for tname in net.terminals:
        print(f"  {tname}: {per_t[tname]} од.")

    print("\nОтримано магазинами:")
    for sname in net.stores:
        print(f"  {sname}: {per_s[sname]} од.")

    # Q&A
    print("\n=== Аналітика ===")
    if per_t:
        top_t = max(per_t.items(), key=lambda kv: kv[1])[0]
        print(f"1) Найбільший потік забезпечує: {top_t}")
    print("2) Найкритичніші маршрути — ребра Термінал→Склад у мінрозрізі; вони обмежують загальний потік.")
    nonzero = [(s,v) for s,v in per_s.items() if v>0]
    least_nz = min(nonzero, key=lambda kv: kv[1])[0] if nonzero else None
    zeros = [s for s in net.stores if per_s[s]==0]
    print(f"3) Мінімальні поставки: {least_nz if least_nz else '—'}, нульові: {', '.join(zeros) if zeros else 'немає'}.")
    print("   Збільшення можливе через підвищення місткостей відповідних ребер Т→Ск.")
    print("4) Вузькі місця: виходи з терміналів. Розширення їх місткості підвищить max-flow і дасть змогу використати резерви Ск→Маг.")

    if args.csv:
        try:
            with open(args.csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Термінал","Магазин","Фактичний потік (од.)"])
                for (tname, sname), v in sorted(ts.items()):
                    w.writerow([tname, sname, v])
            print(f"\n💾 CSV збережено: {args.csv}")
        except Exception as e:
            print(f"❌ Не вдалося записати CSV: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
