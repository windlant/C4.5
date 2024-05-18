from graphviz import Digraph


# 获取所有节点中最多子节点的叶节点
def getMaxLeafs(myTree):
    numLeaf = len(myTree.keys())
    for key, value in myTree.items():
        if isinstance(value, dict):
            sum_numLeaf = getMaxLeafs(value)
            if sum_numLeaf > numLeaf:
                numLeaf = sum_numLeaf
    return numLeaf


def plot_model(tree, name):
    g = Digraph("G", filename=name, format='png', strict=False)

    first_label = list(tree.keys())[0]
    g.node("0", first_label, fontname="FangSong")
    _sub_plot(g, tree, "0")
    leafs = str(getMaxLeafs(tree) // 10)
    g.attr(rankdir='RB', ranksep=leafs)
    g.view()


root = "0"


def _sub_plot(g, tree, inc):
    global root

    first_label = list(tree.keys())[0]
    ts = tree[first_label]
    point = tree["point"]
    point = round(point, 2)

    for i in ts.keys():
        edge_str = i
        if i == 0 and point != 0:
            edge_str = "<=" + str(point)
        if i == 1 and point != 0:
            edge_str = ">" + str(point)
        if isinstance(tree[first_label][i], dict):
            root = str(int(root) + 1)
            g.node(root, list(tree[first_label][i].keys())[0], fontname="FangSong")
            g.edge(inc, root, edge_str, fontname="FangSong")
            _sub_plot(g, tree[first_label][i], root)
        else:
            root = str(int(root) + 1)
            g.node(root, str(tree[first_label][i]), fontname="FangSong")
            g.edge(inc, root, edge_str, fontname="FangSong")


# tree = {
#         "tearRate": {
#             "reduced": "no lenses",
#             "normal": {
#                 "astigmatic": {
#                     "yes": {
#                         "prescript": {
#                             "myope": "hard",
#                             "hyper": {
#                                 "age": {
#                                     "young": "hard",
#                                     "presbyopic": "no lenses",
#                                     "pre": "no lenses"
#                                 }
#                             }
#                         }
#                     },
#                     "no": {
#                         "age": {
#                             "young": "soft",
#                             "presbyopic": {
#                                 "prescript": {
#                                     "myope": "no lenses",
#                                     "hyper": "soft"
#                                 }
#                             },
#                             "pre": "soft"
#                         }
#                     }
#                 }
#             }
#         }
#     }
# plot_model(tree, "tree.gv")
