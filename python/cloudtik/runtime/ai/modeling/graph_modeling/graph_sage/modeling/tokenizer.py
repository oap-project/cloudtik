
from collections import OrderedDict


def tokenize_node_ids(df, config, homogeneous=True):
    def column_index(series, offset=0):
        return {k: v + offset for v, k in enumerate(series.value_counts().index.values)}

    # create dictionary of dictionary to store node mapping for all node types
    offset = 0
    mapping = OrderedDict()
    # create mapping dictionary between original IDs and incremental IDs starting at zero
    # Note: because GNN is converting the graph to homogeneous we need the homogeneous mapping here
    # i,e: node_0: [0, x] node_1: [x,y] node_2: [y,z]
    # each unique real node id will be assigned a indexed id.
    # all the node types indexed ids are fatten numbered in the order defined in "node_columns"
    # key in dict: node + "_2idx" stores a unique mapping from real id -> indexed id  mapping
    # column new_col_name: node + "_Idx" stores the indexed id
    col_map = {}
    for i, node in enumerate(config["node_columns"]):
        key = str(node + "_2idx")
        mapping[key] = column_index(df[config["node_columns"][i]], offset=offset)
        new_col_name = node + "_Idx"
        col_map[node] = new_col_name
        # add new Idx to dataframe
        df[new_col_name] = df[config["node_columns"][i]].map(mapping[key])
        if homogeneous:
            offset = len(mapping[key])
    print("Tokenize column map: ", col_map)
    return mapping, col_map
