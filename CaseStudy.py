"""
INSTRUCTIONS:
1. Ensure the data file 'ocel2-export.json' is in the same folder as this script.
2. This script requires the following external libraries:
   - pm4py
   - networkx
   - python-louvain (for community detection)

   You can install them by running:
   pip install pm4py networkx python-louvain
"""

import pm4py
import networkx as nx
import community.community_louvain as community_louvain
import itertools
import os


def get_golden_model_structure():
    """
    Structural configuration of the Order Management OCPN.
    False: ordinary arc. True: variable arc.
    """
    return {
        "place order": {
            "customers": False,
            "orders": False,
            "items": True,  # Thick arc
            "products": True  # Thick arc
        },
        "confirm order": {
            "orders": False,
            "customers": False,
            "employees": False,
            "items": True,  # Thick arc
            "products": True  # Thick arc
        },
        "pay order": {
            "orders": False,
            "items": True,  # Thick arc
            "products": True  # Thick arc
        },
        "payment reminder": {
            "orders": False,
            "items": True,  # Thick arc
            "products": True  # Thick arc
        },
        "pick item": {
            "items": False,  # Thin arc
            "employees": False,
            "products": False
        },
        "item out of stock": {
            "items": False,
            "employees": False,
            "products": False
        },
        "reorder item": {
            "items": False,
            "employees": False,
            "products": False
        },
        "create package": {
            "packages": False,
            "employees": False,
            "items": True,  # Thick arc
            "products": True  # Thick arc
        },
        "send package": {
            "packages": False,
            "items": True,  # Thick arc
            "products": True,  # Thick arc
            "employees": True  # Thick arc (Variable)
        },
        "package delivered": {
            "packages": False,
            "employees": False,
            "items": True,  # Thick arc
            "products": True  # Thick arc
        },
        "failed delivery": {
            "packages": False,
            "employees": False,
            "items": True,  # Thick arc
            "products": True  # Thick arc
        }
    }


def clean_dataframe(df, possible_names, target_name):
    """
    Searches for a column from possible_names and renames it to target_name.
    """
    if df.index.name in possible_names:
        df = df.reset_index()

    for col in possible_names:
        if col in df.columns:
            return df.rename(columns={col: target_name})

    if target_name in ['_OID', '_EID'] and 'index' in df.columns:
        return df.rename(columns={'index': target_name})

    return df


def prepare_data_tables(ocel):
    """
    Extracts and standardizes Objects and Relations tables.
    """
    if not hasattr(ocel, 'objects') or not hasattr(ocel, 'relations'):
        print("   [Error] OCEL object is not valid.")
        return None, None

    objs = ocel.objects.copy()
    objs = clean_dataframe(objs, ['ocel:oid', 'oid', 'object_id', 'id'], '_OID')
    objs = clean_dataframe(objs, ['ocel:type', 'type', 'object_type'], '_TYPE')

    rels = ocel.relations.copy()
    rels = clean_dataframe(rels, ['ocel:eid', 'eid', 'event_id', 'id'], '_EID')
    rels = clean_dataframe(rels, ['ocel:oid', 'oid', 'object_id', 'objectId'], '_OID')

    req_obj = {'_OID', '_TYPE'}
    req_rel = {'_EID', '_OID'}

    if not req_obj.issubset(objs.columns):
        print(f"   [Error] Object table missing columns. Found: {objs.columns}")
        return None, None
    if not req_rel.issubset(rels.columns):
        print(f"   [Error] Relations table missing columns. Found: {rels.columns}")
        return None, None

    return objs, rels

def calculate_sr_golden(golden_model, ot1, ot2):
    acts_1 = {act for act, types in golden_model.items() if ot1 in types}
    acts_2 = {act for act, types in golden_model.items() if ot2 in types}
    if not acts_1 or not acts_2: return 0.0
    intersect = len(acts_1.intersection(acts_2))
    union = len(acts_1.union(acts_2))
    return intersect / union if union > 0 else 0.0


def calculate_ci_golden(golden_model, ot1, ot2):
    for act, types in golden_model.items():
        if ot1 in types and ot2 in types:
            v1 = types[ot1]
            v2 = types[ot2]
            if v1 != v2: return 1.0  # One-to-Many
            if v1 and v2: return 1.0  # Many-to-Many
    return 0.0


def calculate_lco_clean(objs, rels, ot1, ot2):
    def get_real_type(target, available):
        for a in available:
            if str(a).lower() == str(target).lower(): return a
            if str(a).lower() + 's' == str(target).lower(): return a
            if str(target).lower() + 's' == str(a).lower(): return a
        return None

    all_types = objs['_TYPE'].unique()
    real_ot1 = get_real_type(ot1, all_types)
    real_ot2 = get_real_type(ot2, all_types)

    if not real_ot1 or not real_ot2: return 0.0

    # Get OIDs
    oids_1 = objs[objs['_TYPE'] == real_ot1]['_OID']
    oids_2 = objs[objs['_TYPE'] == real_ot2]['_OID']

    # Get EIDs
    eids_1 = set(rels[rels['_OID'].isin(oids_1)]['_EID'])
    eids_2 = set(rels[rels['_OID'].isin(oids_2)]['_EID'])

    if not eids_1 or not eids_2: return 0.0

    inter = len(eids_1.intersection(eids_2))
    union = len(eids_1.union(eids_2))
    return inter / union

def classify_activities(golden_model, partition):
    """
    Maps activities to modules.
    - Internal activities are assigned to their specific module.
    - Interface activities are REPLICATED across all involved modules.
    """
    module_content = {}

    unique_modules = set(partition.values())
    for m in unique_modules:
        module_content[m] = {'internal': [], 'interface': []}

    for activity, connections in golden_model.items():
        involved_modules = set()
        for ot in connections.keys():
            if ot in partition:
                involved_modules.add(partition[ot])

        if len(involved_modules) == 1:
            mod_id = list(involved_modules)[0]
            module_content[mod_id]['internal'].append(activity)

        elif len(involved_modules) > 1:
            for mod_id in involved_modules:
                module_content[mod_id]['interface'].append(activity)

    return module_content

# MAIN EXECUTION

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path relative to the script
    file_path = os.path.join(script_dir, "ocel2-export.json")

    model = get_golden_model_structure()
    object_types = list(set(t for types in model.values() for t in types))
    print(f"1. Model Types: {object_types}")

    print("2. Loading Data...")
    if os.path.exists(file_path):
        ocel = pm4py.read_ocel2_json(file_path)
        objs, rels = prepare_data_tables(ocel)
        if objs is not None:
            print(f"   Log Types: {objs['_TYPE'].unique()}")
    else:
        print("   File not found.")
        objs, rels = None, None

    print("\n3. Scores:")
    w1, w2, w3 = 1/3, 1/3, 1/3
    G = nx.Graph()
    pairs = list(itertools.combinations(object_types, 2))

    print(f"{'Pair':<30} | {'SR':<5} | {'CI':<5} | {'LCO':<5} | {'Total'}")
    print("-" * 65)

    for ot1, ot2 in pairs:
        sr = calculate_sr_golden(model, ot1, ot2)
        ci = calculate_ci_golden(model, ot1, ot2)
        lco = 0.0
        if objs is not None:
            lco = calculate_lco_clean(objs, rels, ot1, ot2)

        score = (w1 * sr) + (w2 * ci) + (w3 * lco)
        print(f"({ot1}, {ot2}):".ljust(30) + f" | {sr:.2f}  | {ci:.2f}  | {lco:.2f}  | {score:.2f}")

        if score > 0.01:
            G.add_edge(ot1, ot2, weight=score)

    print("\n4. Decomposing...")
    if len(G.edges) > 0:
        partition = community_louvain.best_partition(G, weight='weight', resolution=1.0)

        # --- PRINT RESULTS ---
        clusters = {}
        for ot, cid in partition.items():
            if cid not in clusters: clusters[cid] = []
            clusters[cid].append(ot)

        module_activities = classify_activities(model, partition)

        for cid, members in clusters.items():
            print(f"\n[MODULE {cid}]")
            print(f"  Objects:    {members}")

            # Print Internal Activities
            internal = module_activities[cid]['internal']
            if internal:
                print(f"  Internal:   {internal}")
            else:
                print(f"  Internal:   []")

            # Print Interface Activities
            interface = module_activities[cid]['interface']
            if interface:
                print(f"  Interface:  {interface}  <-- Synchronizes with other modules")
            else:
                print(f"  Interface:  []")
