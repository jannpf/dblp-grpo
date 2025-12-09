import os
import re
import json
import gzip
import dotenv

from rdflib import RDF, RDFS, Graph
from tqdm import tqdm
from urllib.parse import urlsplit

import DBLPDataset as DBLPDataset


dotenv.load_dotenv()
DATASET_PATH = os.getenv("DATASET_PATH").rstrip('/')


def explode(all_items: list) -> set:
    return set(item for sublist in all_items for item in sublist)


def extract_from_dump(dump_path: str, uri_sets: dict[set[str]]) -> dict:
    # this is dirty, but the file has to be read only once this way
    # Initialize results per set/file
    results = {
        subset: {uri: {"types": [], "label": []} for uri in s}
        for subset, s in uri_sets.items()
    }
    # regex to match a simple triple: <s> <p> <o> .
    triple_re = re.compile(
        r'^<([^>]+)>\s+<([^>]+)>\s+(".*?"|<[^>]+>)\s*\.\s*$')

    rdf_type_uri = str(RDF.type)
    rdfs_label_uri = str(RDFS.label)
    # counting lines takes long -> filesize
    total_dump_size = os.path.getsize(dump_path)

    # read binary, because otherwise progress cant be tracked
    with open(dump_path, 'rb') as fh, gzip.GzipFile(fileobj=fh) as gz, \
            tqdm(total=total_dump_size, unit='B', unit_scale=True, desc='Reading dump') as pbar:
        for raw_line in gz:
            # update progress bar based on the stream position
            pbar.update(fh.tell() - pbar.n)

            line = raw_line.decode('utf-8', errors='ignore')
            m = triple_re.match(line)
            if not m:
                continue
            subj, pred, obj = m.groups()
            # Check membership in any of the sets
            for subset, uris in uri_sets.items():
                if subj in uris:
                    if pred == rdf_type_uri and obj.startswith('<'):
                        val = obj[1:-1].rsplit('/', 1)[-1]
                        results[subset][subj]["types"].append(val)
                    elif pred == rdfs_label_uri and obj.startswith('"'):
                        # object should be a literal "..."
                        # remove surrounding quotes, ignore language tag
                        val = obj.strip('"').split('"', 1)[0]
                        results[subset][subj]["label"].append(val)

    out = dict()
    for subset, data in results.items():
        out[subset] = []
        for uri, d in data.items():
            out[subset].append({
                "uri": uri,
                "types": list(set(d["types"])),
                "label": list(set(d["label"]))
            })

    return out


def comments_from_schema(schema_path: str, uri_sets: dict[set[str]]) -> dict:
    # construct prop ids, as in the schema there are no full uris
    # a bit hacky but seems to work
    def get_id(uri: str) -> str:
        return uri.split('/')[-1].split('#')[-1]

    def frag_or_uri(u) -> str:
        if not u:
            return ""
        frag = urlsplit(str(u)).fragment
        return f"#{frag}" if frag else str(u)

    g = Graph()
    g.parse(schema_path, format="xml")

    props = {}
    for s in g.subjects(RDF.type, RDF.Property):
        props[get_id(str(s))] = {
            "label": str(g.value(s, RDFS.label, default="")),
            "comment": str(g.value(s, RDFS.comment, default="")),
            "domain": frag_or_uri(g.value(s, RDFS.domain)),
            "range": frag_or_uri(g.value(s, RDFS.range)),
        }

    results = {}
    for subset, uris in uri_sets.items():
        results[subset] = [
            {
                "uri": uri,
                "types": ["relation"],
                "label": props.get(get_id(uri), {}).get("label", ""),
                "comment": props.get(get_id(uri), {}).get("comment", ""),
                "domain": props.get(get_id(uri), {}).get("domain", ""),
                "range": props.get(get_id(uri), {}).get("range", ""),
            }
            for uri in uris
        ]

    return results


def main():
    dump_file = f"{DATASET_PATH}/dblp.nt.gz"
    schema_file = f"{DATASET_PATH}/schema.rdf"
    if not os.path.isfile(dump_file):
        raise FileNotFoundError(f"Dump file not found at {dump_file}")
    entities = dict()
    relations = dict()
    results = {}

    subsets = ["train", "valid", "test"]

    for subset in subsets:
        ds = DBLPDataset.DblpDataset(DATASET_PATH, subset)
        entities[subset] = {s.strip('<>') for s in explode(ds.questions["entities"])}
        relations[subset] = {s.strip('<>') for s in explode(ds.questions["relations"])}

    try:
        entity_details = extract_from_dump(dump_file, entities)
        relation_details = comments_from_schema(schema_file, relations)

        for subset in subsets:
            results[subset] = entity_details[subset] + relation_details[subset]

    except Exception as e:
        print(f"Error fetching details: {e}")
        raise e

    for dataset, meta in results.items():
        output_json = f"{DATASET_PATH}/{dataset}/dblp_metadata.json"

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
