import json                                                                                   
with open('../../../laczakol/shared/ecco_downloaded/ecco_downloaded.jsonl') as f:
    doc = json.loads(f.readline())
    print(list(doc.keys()))
