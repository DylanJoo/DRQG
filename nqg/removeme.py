import json

f1 = open('data/triples.train.small.v0.sample.pred.jsonl', 'r')

with open('data/goodread.txt', 'w') as f2:
    for line in f1:
        data=json.loads(line.strip())
        f2.write(f"passage: {data['passage']}\n")
        qlist = [f"{i}\t{q}" for (i, q) in enumerate(data['positive'])]
        f2.write("\n".join(qlist))
        f2.write("\n\n")
f1.close()

