import argparse
import bz2
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt1")
    parser.add_argument("--gt2")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--n", type=int, default=10000)
    args = parser.parse_args()

    gt1 = {}
    with bz2.open(args.gt1, 'rt') as fin:
        assert next(fin).strip().split('\t') == ['item', 'countries']
        for i, line in enumerate(fin, start=1):
            line = line.strip().split('\t')
            qid = line[0]
            countries = tuple(sorted(line[1].split('|')))
            gt1[qid] = countries
    print(f"{len(gt1)} QIDs in {args.gt1}")

    gt2 = {}
    with bz2.open(args.gt2, 'rt') as fin:
        assert next(fin).strip().split('\t') == ['item', 'countries']
        for i, line in enumerate(fin, start=1):
            line = line.strip().split('\t')
            qid = line[0]
            countries = tuple(sorted(line[1].split('|')))
            gt2[qid] = countries
    print(f"{len(gt2)} QIDs in {args.gt2}")

    all_qids = set(gt1.keys()).union(set(gt2.keys()))
    top_k = random.sample(all_qids, args.n)
    agreed = 0
    country_disagreements = {}
    for i, qid in enumerate(top_k, start=1):
        if gt1.get(qid) != gt2.get(qid):
            if i <= args.k:
                print(f"https://www.wikidata.org/wiki/{qid}")
                print(f"{args.gt1}:\t{gt1.get(qid)}")
                print(f"{args.gt2}:\t{gt2.get(qid)}")
            for c in set(gt1.get(qid, [])).symmetric_difference(set(gt2.get(qid, []))):
                country_disagreements[c] = country_disagreements.get(c, 0) + 1
        else:
            agreed += 1

    print(f"{100 * agreed / args.n:.1f}% in agreement.")
    print("Countries most in disagreement:")
    for c in sorted(country_disagreements, key=country_disagreements.get, reverse=True):
        print(f'{c}:\t{country_disagreements[c]}')

if __name__ == "__main__":
    main()