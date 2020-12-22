import argparse
import bz2
import csv
import json
import sys

import pandas as pd
from shapely.geometry import shape, Point

csv.field_size_limit(sys.maxsize)

def get_region_properties(properties_tsv):
    """List of properties used for directly linking Wikidata items to regions.

    e.g., P19: place of birth
    These are compiled based on knowledge of Wikidata and Marc Miquel's excellent work:
    https://github.com/marcmiquel/WDO/blob/e482a2df2b41d389945f3b82179b8b7ca338b8d5/src_data/wikipedia_diversity.py
    """
    expected_header = ['Property', 'Label']
    region_properties = []
    with open(properties_tsv, 'r') as fin:
        tsvreader = csv.reader(fin, delimiter='\t')
        assert next(tsvreader) == expected_header
        for line in tsvreader:
            property = line[0]
            label = line[1]
            region_properties.append((property, label))
    return region_properties

def get_aggregation_logic(aggregates_tsv):
    """Mapping of QIDs -> regions not directly associated with them.

    e.g., Sahrawi Arab Democratic Republic (Q40362) -> Western Sahara (Q6250)
    """
    expected_header = ['Aggregation', 'From', 'QID To', 'QID From']
    aggregation = {}
    with open(aggregates_tsv, 'r') as fin:
        tsvreader = csv.reader(fin, delimiter='\t')
        assert next(tsvreader) == expected_header
        for line in tsvreader:
            qid_to = line[2]
            qid_from = line[3]
            if qid_from:
                aggregation[qid_from] = qid_to
    return aggregation

def get_region_data(region_qids_tsv, region_geoms_geojson, aggregation_tsv):
    # load in canonical mapping of QID -> region name for labeling
    qid_to_region = {}
    with open(region_qids_tsv, 'r') as fin:
        tsvreader = csv.reader(fin, delimiter='\t')
        assert next(tsvreader) == ['Region', 'QID']
        for line in tsvreader:
            region = line[0]
            qid = line[1]
            qid_to_region[qid] = region
    print("Loaded {0} QID-region pairs -- e.g., Q31 is {1}".format(len(qid_to_region), qid_to_region['Q31']))
    aggregation = get_aggregation_logic(aggregation_tsv)
    for qid_from in aggregation:
        qid_to = aggregation[qid_from]
        if qid_to in qid_to_region:
            qid_to_region[qid_from] = qid_to_region[qid_to]
    print("{0} QID-region pairs after adding aggregations".format(len(qid_to_region)))

    # load in geometries for the regions identified via Wikidata
    with open(region_geoms_geojson, 'r') as fin:
        regions = json.load(fin)['features']
    region_shapes = {}
    for c in regions:
        qid = c['properties']['WIKIDATAID']
        if qid in qid_to_region:
            region_shapes[qid] = shape(c['geometry'])
    print("Loaded {0} region geometries".format(len(region_shapes)))
    missing = []
    for qid in qid_to_region:
        if qid not in region_shapes:
            not_aggregated = True
            for qid_from in aggregation:
                if aggregation[qid_from] == qid and qid_from in region_shapes:
                    not_aggregated = False
            if not_aggregated:
                missing.append('{0} ({1})'.format(qid_to_region[qid], qid))
    print("QIDs in regions but not shapes:")
    for m in missing:
        print(m)
    return region_shapes, qid_to_region

def main():
    """Get Wikidata region properties for all Wikipedia articles.
    Track which property they came from to compare later.
    Save the resulting dictionary to a TSV file. This process takes ~24 hours.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_tsv',
                        help="Output TSV with data.")
    parser.add_argument('--region_tsv',
                        default="./resources/base_regions_qids.tsv",
                        help="TSV with regions and corresponding Wikidata IDs")
    parser.add_argument('--region_geoms',
                        default="./resources/ne_10m_admin_0_map_units.geojson",
                        help='GeoJSON with region shapes for locating coordinates.')
    parser.add_argument('--properties_tsv',
                        default="./resources/country_properties.tsv",
                        help="TSV with Wikidata properties to check for regions")
    parser.add_argument('--aggregation_tsv',
                        default="./resources/country_aggregation.tsv",
                        help="TSV with details about which regions aggregate up to others.")

    args = parser.parse_args()

    # load in region data
    region_properties = get_region_properties(args.properties_tsv)
    region_shapes, qid_to_region = get_region_data(args.region_tsv, args.region_geoms, args.aggregation_tsv)

    # process Wikidata dump
    dump_fn = '/mnt/data/xmldatadumps/public/wikidatawiki/entities/latest-all.json.bz2'
    print("Building QID->properties map from {0}".format(dump_fn))
    items_written = 0
    num_has_coords = 0
    coords_mapped_to_region = 0
    poorly_formatted = 0
    qids_skipped_over = {}
    with open(args.output_tsv, 'w') as fout:
        print("Writing QID->Props to {0}".format(args.output_tsv))
        header = ['item', 'en_title', 'lat', 'lon', 'P625_region']
        for _, lbl in region_properties:
            header.append(lbl)
        tsvwriter = csv.DictWriter(fout, delimiter='\t', fieldnames=header)
        tsvwriter.writeheader()
        with bz2.open(dump_fn, 'rt') as fin:
            next(fin)
            for idx, line in enumerate(fin, start=1):
                try:
                    item_json = json.loads(line[:-2])
                except Exception:
                    try:
                        item_json = json.loads(line)
                    except Exception:
                        print("Error:", idx, line)
                        continue
                if idx % 100000 == 0:
                    print("{0} lines processed. {1} poorly formatted (skipped). {2} ({3:.3f}) items kept, {4} ({5:.3f}) w/ coordinates and {6} mapped to regions.".format(
                        idx, poorly_formatted, items_written, items_written / idx, num_has_coords, num_has_coords / items_written, coords_mapped_to_region))
                    print("Top ten skipped-over country-esque QIDs: {0}".format([(q, qids_skipped_over[q]) for q in sorted(qids_skipped_over, key=qids_skipped_over.get, reverse=True)[:10]]))
                qid = item_json.get('id', None)
                if not qid:
                    continue
                sitelinks = [l[:-4] for l in item_json.get('sitelinks', []) if l.endswith('wiki') and l != 'commonswiki' and l != 'specieswiki']
                # check that at least one wikipedia in list
                if not sitelinks:
                    continue
                en_title = item_json.get('sitelinks', {}).get('enwiki', {}).get('title', None)
                claims = item_json.get('claims', {})
                output_row = {'item':qid, 'en_title':en_title}
                regions = set()
                for prop, lbl in region_properties:
                    if prop in claims:
                        for statement in claims[prop]:
                            try:
                                value_qid = statement['mainsnak']['datavalue']['value']['id']
                            except KeyError:
                                poorly_formatted += 1
                                continue
                            if value_qid in qid_to_region:
                                output_row[lbl] = output_row.get(lbl, []) + [qid_to_region[value_qid]]
                                regions.add(qid_to_region[value_qid])
                            else:
                                qids_skipped_over[value_qid] = qids_skipped_over.get(value_qid, 0) + 1
                try:
                    lat = claims['P625'][0]['mainsnak']['datavalue']['value']['latitude']
                    lon = claims['P625'][0]['mainsnak']['datavalue']['value']['longitude']
                    output_row['lat'] = lat
                    output_row['lon'] = lon
                    num_has_coords += 1
                    if len(regions) == 0 or (len(regions) == 1 and 'United Kingdom' in regions):  # one of the region properties matched
                        pt = Point(lon, lat)
                        for c in region_shapes:
                            if region_shapes[c].contains(pt):
                                output_row['P625_region'] = qid_to_region[c]
                                coords_mapped_to_region += 1
                                break
                    else:
                        output_row['P625_region'] = "N/A"
                except KeyError:
                    pass
                if len(output_row) > 2:
                    tsvwriter.writerow(output_row)
                    items_written += 1

    print(
        "{0} lines processed. {1} poorly formatted (skipped). {2} ({3:.3f}) items kept, {4} ({5:.3f}) w/ coordinates and {6} mapped to regions.".format(
            idx, poorly_formatted, items_written, items_written / idx, num_has_coords, num_has_coords / items_written,
            coords_mapped_to_region))
    print("Top ten skipped-over country-esque QIDs: {0}".format(
        [(q, qids_skipped_over[q]) for q in sorted(qids_skipped_over, key=qids_skipped_over.get, reverse=True)[:10]]))


def combine_regions(row, cols=('P625_region', 'place of birth', 'country', 'country of citizenship')):
    regions = {}
    for col in cols:
        if type(row[col]) == str:
            if row[col].startswith('['):
                for c in eval(row[col]):
                    regions[c] = regions.get(c, 0) + 1
            else:
                regions[row[col]] = regions.get(row[col], 0) + 1
    return regions


def desc_stats(output_tsv):
    df = pd.read_csv(output_tsv, sep='\t')
    assert list(df.columns)[:4] == ['item', 'en_title', 'lat', 'lon']
    pct_non_null = df.count() / len(df)
    for col in pct_non_null.index:
        print("{0}: {1:.1f}% non-null".format(col, pct_non_null[col] * 100))
    print("{0:.1f}% items with a lat/lon that matched a region.".format(100 * df['P625_region'].count() / df['lat'].count()))
    print("{0:.1f}% items with a lat/lon that matched a otherwise-missing region.".format(
        100 * df['P625_region'].count() / df['lat'].count()))
    print("{0:.1f}% items with either lat/lon region or P17 region.".format(100 * (~df[['P625_region', 'country']].isnull()).any(axis=1).sum() / len(df)))
    print("{0:.1f}% items with either birth region or P17 region.".format(100 * (~df[['place of birth', 'country']].isnull()).any(axis=1).sum() / len(df)))
    print("{0:.1f}% items with either citizenship region or P17 region.".format(100 * (~df[['country of citizenship', 'country']].isnull()).any(axis=1).sum() / len(df)))
    df['region_aggregate'] = df.apply(combine_regions, axis=1)
    return df

def data_to_figshare_format(df, output_json):
    df['region_list'] = df['region_aggregate'].apply(lambda x: sorted(set(c for c in x)))
    with open(output_json, 'w') as fout:
        df[['item', 'region_list']].to_json(fout, orient='records', lines=True)


if __name__ == "__main__":
    main()