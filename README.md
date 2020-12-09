## wiki-region-groundtruth
Code and resources for building a basic, starter set of groundtruth for Wikidata items (and by extension Wikipedia articles) and their associated regions. The words "country" and "region" are used largely interchangeably in this repository. See `resources/REGIONS.md` for more details.

### Scripts
#### `gather_wikidata_region_groundtruth.py`
This script loops through the [Wikidata JSON dump](https://www.wikidata.org/wiki/Wikidata:Database_download#JSON_dumps_(recommended)) and extracts Wikidata items that are explicitly linked to the provided set of regions.
This results in a list of Wikidata items and regions that with high confidence are associated with them. The region identification is done both by:
  * directly matching values to the list of regions for a select set of properties like `place of birth` or `country of citizenship`
  * or, geolocating coordinates associated with the Wikidata item to the regions in `resources/ne_10m_admin_0_map_units.geojson` 

While I consider this data to be high precision (100% precision to be exact), I do not expect the recall to be 100% and so additional methods are needed to extend the data to more items. A few obvious examples:
* Incomplete Wikidata items
* People who are listed as born in a city instead of the country would be missed by this script
* Items that are tied to former countries like the Soviet Union (which could not be directly mapped to a modern country without more information)

### Resources
* `resources/base_regions_qids.tsv`: a list of regions and associated QIDs that is used when checking whether a given Wikidata item is directly associated with a given region.
For the most part, these regions are countries but they span a range of statuses (from territories to limited-recognition states to widely-recognized sovereign nations). See `resources/REGIONS.md` for more information. 
* `resources/country_aggregation.tsv`: a few (tiny) regions that have their own Wikidata IDs but for the purposes of this project are merged with larger entities.
* `resources/country_properties.tsv`: a list of Wikidata properties that are checked for values that match the list of regions.
* `resources/ne_10m_admin_0_map_units.geojson`: country borders that are used to for matching Wikidata coordinates to a region. These come from [Natural Earth](https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-0-details/) and are public domain.