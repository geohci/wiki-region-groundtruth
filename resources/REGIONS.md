## Regions

### Principles
There is no good answer for what is a definitive set of non-overlapping regions in the world. While many of the regions included are countries, this is not intended to be a list of official countries. In the end, the list looks very similar to [this one](https://peakbagger.com/pbgeog/countries.aspx). The general principles in building the list that is used in this repository are as follows:
* Start with sovereign nations (countries): the definition of a country is not free of controversy and ranges from regions as large and populous as China to as tiny as the Vatican City, but it maps well to how people divide the world, has good-quality data, and results in ~200 entities, which is a reasonable number. Continents are far too few and diverse; states or other second-order administrative units are far too numerous and noisier from a data perspective.
* Build from the data: these regions will be applied to pageviews and articles, so there must be a clear mapping between the regions used here, Wikidata (initial groundtruth for articles), and [MaxMind countries](https://dev.maxmind.com/geoip/legacy/codes/iso3166/) (how countries are assigned to readers). Much of the initial decisions are based on information from [en:List of sovereign states](https://en.wikipedia.org/wiki/List_of_sovereign_states) as well.
* Be as inclusive as reasonable: when there is a region that is disputed and easily divided, go with the separate regions and allow the end-user to determine how they handle aggregation.
* All [UN member states](https://en.wikipedia.org/wiki/Member_states_of_the_United_Nations) are included.
* All regions that are recognized as sovereign nations by at least one other UN member state are included. For instance, [South Ossetia](https://en.wikipedia.org/wiki/South_Ossetia) is recognized by Russia and several other countries, so it is included. [Artsakh](https://en.wikipedia.org/wiki/Republic_of_Artsakh) is not recognized by any UN member states and so is not included.

Additionally, overseas territories are collected separately as opposed to being merged in with the main:
* [British territories](https://en.wikipedia.org/wiki/British_Overseas_Territories) and [dependencies](https://en.wikipedia.org/wiki/Crown_dependencies)
    * NOTE: members of the [Commonwealth realm](https://en.wikipedia.org/wiki/Commonwealth_realm) are also treated independently.
* [Kingdom of the Netherlands](https://en.wikipedia.org/wiki/Kingdom_of_the_Netherlands)
* [Overseas France](https://en.wikipedia.org/wiki/Overseas_France)
* [Realm of New Zealand](https://en.wikipedia.org/wiki/Realm_of_New_Zealand)
* [Australian states/territories](https://en.wikipedia.org/wiki/States_and_territories_of_Australia)
* [US territories](https://en.wikipedia.org/wiki/Territories_of_the_United_States)
* [Åland Islands (Finland)](https://en.wikipedia.org/wiki/%C3%85land_Islands)
* [Macau / Hong Kong (China)](https://en.wikipedia.org/wiki/Special_administrative_regions_of_China)	

There are a few adjustments / exceptions:
* [Antartica](https://en.wikipedia.org/wiki/Antarctica) is treated as a single entity.
* Both the [United Kingdom](https://en.wikipedia.org/wiki/United_Kingdom) and its separate countries (England, Scotland, Wales, and Northern Ireland) are collected despite being overlapping entities.
* [Sahrawi Arab Democratic Republic](https://en.wikipedia.org/wiki/List_of_sovereign_states#SADR) is included as Western Sahara.
* [Svalbard and Jan Mayen](https://en.wikipedia.org/wiki/Svalbard_and_Jan_Mayen) are included as part of Norway.
* [Sixth Republic of South Korea](https://en.wikipedia.org/wiki/Sixth_Republic_of_South_Korea) and [French Fifth Republic](https://en.wikipedia.org/wiki/French_Fifth_Republic) are included as South Korea and France, respectively.

### Adding a new region
Regions can be easily added given the following:
* The region is a country or country-like equivalent
* Wikidata item(s) associated with the region and evidence that the items are linked to by other items
* If the change is potentially controversial, a documented public proposal / discussion -- e.g., phabricator, meta, etc.

### Gathering data
The regions in this list were compiled based off of a variety of data sources. A few of them are below:

* [Map units GeoJSON](https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-0-details/) from Natural Earth
```
# SPARQL (https://query.wikidata.org/) query for list of present-day sovereign states
SELECT DISTINCT
  ?country ?countryLabel
WHERE
{
  # instance-of sovereign state
  ?country wdt:P31 wd:Q3624078 .
  #not a former country
  FILTER NOT EXISTS {?country wdt:P31 wd:Q3024240}
  #and no an ancient civilisation (needed to exclude ancient Egypt)
  FILTER NOT EXISTS {?country wdt:P31 wd:Q28171280}
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
}
ORDER BY ?countryLabel
```

```
# Countries with ISO-3166 codes (https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes)
SELECT
  ?item ?itemLabel ?value
WHERE 
{
  ?item wdt:P297 ?value
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
}
ORDER BY ?value
```

### Does this seem to work?
Based on a run of the code from 30 November 2020, these are the top-10 Wikidata items (and how many times they appear) that are used as values for country-level properties but not included in our dataset. All of them are within-country regions or former states:
* Soviet Union (Q15180): 118731 times
* United Kingdom of Great Britain and Ireland 1801-1927 (Q174193): 60350 times
* Ontario (Q1904): 48513 times
* Russian Empire 1721-1917 (Q34266): 45080 times
* Kingdom of Italy 1861-1946 (Q172579): 43070 times
* British Columbia (Q1974): 36433 times
* Paris (Q90): 26707 times
* Quebec (Q176): 26637 times
* New South Wales (Q3224): 24037 times
* Czechoslovakia (Q33946): 23930 times

In general, that run resulted in the following:
* 90,673,426 Wikidata items processed
* 25,781,724 (28.4%) items associated with Wikipedia articles
* 11,065,561 (42.9% of Wikipedia-related items) evaluated because they had associated coordinates or a region found via properties
    * 7,783,576 (70.3%) had a `country` value from the keep-list of regions
    * 5,666,563 (51.2% of the kept items) had coordinates, of which 184,131 (3.2%) were mapped to regions.
        * Point-in-polygon skipped for the vast majority of the rest because other properties had already identified countries. This is great because it makes this process much faster and I verified from a previous run that the coordinates don't add any additional information for >97% of items.
    * 2,552,380 (23.1%) had a `country of citizenship` value from the keep-list of regions
    * 611,940 (5.5%) had a `country of origin` value from the keep-list of regions
    * 126,473 (1.1%) had a `located in the administrative territorial entity` value from the keep-list of regions
    * 117,513 (1.1%) had a `country for sport` value from the keep-list of regions
    * 75,964 (0.7%) had a `place-of-birth` value from the keep-list of regions (many place-of-birth properties are more specific than country)
    * 17,482 (0.2%) had a `facet of` value from the keep-list of regions
    * 627 (0.0%) had a `part of` value from the keep-list of regions
    * 188 (0.0%) had a `located in present-day administrative territorial entity` value from the keep-list of regions
* 254,895 (2.3%) items were associated with multiple regions
* 20,890 (0.2%) items had coordinates that weren't mapped to a region and no other region properties. These misses can be spatial data issues (Antarctica), non-Earth coordinates (e.g., craters on Mars), or outside of borders (ocean features).
* 5,302,583 (47.9%) of items had a single property (or coordinates) that determined its region -- i.e. no data if that property / coordinates weren't included. That property was:
    * `country of citizenship`: 2,387,489 (45.0% of the time)
    * `country`: 2,186,663 (41.2% of the time)
    * `country of origin`: 591,217 (11.1% of the time)
    * `coordinates`: 99,881 (1.9% of the time)
    * `place of birth`: 18,577 (0.4% of the time)
    * `facet of`: 12,614 (0.2% of the time)
    * `country for sport`: 5,413 (0.1% of the time)
    * `located in the administrative territorial entity`: 613 (0.0% of the time)
    * `located in present-day administrative territorial entity`: 92 (0.0% of the time)
    * `part of`: 24 (0.0% of the time)
    * Together, this analysis suggests that `part of`, `located in present-day administrative territorial entity`, `located in the administrative territorial entity` at a minimum could be dropped with almost no effect.