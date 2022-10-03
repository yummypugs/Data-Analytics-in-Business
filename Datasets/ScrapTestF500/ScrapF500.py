import os
import time
import sys
import numpy as np
import pandas as pd
import regex as re
import lxml
import numbers
from bs4 import BeautifulSoup
import requests
import json

req_headers = {
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'accept-encoding': 'gzip, deflate, br',
    'accept-language': 'en-US,en;q=0.8',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'
}

urlDict = [
    {"year": 2022, "url": "https://fortune.com/wp-json/irving/v1/data/franchise-search-results?list_id=3287962&token=Zm9ydHVuZTpCcHNyZmtNZCN5SndjWkkhNHFqMndEOTM="},
    {"year": 2021, "url": "https://fortune.com/wp-json/irving/v1/data/franchise-search-results?list_id=3040727&token=Zm9ydHVuZTpCcHNyZmtNZCN5SndjWkkhNHFqMndEOTM="},
    {"year": 2020, "url": "https://fortune.com/wp-json/irving/v1/data/franchise-search-results?list_id=2814606&token=Zm9ydHVuZTpCcHNyZmtNZCN5SndjWkkhNHFqMndEOTM="},
    {"year": 2019, "url": "https://fortune.com/wp-json/irving/v1/data/franchise-search-results?list_id=2611932&token=Zm9ydHVuZTpCcHNyZmtNZCN5SndjWkkhNHFqMndEOTM="},
    {"year": 2018, "url": "https://fortune.com/wp-json/irving/v1/data/franchise-search-results?list_id=2358051&token=Zm9ydHVuZTpCcHNyZmtNZCN5SndjWkkhNHFqMndEOTM="},
    {"year": 2017, "url": "https://fortune.com/wp-json/irving/v1/data/franchise-search-results?list_id=2013055&token=Zm9ydHVuZTpCcHNyZmtNZCN5SndjWkkhNHFqMndEOTM="},
    {"year": 2016, "url": "https://fortune.com/wp-json/irving/v1/data/franchise-search-results?list_id=1666518&token=Zm9ydHVuZTpCcHNyZmtNZCN5SndjWkkhNHFqMndEOTM="},
    {"year": 2015, "url": "https://fortune.com/wp-json/irving/v1/data/franchise-search-results?list_id=1141696&token=Zm9ydHVuZTpCcHNyZmtNZCN5SndjWkkhNHFqMndEOTM="},
    {"year": 2014, "url": "https://fortune.com/wp-json/irving/v1/data/franchise-search-results?list_id=500629&token=Zm9ydHVuZTpCcHNyZmtNZCN5SndjWkkhNHFqMndEOTM="},
    {"year": 2013, "url": "https://fortune.com/wp-json/irving/v1/data/franchise-search-results?list_id=391746&token=Zm9ydHVuZTpCcHNyZmtNZCN5SndjWkkhNHFqMndEOTM="},
    {"year": 2012, "url": "https://fortune.com/wp-json/irving/v1/data/franchise-search-results?list_id=345184&token=Zm9ydHVuZTpCcHNyZmtNZCN5SndjWkkhNHFqMndEOTM="},
    {"year": 2011, "url": "https://fortune.com/wp-json/irving/v1/data/franchise-search-results?list_id=346195&token=Zm9ydHVuZTpCcHNyZmtNZCN5SndjWkkhNHFqMndEOTM="},
    {"year": 2010, "url": "https://fortune.com/wp-json/irving/v1/data/franchise-search-results?list_id=347212&token=Zm9ydHVuZTpCcHNyZmtNZCN5SndjWkkhNHFqMndEOTM="},
    {"year": 2009, "url": "https://fortune.com/wp-json/irving/v1/data/franchise-search-results?list_id=348231&token=Zm9ydHVuZTpCcHNyZmtNZCN5SndjWkkhNHFqMndEOTM="},
    {"year": 2008, "url": "https://fortune.com/wp-json/irving/v1/data/franchise-search-results?list_id=352816&token=Zm9ydHVuZTpCcHNyZmtNZCN5SndjWkkhNHFqMndEOTM="},
    {"year": 2007, "url": "https://fortune.com/wp-json/irving/v1/data/franchise-search-results?list_id=353857&token=Zm9ydHVuZTpCcHNyZmtNZCN5SndjWkkhNHFqMndEOTM="}
]


def ScrapF500(data_dic):
    response = requests.get(data_dic['url'])
    response.raise_for_status()  # raise exception if invalid response
    comps = response.json()
    if len(comps) > 2:
        print("")
        return
    comp = comps[1]
    format_json = json.dumps(comp, sort_keys=True, indent=5)

    fields = []
    # each 'items' contains a 'fields' and 'permalink'. 'fields' contains the data
    for j in comp['items']:
        # for key, value in comp:
        field = j['fields']
        print(field)
        fields.append(field)

    output_df = pd.DataFrame(fields)
    print(f"Year: {data_dic['year']}, Dimensions: {output_df.shape}")
    output_df.to_csv(f"F1000_{data_dic['year']}.csv", index=False)


if __name__ == '__main__':

    for val in urlDict:
        ScrapF500(val)
