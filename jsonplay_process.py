
# jsonplay.py
# Juan E. Rolon
# @juanerolon
#
### Converts inline json data to a pandas dataframe


import pandas as pd
import json

#Inline json data: example 1
#The right hand side of assingment statement below is the content of an actual json file

data3 = [{
  "_index": "suricata-000568",
  "_type": "doc",
  "_id": "iwudqWQBPs4-MNMx4s59",
  "_version": 1,
  "_score": 1,
  "_source": {
    "vlan": 36,
    "node_ip": "10.10.4.141",
    "dest_port": 53,
    "in_iface": "enp6s0",
    "src_ip": "10.10.4.141",
    "event_type": "dns",
    "timestamp": "2018-07-17T19:00:52.922136+0000",
    "tags": [
      "local_src",
      "local_remote",
      "remote_service",
      "src_node_name"
    ],
    "dns": {
      "tx_id": 0,
      "id": 33846,
      "type": "query",
      "rrtype": "A",
      "rrname": "eum.concursolutions.com"
    },
    "sensor_name": "ege-sensor001",
    "proto": "UDP",
    "local_ip": "10.10.4.141",
    "client_name": "ege",
    "@version": "1",
    "@timestamp": "2018-07-17T19:00:52.922Z",
    "node_composite": "10.10.4.141+ege",
    "geoip": {
      "latitude": 37.7697,
      "postal_code": "94107",
      "country_code2": "US",
      "continent_code": "NA",
      "city_name": "San Francisco",
      "region_code": "CA",
      "country_code3": "US",
      "region_name": "California",
      "ip": "208.67.222.222",
      "country_name": "United States",
      "timezone": "America/Los_Angeles",
      "location": {
        "lon": -122.3933,
        "lat": 37.7697
      },
      "longitude": -122.3933,
      "dma_code": 807
    },
    "src_hostname": "10.10.4.141",
    "path": "/nsm/suricata/eve.json",
    "src_port": 61719,
    "consumer": "logstash-test",
    "type": "doc",
    "dest_ip": "208.67.222.222",
    "service_port": 53,
    "host": "ege-sensor001",
    "flow_id": 1103637965443608,
    "log_source": "suricata"
  },
  "fields": {
    "@timestamp": [
      "2018-07-17T19:00:52.922Z"
    ],
    "timestamp": [
      "2018-07-17T19:00:52.922Z"
    ]
  },
  "sort": [
    1531854052922
  ]
}]


json_string3 = json.dumps(data3)


df3 = pd.read_json(json_string3, orient='records')

print("")
print("")

print("Dataframe for json data3\n")
print(df3)