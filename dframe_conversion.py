
import pandas as pd
import json
import requests



if True:

    datasets_path = '/Users/juanerolon/Dropbox/_github/gits/elastic-playground/datasets/'

    df1 = pd.read_csv(datasets_path + 'Alerts_Severity_1.csv', header='infer')
    print("Number of records in df1: {}".format(df1['@timestamp'].count()))
    df2 = pd.read_csv(datasets_path + 'Alerts_Severity_1B.csv', header='infer')
    print("Number of records in df2: {}".format(df2['@timestamp'].count()))
    df3 = pd.read_csv(datasets_path + 'Alerts_Severity_2.csv', header='infer')
    print("Number of records in df3: {}".format(df3['@timestamp'].count()))
    df4 = pd.read_csv(datasets_path + 'Alerts_Severity_2B.csv', header='infer')
    print("Number of records in df4: {}".format(df4['@timestamp'].count()))
    df5 = pd.read_csv(datasets_path + 'Alerts_Severity_3.csv', header='infer')
    print("Number of records in df5: {}".format(df5['@timestamp'].count()))


    joined_df = pd.concat([df1, df2], axis=0)
    joined_df = pd.concat([joined_df, df3], axis=0)
    joined_df = pd.concat([joined_df, df4], axis=0)
    joined_df = pd.concat([joined_df, df5], axis=0)

    joined_df['_id'] = joined_df['column_1'] + '_' + joined_df['column_2']  # or whatever makes your _id
    df_as_json = joined_df.to_json(orient='records', lines=True)

    final_json_string = ''
    for json_document in df_as_json.split('\n'):
        jdict = json.loads(json_document)
        metadata = json.dumps({'index': {'_id': jdict['_id']}})
        jdict.pop('_id')
        final_json_string += metadata + '\n' + json.dumps(jdict) + '\n'

    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    r = requests.post('http://elasticsearch.host:9200/my_index/my_type/_bulk', data=final_json_string, headers=headers,
                      timeout=60)





#Export concatenated csv datasets into single json dataset

if False:

    datasets_path = '/Users/juanerolon/Dropbox/_github/gits/elastic-playground/datasets/'

    df1 = pd.read_csv(datasets_path + 'Alerts_Severity_1.csv', header='infer')
    print("Number of records in df1: {}".format(df1['@timestamp'].count()))
    df2 = pd.read_csv(datasets_path + 'Alerts_Severity_1B.csv', header='infer')
    print("Number of records in df2: {}".format(df2['@timestamp'].count()))
    df3 = pd.read_csv(datasets_path + 'Alerts_Severity_2.csv', header='infer')
    print("Number of records in df3: {}".format(df3['@timestamp'].count()))
    df4 = pd.read_csv(datasets_path + 'Alerts_Severity_2B.csv', header='infer')
    print("Number of records in df4: {}".format(df4['@timestamp'].count()))
    df5 = pd.read_csv(datasets_path + 'Alerts_Severity_3.csv', header='infer')
    print("Number of records in df5: {}".format(df5['@timestamp'].count()))


    joined_df = pd.concat([df1, df2], axis=0)
    joined_df = pd.concat([joined_df, df3], axis=0)
    joined_df = pd.concat([joined_df, df4], axis=0)
    joined_df = pd.concat([joined_df, df5], axis=0)

    for col in joined_df.columns:
        print(col)


    print("Total Number of records: {}".format(joined_df['@timestamp'].count()))


    if False:
        joined_df.to_json('alerts.json', orient='records', lines=True)




