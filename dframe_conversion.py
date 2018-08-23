
import pandas as pd


#Export concatenated csv datasets into single json dataset

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

    for col in joined_df.columns:
        print(col)


    print("Total Number of records: {}".format(joined_df['@timestamp'].count()))


    if False:
        joined_df.to_json('alerts.json', orient='records', lines=True)




