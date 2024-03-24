import requests
import pandas as pd


def get_data(id: str) -> pd.DataFrame:
    url = f"https://data.gov.sg/api/action/datastore_search?resource_id={id}"
    response = requests.get(url)
    data = response.json()
    print(f"Found {len(data['result']['records'])} records")
    
    df = pd.DataFrame(data['result']['records'])
    df.columns = [i['id'] for i in data['result']['fields']]
    df.columns = [df.columns[-1]] + list(df.columns[:-1])
    return df

year_to_resource = {"1990-1999": "d_ebc5ab87086db484f88045b47411ebc5",
                    "2000-2012": "d_43f493c6c50d54243cc1eab0df142d6a",
                    "2012-2014": "d_2d5ff9ea31397b66239f245f57751537",
                    "2015-2016": "d_2d5ff9ea31397b66239f245f57751537",
                    "2017 Onwards": "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"}

df = get_data(year_to_resource["2017 Onwards"])

print(df.head())