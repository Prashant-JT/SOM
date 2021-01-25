import pandas as pd


def change_date(df):
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %I:%M:%S %p')
    df['HOUR'] = df['Date'].dt.hour
    df['DAY'] = df['Date'].dt.day
    df['MONTH'] = df['Date'].dt.month


def process_data(path):
    df = pd.read_csv(path)
    df = df.dropna()
    change_date(df)
    df = df.drop(columns=['ID', 'Case Number'])  # Ambos cotienen datos únicos*
    df = df.drop(columns=['Year', 'Updated On', 'Location', 'Date'])  # Datos irrrelevantes

    values_dict = dict()
    for col in df.columns:
        if isinstance(df[col][0], str):
            aux = df[col].unique()
            d = dict(enumerate(aux.flatten(), 1))
            d = dict((V, K) for K, V in d.items())
            values_dict[col] = d
            df[col] = df[col].map(d)

    return df.astype(dtype='float32'), values_dict
