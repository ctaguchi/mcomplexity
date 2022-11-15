import pandas as pd
import sys

def calculate_pearson(data):
    df = pd.read_table(data, index_col="treebank ")
    df = drop_sd(df)
    df = df.T
    pearson_matrix = df.corr(method="pearson")
    return pearson_matrix

def drop_sd(df: pd.core.frame.DataFrame):
    dropped_cols = ["ttr_sd", "msp_sd", "ws_sd", "wh_sd", "lh_sd", "is_sd", "mfh_sd"]
    for c in dropped_cols:
        df = df.drop(c, axis=1)
    dropped_langs = ["Korean-GSD", "Urdu-UDTB", "Hindi-HDTB", "Persian-Seraji", "Chinese-GSD",
                    "Arabic-PADT", "Hebrew-HTB"]
    for l in dropped_langs:
        df = df.drop(l, axis=0)
    return df

def get_max_id(df: pd.core.frame.DataFrame, col: str):
    df = df.loc[col]
    ranking = df.sort_values(ascending=False)
    ranking = ranking.reset_index()
    return ranking[:20]

def highest_corr_ja(pearson_matrix: pd.core.frame.DataFrame):
    ja = ["Japanese-GSDOrig", "Japanese-GSDOrig1", "Japanese-GSDOrig2", 
        "Japanese-GSDLUWOrig", "Japanese-GSDLUWOrig1", "Japanese-GSDLUWOrig2"]
    for j in ja:
        top = get_max_id(pearson_matrix, j)
        print("{}: \n{}\n".format(j, top))

if __name__ == "__main__":
    data = sys.argv[1]
    output = sys.argv[2]
    pearson_matrix = calculate_pearson(data)
    pearson_matrix.to_csv(output, sep="\t")
    highest_corr_ja(pearson_matrix)