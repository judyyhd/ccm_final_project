import pandas as pd
df = pd.read_csv("../data/trialdf.csv")
df["agent_color"] = df["subjid"].str.extract(r"(red|purple)$")
df = df[df["agent_color"] == "red"]
df = df[df["gameover"] == False]
print("rows:", len(df))
print("mean helping_event:", df["helping_event"].mean())