import pandas as pd

true_df = pd.read_csv("data/news_data/True.csv")
fake_df = pd.read_csv("data/news_data/Fake.csv")

true_df["label"] = 1
fake_df["label"] = 0

df = pd.concat([true_df, fake_df], ignore_index=True)

df.to_csv("data/news.csv", index=False)

print("news.csv created successfully")