import pandas as pd

df = pd.read_parquet("data/tmdb_with_review_tags.parquet")

print("=== Shape ===")
print(df.shape)

print("\n=== Columns (short) ===")
print(df.columns.tolist())

print("\n=== Sample review_tags values (first 10 non-null) ===")
sample = df["review_tags"].dropna().head(10)
for i, val in enumerate(sample):
    print(f"{i}: {repr(val)}  |  type = {type(val)}")

print("\n=== Value counts for types in review_tags ===")
type_counts = df["review_tags"].dropna().apply(lambda x: type(x).__name__).value_counts()
print(type_counts.head(10))

