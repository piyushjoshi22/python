import pandas as pd
import time
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Load dataset
file_path = r"C:\Users\piyush joshi\OneDrive\Desktop\groceries.csv"

# Read CSV & handle missing values
try:
    df = pd.read_csv(file_path, header=None, encoding='utf-8', on_bad_lines="skip")
except FileNotFoundError:
    print(f"❌ Error: File not found at {file_path}")
    exit()
except pd.errors.ParserError as e:
    print(f"❌ Parser Error: {e}")
    exit()

# Strategy: Replace missing values with 'Unknown'
df.fillna("Unknown", inplace=True)

# Convert transactions into list format
transactions = df.apply(lambda x: x.tolist(), axis=1).tolist()

# Convert transactions to one-hot encoding
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# **Apriori Algorithm**
start_time = time.time()
frequent_itemsets_apriori = apriori(df_encoded, min_support=0.01, use_colnames=True)
apriori_time = time.time() - start_time

# **FP-Growth Algorithm**
start_time = time.time()
frequent_itemsets_fpgrowth = fpgrowth(df_encoded, min_support=0.01, use_colnames=True)
fpgrowth_time = time.time() - start_time

# Generate Association Rules
rules_apriori = association_rules(frequent_itemsets_apriori, metric="confidence", min_threshold=0.1)
rules_fpgrowth = association_rules(frequent_itemsets_fpgrowth, metric="confidence", min_threshold=0.1)

# **Print Results**
print("\U0001F535 Apriori Frequent Itemsets:\n", frequent_itemsets_apriori.head())
print("\U0001F535 Apriori Association Rules:\n", rules_apriori.head())
print("\n\U0001F535 FP-Growth Frequent Itemsets:\n", frequent_itemsets_fpgrowth.head())
print("\U0001F535 FP-Growth Association Rules:\n", rules_fpgrowth.head())

# **Compare Performance**
print("\n\U0001F680 Execution Time:")
print(f"⏳ Apriori: {apriori_time:.4f} seconds")
print(f"⚡ FP-Growth: {fpgrowth_time:.4f} seconds")
