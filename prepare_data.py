import random
import re

from datasets import Dataset, DatasetDict, load_dataset
import re
from collections import defaultdict
from collections import Counter


def select_subset(dataset):
    #def count_sentences(text):
    #    return len(re.findall(r'\.\s+[A-Z]', text))
    
    #filtered_dataset = dataset.filter(lambda x: count_sentences(x["text"]) >= 9)
    
    url_counts = Counter(dataset["url"])

    filtered_dataset = dataset.filter(lambda x: url_counts[x["url"]] > 8)

    #if len(filtered_dataset) < n:
    #    raise ValueError("Not enough entries with at least 9 sentences")
    
    #selected_samples = filtered_dataset.select(random.sample(range(len(filtered_dataset)), n))
    
    #return selected_samples

    return filtered_dataset

# Function to extract article ID (everything before the last underscore)
def get_article_id(paragraph_id):
    return re.match(r"(.+?)_\d+$", paragraph_id).group(1)


# Load your dataset (replace with the actual path or dataset name)
#dataset_dic = Dataset.from_dict({
#    'id': ["20231101.da_508659_0", "20231101.da_508659_1", "20231101.da_508659_2", "20231101.da_508660_0"],
#    'text': ["First paragraph of article 508659.", "Second paragraph of article 508659.", 
#             "Third paragraph of article 508659.", "First paragraph of article 508660."]
#})

dataset_path = "rasdani/cohere-wikipedia-2023-11-da"

dataset = load_dataset(dataset_path, split = "train")

# select only those with 9 or more paragraphs
data_subset = select_subset(dataset)


######### Get middle paragraph and surrounding paragraphs from 9 or more consecutive paragraphs
import pandas as pd
# Initialize lists to store results

middle_cells = []
surrounding_cells = []

# Group by 'url'
df = pd.DataFrame(data_subset)
df = df.groupby('url')

for url, group in df:
    # Ensure there are at least 9 rows for the url
    if len(group) >= 9:
        # Randomly choose a starting index for a window of 9
        start_idx = random.randint(0, len(group) - 9)
        
        # Select the window of 9 rows
        window = group.iloc[start_idx:start_idx + 9]
        
        # Get the middle cell (5th in the list, index 4)
        middle_cell = window.iloc[4]['text']  # Use .iloc[] to access by position
        middle_cells.append(middle_cell)
        
        # Get the surrounding cells (8 cells, excluding the middle)
        surrounding = window['text'].iloc[:4].tolist() + window['text'].iloc[5:].tolist()
        surrounding_cells.append(surrounding)# Now, middle_cells and surrounding_cells contain your desired outputs
print(middle_cells)
print(surrounding_cells)

len(middle_cells)

len(surrounding_cells)
middle_cells[0]
surrounding_cells[0]

#TODO: Remove the entries in middle_cells from surrounding_cells









################### GRAVEYARD
print(data_subset[0])

dataset = dataset.to_pandas()

num_articles = len(dataset["url"].unique())

print(dataset[1])

######## Put all paragraphs from an article in the same row

# Convert Dataset to list of dictionaries and group by article_id
dataset = dataset.to_dict()
grouped_data = defaultdict(list)
for i in range(len(dataset["_id"])):
    article_id = get_article_id(dataset["_id"][i])
    grouped_data[article_id].append(dataset["text"][i])

# Concatenate paragraphs for each article and create a new dataset
grouped_dataset = Dataset.from_dict({
    "_id": list(grouped_data.keys()),
    "text": [" ".join(paragraphs) for paragraphs in grouped_data.values()]
})

print(grouped_dataset)

assert len(grouped_dataset) == num_articles, "n rows in grouped data does not match number of unique urls in original data"

grouped_dataset[0]

grouped_dataset[grouped_dataset["_id"] == "20231101.da_508659"]


grouped_data["20231101.da_508659"]