import os

# Your exact folder paths
train_dir = r"D:\Threat_Detection_Workspace\01_Curated_Dataset\Final_Processed_Data_Granular\train"
test_dir = r"D:\Threat_Detection_Workspace\01_Curated_Dataset\Final_Processed_Data_Granular\test"

def count_items_in_directory(directory):
    class_counts = {}
    if os.path.exists(directory):
        for class_name in os.listdir(directory):
            class_path = os.path.join(directory, class_name)
            if os.path.isdir(class_path):
                # Counts the number of video files (or subfolders) inside the class folder
                num_items = len(os.listdir(class_path))
                class_counts[class_name] = num_items
    else:
        print(f"WARNING: Folder not found -> {directory}")
    return class_counts

# Scan both folders
train_counts = count_items_in_directory(train_dir)
test_counts = count_items_in_directory(test_dir)

# Get a unique list of all classes across both folders
all_classes = set(list(train_counts.keys()) + list(test_counts.keys()))

# Print the results as a formatted table
print(f"{'Class Name':<25} | {'Train Videos':<15} | {'Test Videos':<15} | {'Total':<10}")
print("-" * 75)

total_train = 0
total_test = 0

for cls in sorted(all_classes):
    trn = train_counts.get(cls, 0)
    tst = test_counts.get(cls, 0)
    tot = trn + tst
    
    total_train += trn
    total_test += tst
    
    print(f"{cls:<25} | {trn:<15} | {tst:<15} | {tot:<10}")

print("-" * 75)
print(f"{'GRAND TOTAL':<25} | {total_train:<15} | {total_test:<15} | {total_train + total_test:<10}")