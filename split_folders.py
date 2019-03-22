from sklearn.model_selection import train_test_split
import pandas as pd
import os

"""
Now I'll create lists of train and test id folders
"""

df_binary = pd.read_csv("/Users/viktoria/Downloads/petfinder-adoption-prediction/train.csv")

def f(row):
    if row['AdoptionSpeed'] == 0:
        val = 0
    elif row['AdoptionSpeed'] == 1:
        val = 0
    elif row['AdoptionSpeed'] == 2:
        val = 0
    elif row['AdoptionSpeed'] == 3:
        val = 1
    elif row['AdoptionSpeed'] == 4:
        val = 1
    return val

df_binary["AdoptionSpeed"] = df_binary.apply(f, axis=1)

df_binary.set_index("PetID", inplace=True)

X = df_binary[['Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize', 'FurLength',
            'Vaccinated', 'Dewormed', 'Sterilized', 'Quantity', 'Fee', 'PhotoAmt']]
y = df_binary['AdoptionSpeed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=21)

train_id_list = list(X_train.index)
test_id_list = list(X_test.index)

# """
# sort images into train and test folders
# """
#
# # path to image folder, get all filenames on this folder
# # and store it in the onlyfiles list
# mypath = "/Users/viktoria/Downloads/petfinder-adoption-prediction/train_images"
# onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
#
# # your list of dogID's
# lis1 = train_id_list
# lis2 = test_id_list
#
# # create two seperate lists from onlyfiles list based on lis1 and lis2
# lis1files = [i for i in onlyfiles for j in lis1 if j in i]
# lis2files = [i for i in onlyfiles for j in lis2 if j in i]
#
# # create two sub folders in mypath folder
# subfolder1 = os.path.join(mypath, "train")
# subfolder2 = os.path.join(mypath, "test")
#
# # check if they already exits to prevent error
# if not os.path.exists(subfolder1):
#     os.makedirs(subfolder1)
#
# if not os.path.exists(subfolder2):
#     os.makedirs(subfolder2)
#
# # move files to their respective sub folders
# for i in lis1files:
#     source = os.path.join(mypath, i)
#     destination = os.path.join(subfolder1, i)
#     os.rename(source, destination)
#
# for i in lis2files:
#     source = os.path.join(mypath, i)
#     destination = os.path.join(subfolder2, i)
#     os.rename(source, destination)

"""
create lists of cute and not_so_cute IDs in train
"""
y_train_df = y_train.to_frame()

cute_dogs = y_train_df[y_train_df['AdoptionSpeed'] ==0]
cute_list_id_train = list(cute_dogs.index)

not_so_cute_dogs = y_train_df[y_train_df['AdoptionSpeed'] ==1]
not_so_cute_list_id_train = list(not_so_cute_dogs.index)


"""
sort images in train folder into cute and not_so_cute
"""

mypath_c = "/Users/viktoria/Downloads/petfinder-adoption-prediction/train_images/train"
onlyfiles = [f for f in os.listdir(mypath_c) if os.path.isfile(os.path.join(mypath_c, f))]

# your list of dogID's
lis1 = cute_list_id_train
lis2 = not_so_cute_list_id_train

# create two seperate lists from onlyfiles list based on lis1 and lis2
lis1files = [i for i in onlyfiles for j in lis1 if j in i]
lis2files = [i for i in onlyfiles for j in lis2 if j in i]

# create two sub folders in mypath folder
subfolder1 = os.path.join(mypath_c, "cute")
subfolder2 = os.path.join(mypath_c, "not_so_cute")

# check if they already exits to prevent error
if not os.path.exists(subfolder1):
    os.makedirs(subfolder1)

if not os.path.exists(subfolder2):
    os.makedirs(subfolder2)

# move files to their respective sub folders
for i in lis1files:
    source = os.path.join(mypath_c, i)
    destination = os.path.join(subfolder1, i)
    os.rename(source, destination)

for i in lis2files:
    source = os.path.join(mypath_c, i)
    destination = os.path.join(subfolder2, i)
    os.rename(source, destination)


"""
create lists of cute and not_so_cute IDs in test
"""
y_test_df = y_test.to_frame()

cute_dogs_test = y_test_df[y_test_df['AdoptionSpeed'] ==0]
cute_list_id_test = list(cute_dogs_test.index)

not_so_cute_dogs_test = y_test_df[y_test_df['AdoptionSpeed'] ==1]
not_so_cute_list_id_test = list(not_so_cute_dogs_test.index)


"""
sort images in test folder into cute and not_so_cute
"""

mypath_c = "/Users/viktoria/Downloads/petfinder-adoption-prediction/train_images/test"
onlyfiles = [f for f in os.listdir(mypath_c) if os.path.isfile(os.path.join(mypath_c, f))]

# your list of dogID's
lis1 = cute_dogs_test
lis2 = not_so_cute_list_id_test

# create two seperate lists from onlyfiles list based on lis1 and lis2
lis1files = [i for i in onlyfiles for j in lis1 if j in i]
lis2files = [i for i in onlyfiles for j in lis2 if j in i]

# create two sub folders in mypath folder
subfolder1 = os.path.join(mypath_c, "cute")
subfolder2 = os.path.join(mypath_c, "not_so_cute")

# check if they already exits to prevent error
if not os.path.exists(subfolder1):
    os.makedirs(subfolder1)

if not os.path.exists(subfolder2):
    os.makedirs(subfolder2)

# move files to their respective sub folders
for i in lis1files:
    source = os.path.join(mypath_c, i)
    destination = os.path.join(subfolder1, i)
    os.rename(source, destination)

for i in lis2files:
    source = os.path.join(mypath_c, i)
    destination = os.path.join(subfolder2, i)
    os.rename(source, destination)