import os
import pandas as pd


"""
First I create two lists of petIDs: one for the cute_dogs, i.e those that were adopted within 60 days (i.e. 0)
and another list not_so_cute_dogs who were not adopted after 60 days (i.e. 1)
"""
labels = pd.read_csv("/Users/viktoria/Downloads/petfinder-adoption-prediction/train.csv")
labels.set_index('PetID')
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
labels['AdoptionSpeed'] = labels.apply(f, axis=1)
labels_only = labels['AdoptionSpeed']

cute_dogs = labels[labels['AdoptionSpeed'] ==0]
cute_dogs.set_index('PetID', inplace=True)
cute_list_id = list(cute_dogs.index)

not_so_cute_dogs = labels[labels['AdoptionSpeed'] ==1]
not_so_cute_dogs.set_index('PetID', inplace=True)
not_so_cute_list_id = list(not_so_cute_dogs.index)

# path to image folder, get all filenames on this folder
# and store it in the onlyfiles list
mypath = "/Users/viktoria/Downloads/petfinder-adoption-prediction/train_images"
onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]

# your list of dogID's
lis1 = cute_list_id
lis2 = not_so_cute_list_id

# create two seperate lists from onlyfiles list based on lis1 and lis2
lis1files = [i for i in onlyfiles for j in lis1 if j in i]
lis2files = [i for i in onlyfiles for j in lis2 if j in i]

# create two sub folders in mypath folder
subfolder1 = os.path.join(mypath, "cute")
subfolder2 = os.path.join(mypath, "not_so_cute")

# check if they already exits to prevent error
if not os.path.exists(subfolder1):
    os.makedirs(subfolder1)

if not os.path.exists(subfolder2):
    os.makedirs(subfolder2)

# move files to their respective sub folders
for i in lis1files:
    source = os.path.join(mypath, i)
    destination = os.path.join(subfolder1, i)
    os.rename(source, destination)

for i in lis2files:
    source = os.path.join(mypath, i)
    destination = os.path.join(subfolder2, i)
    os.rename(source, destination)

