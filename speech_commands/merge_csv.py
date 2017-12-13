import csv

adv_file = open('adv.csv', 'r')
filenames_file = open('baseline_submission.csv', 'r')
file2 = open('merged.csv', 'w')
adv = csv.reader(adv_file)
fn = csv.reader(filenames_file)
writer = csv.writer(file2)

labels = []

for row in adv:
    # writer.writerow(fn[id][0], row[1], 'Somevalue')
    labels.append(row[1])

fnames = []

for row in fn:
    fnames.append(row[0])

for i, name in enumerate(fnames):
    writer.writerow((name, labels[i]))