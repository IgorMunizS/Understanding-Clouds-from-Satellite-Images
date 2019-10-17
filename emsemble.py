import numpy as np
import pandas as pd
import os


# def emsemble():
list_submissions = []
for i,file in enumerate(os.listdir('../best_submissions/')):
    sub = pd.read_csv('../best_submissions/' + file)
    sub.columns = ['Image_Label', 'enc' + str(i+1)]
    list_submissions.append(sub)

final_sub = pd.merge(left = list_submissions[0], right = list_submissions[1], on = 'Image_Label', how = 'inner')
# identify the positions where xf1 has empty predictions but xf2 does not
final_sub[final_sub['enc1'] != final_sub['enc2']]
id1 = np.where(final_sub['enc1'] == '-1')[0]
id2 = np.where(final_sub['enc2'] != '-1')[0]
idx = np.intersect1d(id1, id2)

# map non-empty xf2 slots to empty ones in xf1
final_sub['EncodedPixels'] = final_sub['enc1']
final_sub['EncodedPixels'][idx] = final_sub['enc2'][idx]



# if len(list_submissions) > 2:
#     for i in range(2,len(list_submissions)):
#         final_sub = pd.merge(left=final_sub, right=list_submissions[i], on = 'Image_Label', how = 'inner')

final_sub[['Image_Label', 'EncodedPixels']].to_csv('emsemble_sub.csv', index=False)