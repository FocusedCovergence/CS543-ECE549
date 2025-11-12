# CS543-ECE549

Use getFitzpatrick17kImg.py to download all Fitzpatrick images

Use get_new_csv.py to filter the csv file

Train in ResNet/train_fitz.ipynb

    Use the fitzpatrick_scale as skin_tone, map 
    skin_tone_map = {
        1: 12, 2: 12,
        3: 34, 4: 34,
        5: 56, 6: 56,
    } to ddi for test

    Only keep "malignant" and "benign" records in three_partition_label