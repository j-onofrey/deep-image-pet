import nibabel as nib
import numpy as np
import sys, os

to_transform_file_path = str(sys.argv[1])
output_dir = str(sys.argv[2])
threshold = float(sys.argv[3])

with open(to_transform_file_path) as file:
    filenames = file.read().splitlines() 
    for file in filenames:

        if os.path.isfile(file):

            basename = os.path.basename(file)
            print('Fixing %s' % basename)

            img = nib.load(file)

            imgdata = img.get_data()
            imgmatrix = img.affine
            
            s = imgdata.shape

            for i in range(0,s[0]):
                for j in range(0,s[1]):
                    for k in range(0,s[2]):
                        v = imgdata[i,j,k,2]
                        if v < threshold:
                            imgdata[i,j,k,2] = v / threshold
                        else:
                            imgdata[i,j,k,2] = 1.0

            print('Saving image to %s' % os.path.join(output_dir,basename))
            out_image = nib.Nifti1Image(imgdata, imgmatrix);
            nib.save(out_image, os.path.join(output_dir,basename))
