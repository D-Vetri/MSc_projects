<h2><center><span style='Color:Magenta'>
Generating Euler angles from multiple files 

</span></center></h2>

* Ensure the files are rotation matrices saved as .npy with array sizes N,9

* The 9 would be all the elements of the matrices in a single row and N being the size of the batch.

* The <span style='color:Green'>Euler_to_Rotation_ZXZ</span> python script converts the Rotations to matrices and if needed can also be used to convert Euler angles to rotation matrices. 

* Get the directory of the npy file and paste in the variable <span style='color:orange'>rot_mat_dir</span>. Use absolute path (ctrl+shift+c in windows to get the path of the directory)

* Place the desired directory where the Euler angle data is to be saved in the variable <span style='color:orange'>euler_save_path</span>.

* The Euler angles are generated as <strong>.txt</strong> files within the sub-directory <strong><span style='color:Purple'>euler_data</span></strong> with the file names of respective rotation matrices npy files.  


