# DC1

In order to run the code the following steps should be followed:
1. Open the file 'Data-Challenge-1-template' in the github page.
2. Open the file 'dc1' in the github page.
3. Open all .py files in 'dc1' in a python environment.
4. Pip install all required libraries using 'requirements.txt' located in 'Data-Challenge-1-template' 
5. Run image_dataset.py to create the x and y test and train files.
6. Run main.py for the main code.


All other files can be ignored for grading purposes. These files are used by various members of the group to test things out.
Note:
- Early stopping has been applied to the model. To test the best model in the epoch with best accuracy, the last model weights that were saved is the model weights of the epoch that yielded the best accuracy
- The files experiment_not_augmented.py and experiment_augmentex.py were added to show how the experiments were set up
- The file image_dataset1.py includes augmentations and transformations of the data, which is only used in experiment_augmented.py. In main.py image_dataset.py is used since it gives better results.

IMPORTANT!!! It is important to pip install the cuda version of torch to run the model faster. If not installed, after pip installing all libraries in step 4, Pip uninstall torch and then pip install the cuda version of torch. In order to do that, go to https://pytorch.org/get-started/locally/ and download the version of cuda torch that corresponds to the cuda installed on the computer
