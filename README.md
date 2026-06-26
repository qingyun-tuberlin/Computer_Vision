- need to create a folder (mouse_dataset)
- then create a folder (images)
- put all images in the folder (images)
- put the MouseFrimaceFac_main.csv, and MouserimaceFaces_mgs.csv in the folder (mouse_dataset)

so that when the image path are like: mouse_dataset/images/000001.jpg

export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true  
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/Users/baturu/Documents/Project_CV  
label-studio start

source venv/bin/activate 