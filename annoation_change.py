import os
import xml.etree.ElementTree as ET

def update_annotations(annotation_folder):
    # Iterate over all XML files in the folder
    for filename in os.listdir(annotation_folder):
        if filename.endswith('.xml'):
            annotation_path = os.path.join(annotation_folder, filename)
            update_annotation(annotation_path)

def update_annotation(annotation_path):
    # Parse the XML file
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    # Insert missing parameters at the appropriate positions
    size = root.find('size')
    size_index = list(root).index(size)

    for child in root.findall('object'):
        pose = ET.Element('pose')
        pose.text = 'Unspecified'

        truncated = ET.Element('truncated')
        truncated.text = '0'

        difficult = ET.Element('difficult')
        difficult.text = '0'

        child.insert(0, pose)
        child.insert(1, truncated)
        child.insert(2, difficult)

    folder = ET.Element('folder')
    folder.text = 'VOC2007'
    root.insert(size_index, folder)

    source = ET.Element('source')
    database = ET.Element('database')
    database.text = 'Unknown'
    source.append(database)
    root.insert(size_index + 1, source)

    # Save the updated XML
    tree.write(annotation_path)

# Path to the folder containing annotations
annotation_folder = r'/data/mn918/data/VOC2007/Annotations'

# Update all annotations in the folder
update_annotations(annotation_folder)
