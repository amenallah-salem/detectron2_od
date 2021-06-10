import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import pandas as pd 
from config import red_config


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member.find("bndbox").find('xmin').text),
                     int(member.find("bndbox").find('ymin').text),
                     int(member.find("bndbox").find('xmax').text),
                     int(member.find("bndbox").find('ymax').text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    for folder in ['train', 'valid']:
        image_path = os.path.join(os.getcwd(), ('data/' + folder))
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv(('data/'+folder+'_labels.csv'), index=None)
    print('Successfully converted xml to csv.')

if __name__=='__main__':
    data= read_config("./config.yaml")
    main()
    local_dir = os.getcwd()
    list_files = []
    for files in os.listdir(local_dir):

        list_files.append(files)
    if 'train_utils' in list_files:
        print('Preparing train utils..\ file train_utils exist')
        pass
    else:
        print("directory train_utils doen't exist... creating ./train_utils folder in progress")
        os.mkdir('train_utils')
    #############
    if 'history' in list_files:
        print('history exist')
        pass
    else:
        print("directory history doen't exist... creating history folder in progress")
        os.mkdir('history')

    ################


    
    train_labels = pd.read_csv('./data/train_labels.csv')
    valid_labels = pd.read_csv('./data/valid_labels.csv')
    print('head of train labels.csv: {}\nhead of valid_labels.csv: {}'.format(train_labels, valid_labels))
