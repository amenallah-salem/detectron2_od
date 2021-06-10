import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import pandas as pd 
from config_utils import read_config


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
        image_path = data['data_path'] +'/' + folder
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv((data['data_path'] + "/" + folder + '_labels.csv'), index=None)


if __name__=='__main__':
    data= read_config("./config.yaml")
    print("reading config.yaml file: \n", data)

    main()
    print('Successfully converted xml to csv.')
    
    train_labels = pd.read_csv(os.path.join(data['data_path'],'train_labels.csv'))
    valid_labels = pd.read_csv(os.path.join(data['data_path'],'valid_labels.csv'))
    print('head of train labels.csv: {}\nhead of valid_labels.csv: {}'.format(train_labels, valid_labels))
