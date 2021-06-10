
import yaml

def red_config(yaml_dir):
    with open(os.path.join(yaml_dir, "config.yaml"), "r") as yamlfile:
      data = yaml.load(yamlfile, Loader=yaml.FullLoader)
    return data 

  def yaml_dump(filepath_ , data_):
    with open(filepath_,'w') as file_descriptor:
      yaml.dump(data, file_descriptor)

def update_key_dic_cfg_yaml(main_dir, key_old_one, Value_old_one, new_one):
  with open(os.path.join(main_dir,"config.yaml")) as yamlfile:
    data = yaml.load(yamlfile, Loader=yaml.FullLoader)
    print(data)
    print("Reading cfg.yaml file successfully.")
    data[key_old_one][Value_old_one] = new_one
    yamlfile.close()

def update_key_val_cfg_yaml(main_dir, key_old_one, value_new_one):
  with open(os.path.join(main_dir,"cfg.yaml")) as yamlfile:
    data = yaml.load(yamlfile, Loader=yaml.FullLoader)
    print(data)
    print("Updating cfg.yaml file successful.")
    data[key_old_one][Value_old_one] = new_one
     
    yamlfile.close()

