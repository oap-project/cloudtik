import sys
from xml.dom import minidom
import os
import fnmatch
import xml.etree.cElementTree as ET
from xml.dom.minidom import parseString

runtime_home = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_conf = os.path.join(runtime_home, "confs/base_confs")
pretty_print = lambda data: '\n'.join(
    [line for line in parseString(data).toprettyxml(indent=' ' * 2).split('\n') if line.strip()])


def replace_conf_value(conf_file, dict):
    with open(conf_file) as f:
        read = f.read()
    with open(conf_file, 'w') as f:
        for key,val in dict.items():
            read = read.replace(key, val)
        f.write(read)


def replace_name_value(conf_file, name, value):
    tree = ET.parse(conf_file)
    root = tree.getroot()
    for property_tag in root.findall("property/[name='" + name + "']"):
        property_tag.find("value").text = value
    tree.write(conf_file)

def add_config_value(conf_file, dict, delimeter):
    with open(conf_file, 'a') as f:
        for key,val in dict.items():
            if (not key == ""):
                f.write(key + delimeter + val + "\n")

def add_property_element(root_elemnt, name, value):
    property_element = ET.SubElement(root_elemnt, "property")
    name_element = ET.SubElement(property_element, "name")
    value_element = ET.SubElement(property_element, "value")
    name_element.text = name
    value_element.text = value


def format_xml_file(xml_file):
    xmlstr = ""
    with open(xml_file, 'r') as f:
        xmlstr = pretty_print(f.read())
    with open(xml_file, "w") as f:
        f.write(xmlstr)


def get_config_value(conf_file, name):
    tree = ET.parse(conf_file)
    root = tree.getroot()
    for property_tag in root.findall("property/[name='" + name + "']"):
        return property_tag.find("value").text


# merge configuration file
def update_conf(component, conf_root):
    output_conf = os.path.join(conf_root, "output")
    mkdirs(output_conf)
    
    output_component_conf = os.path.join(output_conf, component)

    # create output dir for merged configuration file
    os.system("rm -rf " + output_component_conf)
    os.system("mkdir -p " + output_component_conf)

    all_files = list_files_tree(conf_root, component)
    all_file_names = all_files.keys()
    
    component_conf = os.path.join(conf_root, component)
    
    for xml_file in [file for file in all_file_names if fnmatch.fnmatch(file, '*.xml')]:
        file_name = os.path.join(component_conf, xml_file)
        output_file_name = os.path.join(output_component_conf, xml_file)
        if xml_file == "persistent-memory.xml":
            merge_pmem_xml_properties_tree(conf_root, file_name, output_file_name)
        else:
            merge_xml_properties_tree(conf_root, file_name, output_file_name)
        all_files[xml_file] = "1"

    for conf_file in [file for file in all_file_names if fnmatch.fnmatch(file, '*.conf')]:
        file_name = os.path.join(component_conf, conf_file)
        output_file_name = os.path.join(output_component_conf, conf_file)
        merge_properties_tree(conf_root, file_name, output_file_name, " ")
        all_files[conf_file] = "1"

    for config_file in [file for file in all_file_names if fnmatch.fnmatch(file, '*config')]:
        file_name = os.path.join(component_conf, config_file)
        output_file_name = os.path.join(output_component_conf, config_file)
        merge_properties_tree(conf_root, file_name, output_file_name, " ")
        all_files[config_file] = "1"

    # copy the remaining files in hierarchy way
    for file in all_file_names :
       if (not all_files[file] == "1") :
          copy_conf_tree(conf_root, os.path.join(component, file), output_conf)
          
    return output_component_conf

def list_files_tree(conf_root, component) :
    # list this folder
    all_files = {}
    
    component_conf = os.path.join(conf_root, component)
    if(os.path.isdir(component_conf)) :
       files = os.listdir(component_conf)
       for file in files :
           all_files[file] = "0"
    
    # list base files
    base_conf = get_base_conf(conf_root)
    if( base_conf == "") :
       return all_files
       
    base_files = list_files_tree(base_conf, component)
    for file in base_files :
        all_files[file] = "0"
    
    return all_files


def get_configs_from_properties (filename):
    result = {}
    with open(filename, 'r') as f:
        for line in f:
            kv = line.split()
            if line.startswith('#') or len(kv) != 2:
                continue
            result[kv[0]] = kv[1]
    return result
    

# base hierarchy handling
def get_base_name (conf_root):
    base_conf = ""
    base_meta = os.path.join(conf_root, ".base")
    if( not os.path.isfile(base_meta) ) :
        return base_conf
    
    with open(base_meta, 'r') as f:
        for line in f:
            return line.strip()
            
    return base_conf
    
def get_base_conf (conf_root):
    if( conf_root == root_conf ) :
       return ""
    
    base_name = get_base_name(conf_root)
    if(base_name == ""):
      return root_conf
    
    return os.path.abspath(os.path.join(conf_root, base_name))

def get_base_conf_file (conf_root, conf_file) :
    conf_file_relative = os.path.relpath(conf_file, conf_root)
    return get_base_conf_file_relative(conf_root, conf_file_relative)
    
def get_base_conf_file_relative (conf_root, relative_conf_file):    
    base_conf = get_base_conf(conf_root)
    if(base_conf == "") :
       return ""
    
    return os.path.join(base_conf, relative_conf_file)

# properties file hierarchy handling
def merge_properties_tree (conf_root, conf_file, output_filename, delimeter):
    props_merged = get_properties_tree(conf_root, conf_file, delimeter)
    
    with open(output_filename, 'w') as f:
        for k, v in props_merged.items():
            if(not k == ""):
               f.write(k + delimeter + v + "\n")

def get_properties_tree (conf_root, conf_file, delimeter):
    props_base = get_properties_base(conf_root, conf_file, delimeter)

    if( os.path.isfile(conf_file) ) :
        props_from = parse_properties(conf_file, delimeter)
        props_base = merge_properties(props_base, props_from)
    
    return props_base
    
def parse_properties (filename, delimeter):
    result = {}
    with open(filename, 'r') as f:
        for line in f:
            #kv = line.split(delimeter)
            kv = line.split(delimeter,1)
            #if line.startswith('#') or len(kv) != 2:
            if line.startswith('#'):
                continue
            if len(kv) < 2:
                kv = line.split("=", 1)
                if len(kv) < 2:
                    continue
            result[kv[0]] = kv[1].strip('\n')
    
    return result
        
def get_properties_base (conf_root, conf_file, delimeter):
    result = {}
    
    base_conf = get_base_conf (conf_root)
    if(base_conf == "") :
       return result
    
    conf_file_relative = os.path.relpath(conf_file, conf_root)
    base_conf_file = os.path.join(base_conf, conf_file_relative)
    return get_properties_tree(base_conf, base_conf_file, delimeter)

def merge_properties(props, props_from) :
    for k, v in props_from.items():
        if k.startswith("--"):
            k = k.replace(k[0], "", 2)
            props[k] = props[k].replace(str(v).strip(), '')
        elif k.startswith("++"):
            k = k.replace(k[0], "", 2)
            props[k] = props[k].replace(str(v).strip(), '')
            props[k] = props[k].strip() + str(v).strip()
        elif k.startswith("-"):
            k = k.replace(k[0], "", 1)
            props.pop(k, "nokey")
        else:
            props[k] = v
    return props

# environment hierarchy merge
def get_merged_env (conf_root):
    return get_properties_tree(conf_root, os.path.join(conf_root, "env.conf"), "=")

# xml config hierarchy handling
def mkdirs(path) :
    if(not os.path.exists(path)):
       os.makedirs(path)

def merge_pmem_xml_properties_tree(conf_root, conf_file, output_filename):
    props_merged = get_pmem_xml_properties_tree(conf_root, conf_file)

    mkdirs(os.path.dirname(output_filename))

    xml_output = ET.ElementTree(props_merged)

    with open(output_filename, "w") as f:
        f.write(minidom.parseString(ET.tostring(xml_output.getroot())).toprettyxml(indent="  "))

def get_pmem_xml_properties_tree(conf_root, conf_file):
    props_base = get_pmem_xml_properties_base(conf_root, conf_file)

    if (os.path.isfile(conf_file)):
        props_base = merge_pmem_xml_properties(props_base, conf_file)

    return props_base

def  get_pmem_xml_properties_base(conf_root, conf_file):
    result = ET.Element("persistentMemoryPool")

    base_conf = get_base_conf(conf_root)
    if (base_conf == ""):
        return result

    conf_file_relative = os.path.relpath(conf_file, conf_root)
    base_conf_file = os.path.join(base_conf, conf_file_relative)
    return get_pmem_xml_properties_tree(base_conf, base_conf_file)

def merge_pmem_xml_properties(props, conf_file_from):
    result = ET.Element("persistentMemoryPool")

    tree_from = ET.parse(conf_file_from)
    props_from = tree_from.getroot()

    i = 0
    for numanode_tag in props_from.findall('numanode'):
        initialPath_name = numanode_tag.find('initialPath').text

        numanode_element = ET.SubElement(result, 'numanode',  {'id': str(i)})
        initialPath_element = ET.SubElement(numanode_element, "initialPath")
        initialPath_element.text = initialPath_name
        index=0
        tags_in_default = props.findall('numanode')
        if len(tags_in_default) > 0:
            for element in tags_in_default:
                if element.attrib["id"] == str(i):
                    props.remove(tags_in_default[index])
                index += 1
        i += 1
    return result

def merge_xml_properties_tree (conf_root, conf_file, output_filename):
    props_merged = get_xml_properties_tree(conf_root, conf_file)
    
    mkdirs(os.path.dirname(output_filename))
    
    xml_output = ET.ElementTree(props_merged)

    #xml_output.write(output_filename, encoding="UTF-8", xml_declaration=True)
    with open(output_filename, "w") as f:
        f.write(minidom.parseString(ET.tostring(xml_output.getroot())).toprettyxml(indent="  "))

def get_xml_properties_tree (conf_root, conf_file):
    props_base = get_xml_properties_base(conf_root, conf_file)
    
    if( os.path.isfile(conf_file) ) :
        props_base = merge_xml_properties(props_base, conf_file)
    
    return props_base
        
def get_xml_properties_base (conf_root, conf_file):
    result = ET.Element("configuration")
    
    base_conf = get_base_conf (conf_root)
    if(base_conf == "") :
       return result
    
    conf_file_relative = os.path.relpath(conf_file, conf_root)
    base_conf_file = os.path.join(base_conf, conf_file_relative)
    return get_xml_properties_tree(base_conf, base_conf_file)

def merge_xml_properties(props, conf_file_from) :
    result = ET.Element("configuration")
    
    tree_from = ET.parse(conf_file_from)
    props_from = tree_from.getroot()
    
    for property_tag in props_from.findall("./property"):
        property_name = property_tag.find("name").text
        add_property_element(result, property_name, property_tag.find("value").text)
        tags_in_default = props.findall("*[name='" + property_name + "']")
        if len(tags_in_default) > 0:
            props.remove(tags_in_default[0])

    for property_tag in props.findall("./property"):
        add_property_element(result, property_tag.find("name").text,
            property_tag.find("value").text)
            
    return result

# cluster file handling with hierarchy
def get_cluster_file (conf_root):
    cluster_file = os.path.join(conf_root, "nodes.conf")
    if os.path.isfile(cluster_file):
       return cluster_file
    
    base_conf = get_base_conf(conf_root)
    if(base_conf == "") :
       return ""
    
    return get_cluster_file(base_conf)

# copy files overwritten with hierarchy
def copy_conf_tree(conf_root, component, conf_output):
    component_conf_output = os.path.join(conf_output, component)
    
    if (not os.path.isdir(conf_output)) :
       os.system("mkdir -p " + conf_output)
    
    # clean the output folder
    os.system("rm -rf " + component_conf_output)
    
    copy_component_tree(conf_root, component, component_conf_output)
    
    return component_conf_output
    
def copy_component_tree(conf_root, component, component_conf_output):
    component_conf = os.path.join(conf_root, component)
    
    # copy from the base
    copy_component_base(conf_root, component, component_conf_output)
    
    if (os.path.exists(component_conf)) :
       mkdirs(os.path.dirname(component_conf_output))
       if(os.path.isdir(component_conf_output)) :
          os.system("cp -Trf " + component_conf + " " + component_conf_output)
       else :
          os.system("cp -rf " + component_conf + " " + component_conf_output)
       
    return component_conf_output

def copy_component_base(conf_root, component, component_conf_output):
    base_conf = get_base_conf(conf_root)
    if(base_conf == "") :
       return
    
    copy_component_tree(base_conf, component, component_conf_output)


def get_properties(filename):
    properties = {}
    if not os.path.isfile(filename):
        return properties
    with open(filename) as f:
        for line in f:
            if line.startswith('#') or not line.split():
                continue
            key, value = line.partition("=")[::2]
            properties[key.strip()] = value.strip()
    return properties


def update(custom_conf):
    update_conf("hibench", custom_conf)


def usage():
    print("Usage: python hibench_config_utils.py [conf_dir]/n")
    exit(1)


if __name__ == '__main__':
    args = sys.argv
    if len(args) != 2:
        usage()
    conf_path = args[1]
    update(conf_path)
