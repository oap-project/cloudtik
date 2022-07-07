import sys
from xml.dom import minidom
import os
import fnmatch
import xml.etree.cElementTree as ET
from xml.dom.minidom import parseString

runtime_home = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_conf = os.path.join(runtime_home, "confs")
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
    
    component_conf = conf_root
    
    for xml_file in [file for file in all_file_names if fnmatch.fnmatch(file, '*.xml')]:
        file_name = os.path.join(component_conf, xml_file)
        output_file_name = os.path.join(output_component_conf, xml_file)
        if xml_file == "persistent-memory.xml":
            merge_pmem_xml_properties_tree(conf_root, file_name, output_file_name, component)
        else:
            merge_xml_properties_tree(conf_root, file_name, output_file_name, component)
        all_files[xml_file] = "1"

    for conf_file in [file for file in all_file_names if fnmatch.fnmatch(file, '*.conf')]:
        file_name = os.path.join(component_conf, conf_file)
        output_file_name = os.path.join(output_component_conf, conf_file)
        merge_properties_tree(conf_root, file_name, output_file_name, " ", component)
        all_files[conf_file] = "1"

    for config_file in [file for file in all_file_names if fnmatch.fnmatch(file, '*config')]:
        file_name = os.path.join(component_conf, config_file)
        output_file_name = os.path.join(output_component_conf, config_file)
        merge_properties_tree(conf_root, file_name, output_file_name, " ", component)
        all_files[config_file] = "1"
          
    return output_component_conf

def list_files_tree(conf_root, component) :
    # list this folder
    all_files = {}
    
    component_conf = conf_root
    if(os.path.isdir(component_conf)) :
       files = os.listdir(component_conf)
       for file in files :
           if os.path.isfile(os.path.join(component_conf, file)):
                all_files[file] = "0"
    
    # list base files
    base_conf = get_component_base_conf(conf_root, component)
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

def get_component_base_conf(conf_root, component):
    component_base_conf = os.path.join(os.path.join(root_conf, component), "base")
    if (conf_root == component_base_conf):
        return ""
    return component_base_conf

# properties file hierarchy handling
def merge_properties_tree (conf_root, conf_file, output_filename, delimeter, component):
    props_merged = get_properties_tree(conf_root, conf_file, delimeter, component)
    
    with open(output_filename, 'w') as f:
        for k, v in props_merged.items():
            if(not k == ""):
               f.write(k + delimeter + v + "\n")

def get_properties_tree (conf_root, conf_file, delimeter, component):
    props_base = get_properties_base(conf_root, conf_file, delimeter, component)

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
        
def get_properties_base (conf_root, conf_file, delimeter, component):
    result = {}

    component_base_conf = get_component_base_conf(conf_root, component)
    if(component_base_conf == "") :
       return result
    
    conf_file_relative = os.path.relpath(conf_file, conf_root)
    base_conf_file = os.path.join(component_base_conf, conf_file_relative)
    return get_properties_tree(component_base_dir, base_conf_file, delimeter, component)

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

# xml config hierarchy handling
def mkdirs(path) :
    if(not os.path.exists(path)):
       os.makedirs(path)

def merge_pmem_xml_properties_tree(conf_root, conf_file, output_filename, component):
    props_merged = get_pmem_xml_properties_tree(conf_root, conf_file, component)

    mkdirs(os.path.dirname(output_filename))

    xml_output = ET.ElementTree(props_merged)

    with open(output_filename, "w") as f:
        f.write(minidom.parseString(ET.tostring(xml_output.getroot())).toprettyxml(indent="  "))

def get_pmem_xml_properties_tree(conf_root, conf_file, component):
    props_base = get_pmem_xml_properties_base(conf_root, conf_file, component)

    if (os.path.isfile(conf_file)):
        props_base = merge_pmem_xml_properties(props_base, conf_file)

    return props_base

def  get_pmem_xml_properties_base(conf_root, conf_file, component):
    result = ET.Element("persistentMemoryPool")

    component_base_conf = get_component_base_conf(conf_root, component)
    if (component_base_conf == ""):
        return result

    conf_file_relative = os.path.relpath(conf_file, conf_root)
    base_conf_file = os.path.join(component_base_conf, conf_file_relative)
    return get_pmem_xml_properties_tree(component_base_conf, base_conf_file)

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

def merge_xml_properties_tree (conf_root, conf_file, output_filename, component):
    props_merged = get_xml_properties_tree(conf_root, conf_file, component)
    
    mkdirs(os.path.dirname(output_filename))
    
    xml_output = ET.ElementTree(props_merged)

    #xml_output.write(output_filename, encoding="UTF-8", xml_declaration=True)
    with open(output_filename, "w") as f:
        f.write(minidom.parseString(ET.tostring(xml_output.getroot())).toprettyxml(indent="  "))

def get_xml_properties_tree (conf_root, conf_file, component):
    props_base = get_xml_properties_base(conf_root, conf_file, component)
    
    if( os.path.isfile(conf_file) ) :
        props_base = merge_xml_properties(props_base, conf_file)
    
    return props_base
        
def get_xml_properties_base (conf_root, conf_file, component):
    result = ET.Element("configuration")

    component_base_conf = get_component_base_conf(conf_root, component)
    if(component_base_conf == "") :
       return result
    
    conf_file_relative = os.path.relpath(conf_file, conf_root)
    base_conf_file = os.path.join(component_base_conf, conf_file_relative)
    return get_xml_properties_tree(component_base_conf, base_conf_file, component)

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

def get_component(conf_path):
    component_dir = os.path.dirname(os.path.dirname(os.path.abspath(conf_path)))
    component = os.path.basename(component_dir)
    return component

def usage():
    print("Usage: python hibench_config_utils.py [conf_dir]/n")
    exit(1)


if __name__ == '__main__':
    args = sys.argv
    if len(args) != 2:
        usage()
    conf_path = args[1]
    component_dir = os.path.dirname(os.path.dirname(os.path.abspath(conf_path)))
    component_base_dir = os.path.join(component_dir, "base")
    component = get_component(conf_path)
    if os.path.exists(component_base_dir):
        update_conf(component, conf_path)
    else:
        print("Input directory must under the [component]/workloads/ ")
        exit(1)
