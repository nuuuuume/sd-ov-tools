import argparse
import xml.etree.ElementTree as ET

def analyze_ov_model(ov_model_xml_path):

    with open(ov_model_xml_path, 'rt') as f:
        buf = f.read()

    root = ET.fromstring(buf)
    layers = root.find('layers')
    edges = root.find('edges')
    with open('model.txt', 'w') as f:
        for edge in edges.findall('edge'):
            from_layer = edge.attrib['from-layer']
            from_port = edge.attrib['from-port']
            to_layer = edge.attrib['to-layer']
            to_port = edge.attrib['to-port']

            from_layer_xpath = f"./layer[@id='{from_layer}']"
            from_port_xpath = f"./*/port[@id='{from_port}']"
            from_layer_node = layers.find(from_layer_xpath)

            from_name = f"{from_layer_node.attrib['name']}:{from_port}[{from_layer_node.attrib['type']}]"

            to_layer_xpath = f"./layer[@id='{to_layer}']" 
            to_port_xpath = f"./*/port[@id='{to_port}']"
            to_layer_node = layers.find(to_layer_xpath)
            to_port_node = to_layer.find(to_port_xpath)
            to_name = f"{to_layer_node.attrib['name']}:{to_port}[{to_layer_node.attrib['type']}]"

            f.write(f"{from_name} -> {to_name}\n")

if __name__ == '__main__':

    p = argparse.ArgumentParser()

    p.add_argument('--ov_model_xml_path',
                   dest='ov_model_xml_path',
                   type=str,
                   default=r'models\anzuMix-v1-ov\unet\openvino_model.xml')

    args = p.parse_args()
    analyze_ov_model(args.ov_model_xml_path)