# Example taken from https://www.tutorialspoint.com/python/python_xml_processing.htm
import xml.sax

class MyHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.currentData = ""
        self.type        = ""
        self.format      = ""
        self.rating      = ""
        self.stars       = ""
        self.year        = ""
        self.description = ""

    def startElement(self, name, attrs):
        self.currentData = name
        if name == 'movie':
            print("*** Movies ***")
            print("title : {}".format(attrs['title']))

    def endElement(self, name):
        name = self.currentData
        if hasattr(self, name):
            print("{} : {}".format(name, getattr(self, name)))
        self.currentData = ""

    def characters(self, content):
        name = self.currentData
        if hasattr(self, name):
            setattr(self, name, content)

if __name__ == "__main__":
   parser = xml.sax.make_parser()
   parser.setFeature(xml.sax.handler.feature_namespaces, 0)
   Handler = MyHandler()
   parser.setContentHandler(Handler)
   parser.parse("config.xml")
