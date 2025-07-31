from odf.opendocument import Element

CALCEXT_NS = 'urn:org:documentfoundation:names:experimental:calc:xmlns:calcext:1.0'
CALCEXT = 'calcext'
conditional_formats_key = (CALCEXT_NS, 'conditional-formats')
Element.namespaces[CALCEXT_NS] = CALCEXT

class CalcElement(Element):
    """Specialized Element for calcext namespace with simplified attribute handling."""

    def __init__(self, name, attributes=None, text=None, cdata=None, qattributes=None,
                 check_grammar=True, **args):
        # Initialize with calcext namespace and provided element name
        qname = (CALCEXT_NS, name)
        super(CalcElement, self).__init__(
            attributes=attributes,
            text=text,
            cdata=cdata,
            qname=qname,
            qattributes=qattributes,
            check_grammar=check_grammar,
            **args
        )
        # Ensure CALCEXT_NS uses 'calcext' prefix
        self.tagName = f"{CALCEXT}:{name}"  # Set consistent tag name
    def get_nsprefix(self, namespace):
        """Override to force 'calcext' prefix for CALCEXT_NS"""
        if namespace == CALCEXT_NS:
            return CALCEXT
        return super(CalcElement, self).get_nsprefix(namespace)

    def setAttrNS(self, namespace, localpart, value):
        """Override to enforce 'calcext' prefix for CALCEXT_NS attributes"""
        # Handle CALCEXT_NS specially to ensure consistent prefix
        if namespace == CALCEXT_NS:
            # Force registration of namespace with our prefix
            if namespace not in self.namespaces:
                self.namespaces[namespace] = CALCEXT
            # Use our helper to set the attribute
            self.set_calcext_attribute(localpart, value)
        else:
            # Default behavior for other namespaces
            super(CalcElement, self).setAttrNS(namespace, localpart, value)

    def set_calcext_attribute(self, localname, value):
        """Set an attribute in the calcext namespace using our prefix"""
        # Directly store attribute with forced namespace mapping
        self.attributes[(CALCEXT_NS, localname)] = value
        # Ensure namespace is registered with our prefix
        if CALCEXT_NS not in self.namespaces:
            self.namespaces[CALCEXT_NS] = CALCEXT

    def get_calcext_attribute(self, localname):
        """Get an attribute value from the calcext namespace."""
        return self.getAttrNS(CALCEXT_NS, localname)