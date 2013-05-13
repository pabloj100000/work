def readHeader(fileName, equalString='='):
    '''
      Read parameters from header.
      Each header line should start with a comment char '#' and be followed by a parameter name = parameter value
      
      Parameters
      ----------
      fileName : str
          full or relative path to the file
      
      Returns
      -------
      dictionary : key:value pairs from header
    '''
    output = {}

    f_object = open(fileName, 'r')
    line = f_object.readline()
    
    while line[0] is '#':
        keyValue = line.split(equalString)
        if len(keyValue) is 2:
            output[keyValue[0][1:].strip()] = tryeval(keyValue[1].strip())
        
        line = f_object.readline()
        
    f_object.close()
    return output


def tryeval(val):
    import ast
    try:
        val = ast.literal_eval(val)
    except ValueError:
        pass
    
    return val
