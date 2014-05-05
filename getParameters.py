def getParameters(defaultDict):
  import sys
  
  for key, value in defaultDict.items():
    valueType = type(value)

    if valueType==type([]):
      sys.exit('getParameters can NOT deal with lists yet')
    elif valueType==type({}):
      sys.exit('getParameters can NOT deal with dictionaries yet')

    promptStr = key + '(' + str(value) + '): '
	
    newValue = raw_input(promptStr)

    if newValue:
      defaultDict[key] = valueType(newValue)
            
  return defaultDict


