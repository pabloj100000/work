# To run the test execute in bash:
# nosetests test_createExpLog.py 
from createExpLog import createExpLog

def test_AddingLine():
    """
    Test the case in which the log file exists. but the line I want is not in there
    """
    import os.path as osp
    import string
    
    for a in string.lowercase:
        newfile = osp.join('/Users/jadz/Desktop/', 'trash_' + a)
        if osp.isfile(newfile):
            print newfile, 'exists'
        else:
            print newfile, 'does not exist'
            createExpLog(newfile, 'test just created')
            break
            

    # try adding again the line 'test just created'
    print 'trying to add "test just created" again to', newfile
    createExpLog(newfile, 'test just created')
    
    # if everything went well I should have a file in Desktop with just one line
    obj = file_len(newfile)
    exp = 1
    assert exp == obj
    
    # remove teh file just created
    print 'removing file', newfile
    import os
    os.remove(newfile)
    
def test_NewFile():
    """
    Test the case in which the log file does not exist. For that I have to find a file name that does not exist 1st
    """
    import os.path as osp
    import string
    
    for a in string.lowercase:
        newfile = osp.join('/Users/jadz/Desktop/', 'trash_' + a)
        if osp.isfile(newfile):
            print newfile, 'exists'
        else:
            print newfile, 'does not exist'
            createExpLog(newfile, 'test just created')
            break
            
    # if everything went well I should have a file in Desktop with just one line
    obj = file_len(newfile)
    exp = 1
    assert exp == obj
  
    # remove the file just created
    print 'removing file', newfile
    import os
    os.remove(newfile)

def test_ExistingLine():
  """
    Test the case in which the log file exists. but the line I want is not in there
  """
  import os.path as osp
  import string
  
  for a in string.lowercase:
    newfile = osp.join('/Users/jadz/Desktop/', 'trash_' + a)
    if osp.isfile(newfile):
      print newfile, 'exists'
    else:
      print newfile, 'does not exist'
      createExpLog(newfile, 'test just created')
      break
  
  
  # add a new line
  createExpLog(newfile, '2nd line of the expeirment')
  
  # try adding again the line 'test just created'
  createExpLog(newfile, 'test just created')
  
  # try adding again
  createExpLog(newfile, '2nd line of the expeirment')
  
  # if everything went well I should have a file in Desktop with just one line
  obj = file_len(newfile)
  exp = 2
  assert exp == obj
  
  # remove teh file just created
  print 'removing file', newfile
  import os
  os.remove(newfile)

def file_len(fname):
  with open(fname) as f:
    for i, l in enumerate(f):
      pass
  return i + 1

