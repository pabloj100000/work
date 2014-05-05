def createExpLog(logFile, currentExperiment):
    f_log = open(logFile, 'a+')  # open file for appending and reading. Current position is at end of file
    
    f_log.seek(0) # set current position at start of file
    
    line = f_log.readline()[0:-1]  # discard the \n at the end of the line
    while line:
        if line == currentExperiment:
            f_log.close()
            return
        
        # read another line until EOF
        line = f_log.readline()[0:-1] 
        
    if not f_log.closed:
        # if function got to this point, means that currentExperiment is not in logFile
        f_log.write(currentExperiment + '\n')
        f_log.close()
    
