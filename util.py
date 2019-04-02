#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 10:30:24 2019

@author: zhenhao
"""
import time
import pickle

class timeKeeper:
    def __init__(self):
        self.initTime=time.time()
        self.latestTime=None
        self.startTime=self.initTime
    
    def startTimer(self):
        self.startTime = time.time()
        self.latestTime=self.startTime
        
    def timeTranslate(self, elapsedTime, ttype):
        if ttype == 'sec':
            return round(elapsedTime,2)
        else:
            return round(elapsedTime/60,2)
        
    def iterTime(self, event='', ttype='sec'):
        cur=time.time()
        elapsed=cur-self.latestTime
        self.latestTime=cur
        output=self.timeTranslate(elapsed, ttype)
        print('elapsed Time for %s: '%(event), output, ttype)
        return output
        
    def totalElapsedTime(self, event='', ttype='sec'):
        cur=time.time()
        elapsed=cur-self.startTime
        self.latestTime=cur
        output=self.timeTranslate(elapsed, ttype)
        print('elapsed Time for %s: '%(event), output, ttype)
        return output
    
class storage:
    def __init__(self):
        self.storeDir='store/storage.p'
        self.storeLogs=self.openFile(self.storeDir)
        if self.storeLogs is None:
            self.storeLogs={}
        
    def store(self, fname, data, restore=True):
        with open(fname, 'wb') as f:
            pickle.dump(data, f)
        if restore:
            self.storeLogs[fname] = data
            self.store(self.storeDir, self.storeLogs, False)
            
    def openFile(self, fname):
        try:
            with open(fname, 'rb') as f:
                return pickle.load(f)
        except:
            return None