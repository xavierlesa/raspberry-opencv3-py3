from random import randint
import uuid
import time

class Person:
    tracks = []
    def __init__(self, v1, v2, max_age):
        self.id = uuid.uuid4().hex
        self.v1 = v1
        self.v2 = v2
        self.tracks = []
        self.R = randint(0,255)
        self.G = randint(0,255)
        self.B = randint(0,255)
        self.done = False
        self.state = '0'
        self.age = 0
        self.max_age = max_age
        self.dir = None

    def get_centroid(self):
        """
        Return the centroid of vector
        """
        x, y = self.v1
        w, h = self.v2
        dx = (w + x)/2
        dy = (h + y)/2
        return int(dx), int(dy)

    def get_area(self):
        """
        Calculate and return area
        """
        pass

    def getRGB(self):
        return (self.R,self.G,self.B)

    def getTracks(self):
        return self.tracks

    def getId(self):
        return self.id

    def getState(self):
        return self.state

    def getDir(self):
        return self.dir
    
    def getX(self):
        return self.v1[0]
    
    def getY(self):
        return self.v1[1]
    
    def updateCoords(self, v1, v2):
        self.age = 0
        self.tracks.append([self.v1, self.v2])
        self.v1 = v1
        self.v2 = v2
    
    def setDone(self):
        self.done = True
    
    def timedOut(self):
        return self.done
    
#    def going_UP(self,mid_start,mid_end):
#        if len(self.tracks) >= 2:
#            if self.state == '0':
#                if self.tracks[-1][1] < mid_end and self.tracks[-2][1] >= mid_end: #cruzo la linea
#                    state = '1'
#                    self.dir = 'up'
#                    return True
#            else:
#                return False
#        else:
#            return False
#    
#    def going_DOWN(self,mid_start,mid_end):
#        if len(self.tracks) >= 2:
#            if self.state == '0':
#                if self.tracks[-1][1] > mid_start and self.tracks[-2][1] <= mid_start: #cruzo la linea
#                    state = '1'
#                    self.dir = 'down'
#                    return True
#            else:
#                return False
#        else:
#            return False
    
    def age_one(self):
        self.age += 1
        if self.age > self.max_age:
            self.done = True
        return True

class MultiPerson:
    def __init__(self, persons, xi, yi):
        self.persons = persons
        self.x = xi
        self.y = yi
        self.tracks = []
        self.R = randint(0,255)
        self.G = randint(0,255)
        self.B = randint(0,255)
        self.done = False
        
