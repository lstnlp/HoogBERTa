import sys 
from sklearn import metrics 

from seqeval.metrics import classification_report
from seqeval.metrics import f1_score

if __name__ == "__main__":
    f1 = open(sys.argv[1],"r").readlines()
    f2 = open(sys.argv[2],"r").readlines()

    if len(f1) != len(f2):
        print("Error : Line not equal !")
    
    A = []
    B = []

    for a , b in zip(f1,f2):
        a = a.strip().split()
        b = b.strip().split()

        if len(a) != len(b):
            print("Error : Size not equal" , len(a) , len(b) )
            print(a,b)

        A.extend(a)
        B.extend(b)
    
    acc = metrics.classification_report(A, B, digits=3)
    print(acc)