# train utils



def print_top_accuracy_TP(P, T, F, st):
    if len(P) != len(T) or len(T) != len(F):
        raise
    s = "\t"+st + "\t"
    for i in range(len(P)):
        s += "P" +str(i+1)+"="
        s += "%3.3f," %(P[i])
    s = s[:-1] + "; "
    for i in range(len(T)):
        s += "T"+str(i+1)+"="+str(T[i]) + ",F" + str(i+1)+"="+str(F[i])+","
    s = s[:-1]
    print(s)
    
    
# calculate details
def cal_metrics(results, icons, str, N=2):
    # results: top N icon indices returned by Text2Vec for each phrase
    # icon idx for each phrase-icon pair
#         print(results)
#         print(icons)

#         if len(results) != len(labels) or len(results) != len(icons):
#             print("error: len of inputs not equal")
#             raise
    P, T, F = [-404]*N, [0]*N, [0]*N
    for i in range(len(results)):
        if icons[i][results[i][0]] == 1:
            T[0] += 1
            T[1] += 1
        elif icons[i][results[i][1]] == 1:
            F[0] += 1
            T[1] += 1
        else:
            F[0] += 1
            F[1] += 1
#                 print(icons[i][results[i][0]],  icons[i][results[i][1]], T, F)
    for n in range(N):
        P[n] = T[n]/(T[n]+F[n])
        
    print_top_accuracy_TP(P, T, F, str)
    return P

def cal_NewIconFireRate(results):
    totalSentence = len(results)
    newIconSentence = 0
    for i in range(len(results)):
        if results[i][0] > 490 or results[i][1] > 490:
            newIconSentence += 1

    s = "\t\tNewIconFireRate="
    s += "%3.3f," %(newIconSentence/totalSentence)
    s += "newIconSentence=" + str(newIconSentence)
    print(s)

    
    
    
    

