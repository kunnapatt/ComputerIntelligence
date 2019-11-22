import matplotlib.pyplot as plt
import numpy as np
import random

plt.rcParams['font.family'] = 'Tahoma'

#### Input
inpu = [187, 13]

xlist = range(150, 250)
tlist = range(9, 16)

graph = 1

def main():

    if graph :
        graphrule()

    for i in range(10):
        inpu = []
        inpu.append(random.randrange(150, 250))
        inpu.append(random.randrange(8, 20))
        print("Input =", inpu)
        minrule = minOfrule(inpu)
        # print("Min Rule =", minrule)
        mamdani(minrule)
        print("--------------------")
    

##### Rule 
def temperature10(i, down = 160):
    ran = 20
    if i < down :
        membership = 1
    elif i >= down and i < down + ran :
        membership = i *(-1/ran) + (1 + down/ran)
    else :
        membership = 0
    return membership

def temperature11(i, mid = 180):
    ran = 20
    if i == mid :
        membership = 1
    elif i > mid - ran and i < mid :
        membership = i *(1/ran) + (1 - mid/ran)
    elif i > mid and i < mid + ran  :
        membership = i *(-1/ran) + (1 + mid/ran)
    else :
        membership = 0
        
    return membership

def temperature12(i, up = 200):
    ran = 20
    if i > up :
        membership = 1
    elif i > up - ran and i <= up :
        membership = i *(1/ran) + (1 - up/ran)  
    else :
        membership = 0
    return membership

def time10(i, down = 10):
    ran = 2
    if i < down :
        y = 1
    elif i >= down and i < down + ran :
        y = (down + ran - i)*(1/ran)
    else :
        y = 0

    return y

def time11(i, mid = 12): 
    ran = 2
    if i == mid :
        y = 1
    elif i > mid - ran and i < mid :
        y = i * (1/ran) - mid/ran + 1
    elif i > mid and i < (mid + ran) :
        y = i * (1/ran) - mid/ran
    else :
        y = 0

    return y

def time12(i, up = 14):
    ran = 2
    if i < up - ran :
        y = 0
    elif i >= up - ran and i < up :
        y = i/(ran) - (up/ran - 1)
    else :
        y = 1

    return y

def roast00(i, down = 1):
    ran = 1
    if i <= down :
        membership = 1
    elif i > down and i < down + ran :
        membership = ((1)/(-ran))*i
    else :
        membership = 0

    return membership

def roast01(i, mid = 2):
    ran = 1
    if i == mid :
        membership = 1
    elif i > mid and i < mid + ran or i < mid and i > mid - 1 :
        membership = ((1)/(-ran))*i
    else :
        membership = 0

    return membership

def roast02(i, up = 2):
    ran = 1
    if i <= up :
        membership = 0
    elif i > up and i < up + ran :
        membership = ((1)/(up - ran))*i
    else :
        membership = 1

    return membership

#### Find min
def minOfrule(inpu): # Find Alpha cut return listmin
    Listmin = []
    
    L1 = min(temperature10(inpu[0]), time10(inpu[1]))
    L2 = min(temperature10(inpu[0]), time11(inpu[1]))

    M1 = min(temperature11(inpu[0]), time10(inpu[1]))
    M2 = min(temperature12(inpu[0]), time10(inpu[1]))
    M3 = min(temperature11(inpu[0]), time11(inpu[1]))
    M4 = min(temperature10(inpu[0]), time12(inpu[1]))
    M5 = min(temperature11(inpu[0]), time12(inpu[1]))

    D1 = min(temperature12(inpu[0]), time11(inpu[1]))
    D2 = min(temperature12(inpu[0]), time12(inpu[1]))

    Listmin.append([L1, L2])
    Listmin.append([M1, M2, M3, M4, M5])
    Listmin.append([D1, D2])

    return Listmin

#### Mamdani
def mamdani(minOfrule):
    alpha = minOfrule

    maxlevel = []

    # print(alpha)

    for i in range(len(alpha)):
        maxlevel.append(max(alpha[i]))
    
    # print(maxlevel)

    ###### Simplified Centroid
    defuzzi = np.zeros(2)
    # print(len(alpha))
    for i in range(len(alpha)):
        for j in range(len(alpha[i])):
            defuzzi[0] += alpha[i][j]*(i+1)
            defuzzi[1] += alpha[i][j]

    defuzzify = defuzzi[0]/defuzzi[1]

    if ( defuzzify >= 1 and defuzzify < 2):
        if defuzzify - 1 != 0 :
            print("Light Roast: {:.2f} %" .format((defuzzify - 1)*100))
        else :
            print("Light Roast")
    elif ( defuzzify >= 2 and defuzzify < 3 ) :
        if defuzzify - 2 != 0 :
            print("Medium Roast: {:.2f} %" .format((defuzzify - 2)*100))
        else :
            print("Medium Roast")
    elif ( defuzzify >= 3 ) :
        if defuzzify - 3 != 0 :
            print("Dark Roast: {:.2f} %" .format((defuzzify - 3)*100))
        else :
            print("Dark Roast")

##### Graph Rule
def graphrule():
    tem1 = [temperature10(x) for x in xlist]
    tem2 = [temperature11(x) for x in xlist]
    tem3 = [temperature12(x) for x in xlist]

    tim = [time10(t, 10) for t in tlist]
    tim1 = [time11(t, 12) for t in tlist]
    tim2 = [time12(t, 14) for t in tlist]

    roaslist = range(0,5)
    roas = [roast00(x) for x in roaslist]
    roas1 = [roast01(x) for x in roaslist]
    roas2 = [roast02(x) for x in roaslist]
    # roas = [roast00(x) for t in roaslist]

    fig, axs = plt.subplots(3, 1)

    axs[0].set_title(label=u"อุณหภูมิ")
    axs[0].plot(xlist, tem1, label=u'ร้อนน้อย')
    axs[0].plot(xlist, tem2, label=u'ร้อนกลาง')
    axs[0].plot(xlist, tem3, label=u'ร้อนมาก')
    axs[0].set_xlabel('Temperature')
    axs[0].set_ylabel('Membership')
    axs[0].legend()

    axs[1].set_title(label=u"เวลา")
    axs[1].plot(tlist, tim, label=u'น้อย')
    axs[1].plot(tlist, tim1, label=u'ปานกลาง')
    axs[1].plot(tlist, tim2, label=u'มาก')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Membership')
    axs[1].legend()

    axs[2].plot(roaslist, roas, label=u'คั่วอ่อน')
    axs[2].plot(roaslist, roas1, label=u'คั่วกลาง')
    axs[2].plot(roaslist, roas2, label=u'คั่วเข้ม')
    axs[2].set_xlabel('Roast')
    axs[2].set_ylabel('Membership')
    axs[2].legend()
    # axs[2].fill_between(roaslist, 0, [x - 0.3 for x in roas])

    plt.show()

if __name__ == "__main__":    
    main()
