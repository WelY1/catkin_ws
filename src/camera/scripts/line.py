# 此文件用来计算图中两直线交点
# 直线定义[x1,y1,x2,y2]

def cross_point(line1,line2):#计算交点函数
    x1=line1[0]#取四点坐标
    y1=line1[1]
    x2=line1[2]
    y2=line1[3]
    
    x3=line2[0]
    y3=line2[1]
    x4=line2[2]
    y4=line2[3]
    
    k1=(y2-y1)*1.0/(x2-x1)#计算k1,由于点均为整数，需要进行浮点数转化
    b1=y1*1.0-x1*k1*1.0#整型转浮点型是关键
    if (x4-x3)==0:#L2直线斜率不存在操作
        k2=None
        b2=0
    else:
        k2=(y4-y3)*1.0/(x4-x3)#斜率存在操作
        b2=y3*1.0-x3*k2*1.0
    if k2==None:
        x=x3
    else:
        x=(b2-b1)*1.0/(k1-k2)
    y=k1*x*1.0+b1*1.0
    return [x,y]
    
if __name__ == '__main__':
    line1 = [775,631,822,542]
    line2 = [775,631,1674,618]
    line3 = [1537,393,1622,426]
    line4 = [1005,327,1205,323]
    [x1,y1] = cross_point(line1,line2)
    [x2,y2] = cross_point(line2,line3)
    [x3,y3] = cross_point(line3,line4)
    [x4,y4] = cross_point(line4,line1)
    print((x1,y1))            # (775, 631)
    print((x2,y2))            # (2101, 612)
    print((x3,y3))            # (1349, 320)
    print((x4,y4))            # (935, 328)