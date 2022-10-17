
import os


file = open('label.txt', mode='a+')
list = os.listdir('./data/label/')
for i in range(len(list)):
    file.write('./data/label/%s \n' % (list[i]))
file.close()
