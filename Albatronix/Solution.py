from pprint import pprint
import networkx as nx
def give_regions(name):
    regions="Default"
    location=["right upper quadrant","right lower quadrant",
                   "left upper quadrant","right upper quadrant",
              "left side","right side"]
    severity=["mild", "moderate","severe"]
    if name in location:
        regions='location'
    elif name in severity:
        regions='severity'
    return regions
my_dict={
"Abdominal pain": ["right upper quadrant","right lower quadrant",
                   "left upper quadrant","right upper quadrant",
                   "mild","moderate","severe"],

"Chest pain": ["left side","right side","mild", "moderate","severe"]}
#print(my_dict)

new_dict={}
for k,v in my_dict.items():
    #print(k,v)
    
    my_arr=v
    regions=[]
    new_arr=[]
    for m in my_arr:
        region=give_regions(m)
        #print(region)
        new_arr.append([m,region])
    new_dict[k]=new_arr

"""
new_dict=
{'Abdominal pain': [['right upper quadrant', 'location'],
                    ['right lower quadrant', 'location'],
                    ['left upper quadrant', 'location'],
                    ['right upper quadrant', 'location'],
                    ['mild', 'severity'],
                    ['moderate', 'severity'],
                    ['severe', 'severity']],
 'Chest pain': [['left side', 'location'],
                ['right side', 'location'],
                ['mild', 'severity'],
                ['moderate', 'severity'],
                ['severe', 'severity']]}
"""
#pprint(new_dict)
G=nx.MultiGraph()
for k,v in new_dict.items():
    my_arr=v
    for [a,b] in my_arr:
        #print(a,b)
        G.add_edge(k,a,relation=b)
"""
[('Abdominal pain', 'right upper quadrant'), ('Abdominal pain', 'right upper quadrant'), ('Abdominal pain', 'right lower quadrant'), ('Abdominal pain', 'left upper quadrant'), ('Abdominal pain', 'mild'), ('Abdominal pain', 'moderate'), ('Abdominal pain', 'severe'), ('mild', 'Chest pain'), ('moderate', 'Chest pain'), ('severe', 'Chest pain'), ('Chest pain', 'left side'), ('Chest pain', 'right side')]
"""
#print(G.edges())

"""
['Abdominal pain', 'right upper quadrant']
['Abdominal pain', 'right upper quadrant']
['Abdominal pain', 'right lower quadrant']
['Abdominal pain', 'left upper quadrant']
['Abdominal pain', 'right upper quadrant']
['Abdominal pain', 'right upper quadrant']
['Abdominal pain', 'mild']
['Abdominal pain', 'moderate', 'Chest pain', 'mild']
['Abdominal pain', 'mild', 'Chest pain', 'moderate']
['Abdominal pain', 'mild', 'Chest pain', 'severe']
['Chest pain', 'left side']
['Chest pain', 'right side']
['Chest pain', 'mild']
['Chest pain', 'moderate', 'Abdominal pain', 'mild']
['Chest pain', 'mild', 'Abdominal pain', 'moderate']
['Chest pain', 'mild', 'Abdominal pain', 'severe']
"""
keys=list(new_dict.keys())
#print(keys)
"""
symptom  Abdominal pain
location Abdominal pain right upper quadrant
location Abdominal pain right upper quadrant
location Abdominal pain right lower quadrant
location Abdominal pain left upper quadrant
location Abdominal pain right upper quadrant
location Abdominal pain right upper quadrant
severity Abdominal pain mild
severity Abdominal pain moderate
severity Abdominal pain severe
symptom  Chest pain
location Chest pain left side
location Chest pain right side
severity Chest pain mild
severity Chest pain moderate
severity Chest pain severe

"""
for k,v in new_dict.items():
    my_arr=v
    print("symptom ",k)
    for [a,b] in my_arr:
        for path in nx.all_simple_paths(G,source=k,target=a):
            count=0
            flag=True
            #print(path)
            for p in path:
                
                if p in keys:
                    count+=1
            #print("count => ",count)
            if count>1:
                flag=False
            if flag==True:
                print(b,*path)
print("*"*100)
u,v='Abdominal pain', 'right upper quadrant'
print("Common neighbors between u and v:-")
"""
['right upper quadrant', 'right lower quadrant', 'left upper quadrant', 'mild', 'moderate', 'severe']
['Abdominal pain']
"""
#print(list(nx.all_neighbors(G,u)))
#print(list(nx.all_neighbors(G,v)))
symptoms_of_u=list(nx.all_neighbors(G,u))
symptoms_of_v=list(nx.all_neighbors(G,v))
for s1 in symptoms_of_u:
    for s2 in symptoms_of_v:
        print(s2,s1)
print("*"*50)

u='Abdominal pain'
print("Common neighbors in the root element u is :-")
"""
"""
last_ans=[]
if u in keys:
    for k in keys:
        if k!=u:
            last_ans.append(k)
for k in last_ans:
    print(k)

