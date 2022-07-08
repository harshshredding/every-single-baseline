from util import read_disease_gazetteer
dis_list = read_disease_gazetteer()
with open('dis_gaz.lst', 'w') as f:
    for disease_term in dis_list:
        print(disease_term, file=f)