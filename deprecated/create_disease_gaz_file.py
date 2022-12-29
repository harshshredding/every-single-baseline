from util import read_disease_gazetteer
dis_list = read_disease_gazetteer()
with open('gazetteers/dis_gaz.lst', 'w') as f:
    for disease_term in dis_list:
        disease_term = disease_term.replace(":","")
        print(disease_term, file=f)