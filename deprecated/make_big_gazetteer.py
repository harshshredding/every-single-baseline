with open('./gazetteers/big_disease_gazetteer.lst', 'w') as output_file:
    with open('./gazetteers/train_gazetteer.lst', 'r') as train_gaz_file, \
            open('./gazetteers/dis_gaz.lst', 'r') as diz_gaz_file, \
            open('./gazetteers/umls_disease_gazetteer_new.lst', 'r') as umls_gaz_file:
        for line in train_gaz_file.readlines():
            output_file.write(line)
        for line in diz_gaz_file.readlines():
            output_file.write(line)
        for line in umls_gaz_file.readlines():
            output_file.write(line)