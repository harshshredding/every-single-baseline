disease_uis = set()
with open('umls_disease_ids.lst', 'r') as f:
    for line in f.readlines():
        line = line.strip()
        disease_uis.add(line)
output_set = set()
with open('MRCONSO.RRF', 'r') as input_file:
    for line in input_file.readlines():
        split_line = line.split('|')
        umls_id = split_line[0].strip()
        lang = split_line[1].strip()
        atom_string = split_line[14].strip()
        if (umls_id in disease_uis) and (lang == 'ENG' or lang == 'SPA'):
            output_set.add(atom_string)
with open('gazetteers/umls_disease_gazetteer_new.lst', 'w') as output_file:
    for term in output_set:
        print(term, file=output_file)