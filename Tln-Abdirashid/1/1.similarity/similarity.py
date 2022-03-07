from io import open
import csv
import re
DEFINITIONS= "utils/tlndicaro_1.1_defs - Foglio1.csv"
STOP_WORDS="utils/stop_words_FULL.txt"


def load_def(path):
    result=[]
    with open(DEFINITIONS, newline='') as csvfile:
        r = csv.reader(csvfile)
        for i,row in enumerate(r):
            #result.append(ele)
            #x=' '.join(row)
            if i!=0:
                result.append(
                    {'Partecipante': row[0],
                     'Courage':row[1],
                     'Paper':row[2],
                     'Apprehension':row[3],
                     'Sharpener':row[4]}
                )
    return result


def load_stopwords(STOP_WORDS):
    result=[]
    with open(STOP_WORDS,'r') as f:
        lines=f.readlines()
    for line in lines:
        line=line.strip('\n')
        result.append(line)

    return result



def get_dict_items(list,string):
    li=[]
    if string=='courage':
        for i in list:
            for l in i.keys():
                if l=='Courage' and i.get(l):
                    li.append(i.get(l))

    if string=='paper':
        for i in list:
            for l in i.keys():
                if l=='Paper' and i.get(l):
                    li.append(i.get(l))

    if string=='apprehension':
        for i in list:
            for l in i.keys():
                if l=='Apprehension' and i.get(l):
                    li.append(i.get(l))
    if string=='sharpener':
        for i in list:
            for l in i.keys():
                if l=='Sharpener' and i.get(l):
                    li.append(i.get(l))


    return li

def remove_stopwords(list,string,stop_words):
    l=[]
    if string=='courage':

        for i,phrase in enumerate(list):
            ph = re.split(',| ', phrase)
            phr=[]
            for word in ph:
                #print(i)

                if word not in stop_words:
                    phr.append(word)

            l.append(
                {'Id':i,
                 'Relevant_words':phr,
                 'Length':len(phr)
                 }
            )

    if string=='paper':
        for i,phrase in enumerate(list):
            ph = re.split(',| ', phrase)
            phr=[]
            for word in ph:
                #print(i)

                if word not in stop_words:
                    phr.append(word)

            l.append(
                {'Id':i,
                 'Relevant_words':phr,
                 'Length':len(phr)
                 }
            )


    if string=='apprehension':
        for i,phrase in enumerate(list):
            ph = re.split(',| ', phrase)
            phr=[]
            for word in ph:
                #print(i)

                if word not in stop_words:
                    phr.append(word)

            l.append(
                {'Id':i,
                 'Relevant_words':phr,
                 'Length':len(phr)
                 }
            )


    if string=='sharpener':
        for i,phrase in enumerate(list):
            ph = re.split(',| ', phrase)
            phr=[]
            for word in ph:
                #print(i)

                if word not in stop_words:
                    phr.append(word)

            l.append(
                {'Id':i,
                 'Relevant_words':phr,
                 'Length':len(phr)
                 }
            )

    return l
def intersection(l1,l2):
    for i in l1:

        return list(set(i) & set(l2))

def overlap(list,string):
    overlap_similarity=0
    if string=='courage':
        tot_ann=len(list)
        average_relevant_words=[]
        result=[]
        avg_length=0
        len_overlap=0
        for i in list:
            for l in i.keys():
                if l=='Relevant_words':
                    if len(average_relevant_words)<1:
                        average_relevant_words.append(i.get(l))
                    else:
                        overlap=intersection(average_relevant_words,i.get(l))
                        #print(overlap)
                        if len(overlap)>len_overlap:

                            len_overlap=len(overlap)
                            result.append(overlap)

                if l=='Length':
                    value=i.get(l)
                    avg_length+=value
        avg_length=avg_length/tot_ann

        # overlap similarity = length of overlap list / length of average annotated line
        overlap_similarity= len_overlap / avg_length
        result=result[-1]


    elif string=='paper':
        tot_ann=len(list)
        average_relevant_words=[]
        result=[]
        avg_length=0
        len_overlap=0
        for i in list:
            for l in i.keys():
                if l=='Relevant_words':
                    if len(average_relevant_words)<1:
                        average_relevant_words.append(i.get(l))
                    else:
                        overlap=intersection(average_relevant_words,i.get(l))
                        #print(overlap)
                        if len(overlap)>len_overlap:

                            len_overlap=len(overlap)
                            result.append(overlap)

                if l=='Length':
                    value=i.get(l)
                    avg_length+=value
        avg_length=avg_length/tot_ann

        # overlap similarity = length of overlap list / length of average annotated line
        overlap_similarity= len_overlap / avg_length
        result=result[-1]

    elif string=='apprehension':
        tot_ann=len(list)
        average_relevant_words=[]
        result=[]
        avg_length=0
        len_overlap=0
        for i in list:
            for l in i.keys():
                if l=='Relevant_words':
                    if len(average_relevant_words)<1:
                        average_relevant_words.append(i.get(l))
                    else:
                        overlap=intersection(average_relevant_words,i.get(l))
                        #print(overlap)
                        if len(overlap)>len_overlap:

                            len_overlap=len(overlap)
                            result.append(overlap)
                            print(result,len_overlap)

                if l=='Length':
                    value=i.get(l)
                    avg_length+=value
        avg_length=avg_length/tot_ann

        # overlap similarity = length of overlap list / length of average annotated line
        overlap_similarity= len_overlap / avg_length
        #result=result[-1]



    elif string=='sharpener':
        tot_ann=len(list)
        average_relevant_words=[]
        result=[]
        avg_length=0
        len_overlap=0
        for i in list:
            for l in i.keys():
                if l=='Relevant_words':
                    if len(average_relevant_words)<1:
                        average_relevant_words.append(i.get(l))
                    else:
                        overlap=intersection(average_relevant_words,i.get(l))
                        #print(overlap)
                        if len(overlap)>len_overlap:

                            len_overlap=len(overlap)

                            result.append(overlap)
                if l=='Length':
                    value=i.get(l)
                    avg_length+=value
        avg_length=avg_length/tot_ann

        # overlap similarity = length of overlap list / length of average annotated line
        overlap_similarity= len_overlap / avg_length
        result=result[-1]



    return overlap_similarity,result

def calculate_similarity(list,stop_words):
    # WORD1 = COURAGE
    courage_list=get_dict_items(list,'courage')
    clean_courage=remove_stopwords(courage_list,'courage',stop_words)
    calculate_courage_overlap,c_overlap_list=overlap(clean_courage,'courage')
    #print(calculate_courage_overlap)

    # WORD2 = PAPER
    paper_list=get_dict_items(list,'paper')
    clean_paper=remove_stopwords(paper_list,'paper',stop_words)
    calculate_paper_overlap,p_overlap_list=overlap(clean_paper,'paper')
    #print(calculate_paper_overlap)

    # WORD3 = APPREHENSION
    apprehension_list=get_dict_items(list,'apprehension')
    clean_apprehension=remove_stopwords(apprehension_list,'apprehension',stop_words)
    calculate_apprehension_overlap,a_overlap_list=overlap(clean_apprehension,'apprehension')
    #print(calculate_apprehension_overlap)

    # WORD4 = SHARPENER
    sharpener_list=get_dict_items(list,'sharpener')
    clean_sharpener=remove_stopwords(sharpener_list,'sharpener',stop_words)
    calculate_sharpener_overlap,s_overlap_list=overlap(clean_sharpener,'sharpener')
    #print(calculate_sharpener_overlap)

    print('Results:')
    print('*'*100)
    print('Similarity for the word "COURAGE": {}'.format(calculate_courage_overlap))
    print('Overlap list for "COURAGE": {}'.format(c_overlap_list))
    print('')
    print('Similarity for the word "PAPER": {}'.format(calculate_paper_overlap))
    print('Overlap list for "PAPER": {}'.format(p_overlap_list))
    print('')
    print('Similarity for the word "APPREHENSION": {}'.format(calculate_apprehension_overlap))
    print('Overlap list for "APPREHENSION": {}'.format(a_overlap_list))
    print('')
    print('Similarity for the word "SHARPENER": {}'.format(calculate_sharpener_overlap))
    print('Overlap list for "SHARPENER": {}'.format(s_overlap_list))



def main():

    # courage=generico astratto       ------> GA
    # paper=generico concreto         ------> GC
    # Apprehension=specifico astratto ------> SA
    # Sharpener=specifico concreto    ------> SC

    definitions_ann=load_def(DEFINITIONS)
    stop_words=load_stopwords(STOP_WORDS)
    similarity_GA=calculate_similarity(definitions_ann,stop_words)
    #print(stop_words)





if __name__=='__main__':
    main()