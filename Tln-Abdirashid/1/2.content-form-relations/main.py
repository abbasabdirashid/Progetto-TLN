from methods import *


def get_info(relational_set,len_all_synsets):

    for relation in relational_set.keys():
        len_keys = len(relational_set[relation].keys())
        print('\nlen of relations of type ', relation.upper(),' : ',len_keys,' of ',len_all_synsets,' ---> ',len_keys/len_all_synsets)
        # for sense in relational_set[relation].keys():
        #     print('sense : ',sense,' words in common :',relational_set[relation][sense])


def main():
    # studio di forma
    # avg, len_all_synsets=get_average_len_def()
    # print("\nAverage length of definitions in Wordnet: {}".format(avg))
    # print()
    # print("Length of all definitions in Worndet: {}".format(len_all_synsets))

    # studio di contenuto
    avg_height= get_avg_height_of_definitions()
    print("\nAverage height of definitions from Wordnet Root: {}".format(avg_height))

    # studio di relazioni
    # relational_set = get_relational_set()
    # get_info(relational_set,len_all_synsets)



if __name__=='__main__':
    main()