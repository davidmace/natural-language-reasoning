import nltk
import numpy as np
from nltk.stem.porter import *
from collections import defaultdict
import spacy

######################################################################
############################# LOAD BRAIN #############################
######################################################################

def load_brain() :

    spacy_en = spacy.load('en')

    with open('logic-rules1.txt','r') as f :
        lines = f.read().split('\n')

    groups = defaultdict(list)
    category = ''
    for line in lines :
        if line=='' or line[0] == '#' :
            continue
        if line[0] == '=' :
            category = line.lower()[2:]
        else :
            groups[category].append(line.lower())

    # (animal,eat,garbage): True
    observations = {}
    for line in groups['observations'] :
        parts = line.split()
        parts = [stemmer.stem(w) for w in parts]
        observations[(parts[1],parts[2],parts[3])] = (parts[0]=='t')
        
    # red:[orange,yellow,...]
    subclass_neighbors = defaultdict(list)
    subclass_parent = {}
    for line in groups['subclasses'] :
        parts = line.split()
        parts = [stemmer.stem(w) for w in parts]
        classname = stemmer.stem(parts[0].replace(':','').strip())
        for part1 in parts[1:] :
            for part2 in parts[1:] :
                if part1 == part2 :
                    continue
                subclass_neighbors[part1].append(part2)
            subclass_parent[part1] = classname

    # alive:dead
    negations = {}
    for line in groups['negations'] :
        parts = line.split()
        parts = [stemmer.stem(w) for w in parts]
        negations[parts[0]] = parts[1]
        
    # ('animal', -1): ('air', -1, True)
    requirements = defaultdict(list)
    for line in groups['requirements'] :
        parts = line.split()
        set_to = True
        if parts[1][0] == '!' :
            set_to = False
            parts[1] = parts[1][1:]
        position0 = -1
        if parts[0][-1] == ')' :
            position0 = int(parts[0][-2])
            parts[0] = parts[0][:-3]
        position1 = -1
        if parts[1][-1] == ')' :
            position1 = int(parts[1][-2])
            parts[1] = parts[1][:-3]
        requirements[(stemmer.stem(parts[0]),position0)].append( (stemmer.stem(parts[1]),position1,set_to) )

    # fit:size
    reasons = {}
    for line in groups['reasons'] :
        parts = line.split()
        reasons[parts[0]] = parts[1]

    # size:airplane : 0.9
    scales = {}
    for line in groups['scales'] :
        parts = line.split()
        quality = parts[0].replace(':','')
        for i in range(1,len(parts)) :
            key = quality+':'+parts[i]
            scales[key] = 1.0-(1.0*i/len(parts))

    # (human,jump) : (height,gate)
    limits = {}
    for line in groups['limits'] :
        parts = line.split()
        limits[(parts[0],parts[1])] = (parts[2],parts[3])

    rules = {
        'observations': observations,
        'subclass_neighbors': subclass_neighbors,
        'subclass_parent': subclass_parent,
        'negations': negations,
        'requirements': requirements,
        'reasons': reasons,
        'scales': scales,
        'limits':limits,
        'spacy_en': spacy_en
    }

    return rules



######################################################################
########################### PART OF SPEECH ###########################
######################################################################

def parts_of_speech(text,rules) :
    doc = rules['spacy_en']( unicode(text) )
    pos = {}
    for word in doc:
        w = word.text.lower()
        pos[w] = word.tag_[0]
    return pos



######################################################################
########################### REDUCE GRAMMAR ###########################
######################################################################

def reduce_sentence(words, pos) :

    # make linked list with type labels for string
    prev_mem = None
    for i in range(len(words)-1,-1,-1) :
        word = words[i]
        mem = {'type':pos[word], 'val':word, 'next':prev_mem}
        prev_mem = mem
    root = mem
        
    # reductions
    for i in range(10) :
        changed = False
        mem = root
        for d in range(20) :
            next_mem = mem['next']
            if next_mem == None :
                break
            if mem['type']=='D' and next_mem['type']=='N' :
                mem['type'] = 'N'
                mem['val'] = next_mem['val']
                mem['next'] = next_mem['next']
                changed = True
            if mem['type']=='I' and next_mem['type']=='N' :
                mem['type'] = 'N'
                mem['val'] = next_mem['val']
                mem['next'] = next_mem['next']
                changed = True
            mem = next_mem
        if changed==False :
            break

    mem = root
    words = []
    for d in range(20) :
        if mem == None :
            break
        words.append(mem['val'])
        mem = mem['next']
        
    return words


def reduce_linguistics(words) :
    specials = ['rather','be','or','than']
    form = [word if word in specials else pos[word] for word in words]
    if form == ['N','J'] :
        return reduce_logic(words[0],words[1])
    if form == ['N','V','N'] :
        return reduce_logic(words[0],words[1],words[2])
    return -1


def reduce_logic_compare(w1,w2,w3,w4,w5,w6) :
    return ( reduce_logic(w1,w2,w3) > reduce_logic(w1,w2,w3) )

def reduce_logic(w1,w2,w3) :
    if (w1,w2,w3) in observations :
        return observations[(w1,w2,w3)]



######################################################################
########################### REDUCE WORDS #############################
######################################################################


stemmer = PorterStemmer()
specials = {'person':'human', 'people':'human'}
def reduce_words(w0,w1,w2) :
    w0 = stemmer.stem(w0)
    w1 = stemmer.stem(w1)
    w2 = stemmer.stem(w2)
    if w0 in specials :
        w0 = specials[w0]
    if w1 in specials :
        w1 = specials[w1]
    if w2 in specials :
        w2 = specials[w2]
    return w0,w1,w2


######################################################################
########################## LOGIC REDUCTIONS ##########################
######################################################################

def reduce_quality(quality_and_word, rules) :
    if quality_and_word in rules['scales'] :
        val = rules['scales'][quality_and_word]
        return val

    edges = rules['requirements'][(quality_and_word,-1)]
    if len(edges)==1 :
        next_quality_and_word = edges[0][0]
        return reduce_quality(next_quality_and_word,rules)

    return None


def check_reductions(w0,w1,w2, rules) :

    if (w0,w1) in rules['limits'] :
        (quality,word) = rules['limits'][(w0,w1)]
        upper_scale = rules['scales'][quality+':'+word]
        w2_scale = reduce_quality(quality+':'+w2,rules)
        return w2_scale >= upper_scale

    if w1 in rules['reasons'] :
        quality = rules['reasons'][w1]
        quality_and_word0 = quality+':'+w0
        quality_and_word2 = quality+':'+w2
        val1 = reduce_quality(quality_and_word0,rules)
        val2 = reduce_quality(quality_and_word2,rules)
        return val1 > val2

    if w1 in rules['negations'] :
        valid = (check_reductions(w0,rules['negations'][w1],w2, rules) == False)
        if valid != None :
            return valid

    if w0 in rules['subclass_parent'] :
        valid = check_reductions(rules['subclass_parent'][w0],w1,w2, rules)
        if valid != None :
            return valid

    if w2 in rules['subclass_parent'] :
        valid = check_reductions(w0,w1,rules['subclass_parent'][w2], rules)
        if valid != None :
            return valid

    return None


######################################################################
########################### OBSERVATIONS #############################
######################################################################

def check_observations(w0,w1,w2, rules) :
    w0_options = []
    word = w0
    for d in range(10) :
        w0_options.append(word)
        if word not in rules['subclass_parent'] :
            break
        word = rules['subclass_parent'][word]
    
    w2_options = []
    word = w2
    for d in range(10) :
        w2_options.append(word)
        if word not in rules['subclass_parent'] :
            break
        word = rules['subclass_parent'][word]

    for word0 in w0_options :
        for word2 in w2_options :
            if (word0,w1,word2) in rules['observations'] :
                return rules['observations'][(word0,w1,word2)]
    return None


######################################################################
########################## REQUIREMENTS ##############################
######################################################################

def explore(word,place,val,lemmas,rules) :
    if (word,place) in lemmas and lemmas[(word,place)]==val :
        return True
    if (word,place) in lemmas and lemmas[(word,place)]!=val :
        return False
    lemmas[(word,place)] = val # save value

    # loop all requirement edges
    all_valid = True
    for (next_word,next_place,next_val) in rules['requirements'][(word,-1)] + rules['requirements'][(word,place)] :
        if next_place == -1 :
            next_place = place
        valid = explore(next_word,next_place,next_val,lemmas,rules)
        all_valid = (all_valid and valid)
    
    # traverse class parent edge
    if word in rules['subclass_parent'] :
        next_word = rules['subclass_parent'][word]
        next_place = place
        next_val = val
        valid = explore(next_word,next_place,next_val,lemmas,rules)
        all_valid = (all_valid and valid)

    # negations
    if word in rules['negations'] :
        next_word = rules['negations'][word]
        next_place = place
        next_val = (val == False)
        valid = explore(next_word,next_place,next_val,lemmas,rules)
        all_valid = (all_valid and valid)

    # subclass neighbors
    for next_word in rules['subclass_neighbors'][word] :
        next_place = place
        next_val = (val == False)
        valid = explore(next_word,next_place,next_val,lemmas,rules)
        all_valid = (all_valid and valid)
        
    return all_valid


def check_requirements(w0,w1,w2,rules) :
    lemmas = {}
    val = explore(w0,0,True,lemmas,rules) 
    val = val and explore(w1,1,True,lemmas,rules)
    val = val and explore(w2,2,True,lemmas,rules)
    return val


######################################################################
############################### ANSWERS ##############################
######################################################################

def get_answer(s) :
    print s
    
    words = nltk.word_tokenize(s)
    words = filter(lambda w: w!='?',words[1:])
    pos = parts_of_speech(s,rules)
    
    words = reduce_sentence(words,pos)
    
    if len(words) == 3 :
        w0,w1,w2 = words[0], words[1], words[2]
    elif pos[words[1]] == 'V' :
        w0,w1,w2 = words[0], words[1], ''
    else :
        w0,w1,w2 = words[0], '', words[1]
    print w0,'|',w1,'|',w2
    
    (w0,w1,w2) = reduce_words(w0,w1,w2)
    
    answer = check_reductions(w0,w1,w2,rules)
    if answer != None :
        return answer
    
    answer = check_observations(w0,w1,w2,rules)
    if answer != None :
        return answer
    
    answer = check_requirements(w0,w1,w2,rules)
    if answer != None :
        return answer
    
    return None














