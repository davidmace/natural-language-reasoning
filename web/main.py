from flask import Flask, render_template, request
from collections import defaultdict
from nltk.stem.porter import *
import pattern.en
import spacy
import nltk
import string

######################################################################
########################## LOAD RULES ##########################
######################################################################

def stem_quality(s,stemmer) :
  words = s.split(':')
  return stemmer.stem(words[0])+':'+stemmer.stem(words[1])


def load_rules() :

  stemmer = PorterStemmer()

  with open('logic-rules.txt','r') as f :
    lines = f.read().split('\n')

  # parse file into sections
  groups = defaultdict(list)
  category = ''
  for line in lines :
    if line=='' or line[0] == '#' :
      continue
    if line[0] == '=' :
      category = line.lower()[2:]
    else :
      groups[category].append(line.lower())

  # (animal,eat,garbage) : True
  observations = {}
  for line in groups['observations'] :
    parts = line.split()
    parts = [stemmer.stem(w) for w in parts]
    observations[(parts[1],parts[2],parts[3])] = (parts[0]=='t')
  
  # red : [orange,yellow,...]
  subclass_parents = {}
  for line in groups['subclasses'] :
    parts = line.split()
    classname = stemmer.stem(parts[0].replace(':','').strip())
    members = [stemmer.stem(w) for w in parts[1:]]
    for part1 in members :
      for part2 in members :
        if part1 == part2 :
          continue
        #subclass_neighbors[part1].append(part2)
      subclass_parents[part1] = classname

  # alive : dead
  negations = {}
  for line in groups['negations'] :
    parts = line.split()
    parts = [stemmer.stem(w) for w in parts]
    negations[parts[0]] = parts[1]
      
  # ('animal', -1) : ('air', -1, True)
  requirements = {}
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

    word_in = stemmer.stem(parts[0])
    word_out = stemmer.stem(parts[1])
    if (word_in,position0) not in requirements :
      requirements[(word_in,position0)] = []
    requirements[(word_in,position0)].append( (word_out,position1,set_to) )

  # fit : size
  reasons = {}
  for line in groups['reasons'] :
    parts = line.split()
    parts = [stemmer.stem(w) for w in parts]
    reasons[parts[0]] = parts[1]

  # size:airplane : 0.9
  scales = {}
  scale_options = defaultdict(list)
  for line in groups['scales'] :
    parts = line.split()
    quality = stemmer.stem(parts[0].replace(':',''))
    for i in range(1,len(parts)) :
      word = stemmer.stem(parts[i])
      key = quality+':'+word
      scales[key] = 1.0-(1.0*i/len(parts))
      scale_options[quality].append(word)

  # (human,jump) : height:gate
  limits = {}
  for line in groups['limits'] :
    parts = line.split()
    stemmed_quality = stem_quality(parts[2],stemmer)
    limits[(stemmer.stem(parts[0]),stemmer.stem(parts[1]))] = stemmed_quality

  tokenize = spacy.load('en') # tokenize is a function

  verb_types = {'fly':'regular',
              'speak':'regular',
              'eat':'regular',
              'sink':'passive',
              'throw':'active',
              'smash':'passive',
              'jump':'active',
              'lift':'active',
              'hold':'active',
              'carry':'active'}

  Z = {
    'observations':observations,
    'negations':negations,
    'scales':scales,
    'scale_options':scale_options,
    'reasons':reasons,
    'limits':limits,
    'subclass_parents':subclass_parents,
    'requirements':requirements,
    'verb_types':verb_types,
    'stemmer':stemmer,
    'tokenize':tokenize
  }

  return Z


######################################################################
########################## LOGIC REDUCTIONS ##########################
######################################################################

# turn quality:word into real value
# NOTE: this is written oddly because you only need to traverse one edge
def reduce_quality(quality_and_word,Z) :
  if Z['DEBUG'] :
    print 'reduce_quality', quality_and_word

  # Leaf: weight:car -> 0.3
  if quality_and_word in Z['scales'] :
    val = Z['scales'][quality_and_word]
    return val

  # Transition: weight:unicorn -> weight:car
  if (quality_and_word,-1) in Z['requirements'] :
    base_key = Z['requirements'][(quality_and_word,-1)][0]
    next_quality_and_word = base_key[0]
    return reduce_quality(next_quality_and_word,Z)

  return None

# Reduce (w0,w1,w2) to active or passive very comparison.
# NOTE: this is inefficient because I have to check every pairwise w1,w2 path
# rather than each individually
def check_reductions(w0,w1,w2,Z) :
  if Z['DEBUG'] :
    print 'check_reductions',w0,w1,w2

  # Leaf: active verb
  # try to get quality:w2 to real value
  if (w0,w1) in Z['limits'] :
    quality_and_word = Z['limits'][(w0,w1)]
    upper_scale = Z['scales'][quality_and_word]
    quality = Z['reasons'][w1]
    w2_scale = reduce_quality(quality+':'+w2,Z)
    if w2_scale != None :
      return w2_scale <= upper_scale

  # Leaf: passive verb
  # try to get both quality:word to real value
  if w1 in Z['reasons'] :
    quality = Z['reasons'][w1]
    quality_and_word0 = quality+':'+w0
    quality_and_word2 = quality+':'+w2
    val1 = reduce_quality(quality_and_word0,Z) # get real value corresponding to quality:word
    val2 = reduce_quality(quality_and_word2,Z) 
    if val1 != None and val2 != None :
      return val1 > val2

  # Transition: w1,v,w2 -> !(w1,!v,w2)
  if w1 in Z['negations'] :
    valid = check_reductions(w0,Z['negations'][w1],w2,Z)
    if valid != None :
      return (valid==False)

  # Transition: w1,v,w2 -> (parent[w1],v,w2)
  if w0 in Z['subclass_parents'] :
    valid = check_reductions(Z['subclass_parents'][w0],w1,w2,Z)
    if valid != None :
      return valid

  # Transition: w1,v,w2 -> (w1,v,parent[w2])
  if w2 in Z['subclass_parents'] :
    valid = check_reductions(w0,w1,Z['subclass_parents'][w2],Z)
    if valid != None :
      return valid

  return None


######################################################################
########################### OBSERVATIONS #############################
######################################################################

# check if (w0,w1,w2) is stored in observations as True/False
# and for w0's parent classes and w2's parent classes
def check_observations(w0,w1,w2,Z) :

  # get list of subclasses of w0
  w0_options = []
  word = w0
  for d in range(10) :
    w0_options.append(word)
    if word not in Z['subclass_parents'] :
      break
    word = Z['subclass_parents'][word]
  
  # get list of subclasses of w2
  w2_options = []
  word = w2
  for d in range(10) :
    w2_options.append(word)
    if word not in Z['subclass_parents'] :
      break
    word = Z['subclass_parents'][word]

  # check if any of the subclass pairs like (sub[w0],w1,sub[w2]) is in observations
  for word0 in w0_options :
    for word2 in w2_options :
      if Z['DEBUG'] :
        print 'observations',(word0,w1,word2)
      if (word0,w1,word2) in Z['observations'] :
        return Z['observations'][(word0,w1,word2)]

  # no observation found
  return None


######################################################################
########################## REQUIREMENTS ##############################
######################################################################

def explore(word,place,val,lemmas,visited,Z) :
  if Z['DEBUG'] :
    print 'explore',(word,place,val,lemmas)

  # for each of the three original words, can only visit a state once
  if (word,place) in visited :
    return None

  # this lemma contradicts a previously stored lemma
  if (word,place) in lemmas and lemmas[(word,place)]==val :
    if Z['DEBUG'] :
      print 'explore True',(word,place)
    return True
  if (word,place) in lemmas and lemmas[(word,place)]!=val :
    if Z['DEBUG'] :
      print 'explore False',(word,place)
    return False

  # save current lemma
  lemmas[(word,place)] = val
  visited.add((word,place))

  all_valid = None

  # Transition: requirements
  # word,position,value -> new_word,new_position,new_value
  reqs = []
  if (word,-1) in Z['requirements'] :
    reqs += Z['requirements'][(word,-1)] # -1 means any role
  if (word,place) in Z['requirements'] :
    reqs += Z['requirements'][(word,place)]
  for (next_word,next_place,next_val) in reqs :
    if next_place == -1 : # -1 means the current role
      next_place = place
    valid = explore(next_word,next_place,next_val,lemmas,visited,Z)
    all_valid = valid if all_valid==None else (all_valid and valid) # False > True > None
  
  # Transition: subclass parents
  # word,position,value -> parent,position,value
  if word in Z['subclass_parents'] :
    next_word = Z['subclass_parents'][word]
    next_place = place
    next_val = val
    valid = explore(next_word,next_place,next_val,lemmas,visited,Z)
    all_valid = valid if all_valid==None else (all_valid and valid) # False > True > None

  # Transition: negations
  # word,position,value -> neg,position,~value
  if word in Z['negations'] :
    next_word = Z['negations'][word]
    next_place = place
    next_val = (val == False)
    valid = explore(next_word,next_place,next_val,lemmas,visited,Z)
    all_valid = valid if all_valid==None else (all_valid and valid) # False > True > None

  # subclass neighbors
  #for next_word in subclass_neighbors[word] :
  #    next_place = place
  #    next_val = (val == False)
  #    valid = explore(next_word,next_place,next_val,lemmas)
  #    all_valid = valid if all_valid==None else (all_valid and valid)
      
  return all_valid

# Traverse a graph of rules and find if any of the words have paths to
# lemmas that contradict each other.
def check_requirements(w0,w1,w2,Z) :
  lemmas = {}
  val0 = explore(w0,0,True,lemmas,set([]),Z) 
  val1 = explore(w1,1,True,lemmas,set([]),Z)
  val2 = explore(w2,2,True,lemmas,set([]),Z)

  # False > True > None
  if Z['DEBUG'] :
    print 'requirements',val0,val1,val2
  if False in (val0,val1,val2) :
    return False
  if True in (val0,val1,val2) :
    return True
  return None
    


def answer_question(w0,w1,w2,Z) :
  answer = check_reductions(w0,w1,w2,Z)
  if Z['DEBUG'] :
    print 'check_reductions returns',answer
  if answer != None :
    return answer
  
  answer = check_observations(w0,w1,w2,Z)
  if Z['DEBUG'] :
    print 'check_observations returns',answer
  if answer != None :
    return answer
  
  answer = check_requirements(w0,w1,w2,Z)
  if Z['DEBUG'] :
    print 'check_requirements returns',answer
  if answer != None :
    return answer
  
  return None









######################################################################
############################### REASONS ##############################
######################################################################


def load_reasons() :
  reason_formulas = {}
  syntax_words = set([])

  with open('reason-formulas.txt','r') as f :
      lines = filter(lambda l: l.strip()!='' and l[0]!='#', f.read().split('\n'))
      
  for line in lines :
      
      # parse sentence and logic formula
      sent = ' '+line[:line.find('|')].strip().lower()+' '
      logic = line[line.find('|')+1:].strip().lower()+' .' # put a . to handle occasional third word

      this_syntax_words = sent.split()
      
      # handle special word reductions
      sent = sent.replace(' an ',' a ')
      sent = sent.replace(' the ',' a ')
      sent = sent.replace(" can't ",' cannot ')
      sent = sent.replace(" doesn't ",' cannot ')
      sent = sent.replace(" don't ",' cannot ')
      sent = sent.replace(" does ",' can ')
      sent = sent.replace(" do ",' can ')
      sent = sent.replace(" on ",' in ')
      sent = sent.replace(" to ",' in ')
      sent = sent.replace(" at ",' in ')
      sent = sent.replace(" from ",' in ')
      sent = sent.replace(" off ",' in ')
      sent = sent.replace(" through ",' in ')

      # extract words from formula on right
      logic_parts = logic.split(' ')
      type = logic_parts[0]
      logic_words = [logic_parts[1],logic_parts[2],logic_parts[3]]
      
      # extract logic value
      logic_values = [True,True,True]
      for i in range(3) :
          if len(logic_words[i])>1 and logic_words[i][0]=='!' :
              logic_values[i] = False
              logic_words[i] = logic_words[i][1:]
      
      # extract position id
      logic_pos = [-1,-1,-1]
      for i in range(3) :
          if len(logic_words[i])>2 and logic_words[i][-2]=='-' :
              logic_pos[i] = int(logic_words[i][-1])
              logic_words[i] = logic_words[i][:-2]
      
      # handle plural vs singular words
      plural_values = [False,False,False]
      formula = sent
      for i in range(3) :
          plural = pattern.en.pluralize(logic_words[i])
          if plural!=logic_words[i] and ' '+plural+' ' in formula :
              formula = formula.replace(' '+plural+' ',' $'+str(i)+' ')
              #plural_values[i] = True
          else :
              formula = formula.replace(' '+logic_words[i]+' ',' $'+str(i)+' ')
              #plural_values[i] = False
      


      syntax = filter(lambda c: c not in ['0','1','2','3','4','5','6','7','8','9','$'], formula)
      syntax = set(syntax.strip().split())
      syntax_words = set.union(syntax_words,syntax)
      
      # "one cannot $1 in a $0" -> "one cannot $ in a $", "10"
      formula_unordered = filter(lambda c: c not in set(['0','1','2','3','4','5','6','7','8','9']), formula)
      order = filter(lambda c: c in ['0','1','2','3','4','5','6','7','8','9'], formula)

      reason_formulas[formula_unordered] = (type, order, logic_values,logic_pos)

  Y = {'reason_formulas':reason_formulas, 'syntax_words':syntax_words}
  return Y



def process_reason(sent,Z,Y) :
  sent = ' '+filter(lambda c: str.isalpha(c) or c == ' ', sent)+' '
  
  # handle special word reductions
  sent = sent.replace(' an ',' a ')
  sent = sent.replace(' the ',' a ')
  sent = sent.replace(" can't ",' cannot ')
  sent = sent.replace(" doesn't ",' cannot ')
  sent = sent.replace(" don't ",' cannot ')
  sent = sent.replace(" does ",' can ')
  sent = sent.replace(" do ",' can ')
  sent = sent.replace(" on ",' in ')
  sent = sent.replace(" to ",' in ')
  sent = sent.replace(" at ",' in ')
  sent = sent.replace(" from ",' in ')
  sent = sent.replace(" off ",' in ')
  sent = sent.replace(" through ",' in ')

  words = list(set(sent.split()) - Y['syntax_words'])
  formula = sent

  pos = []
  for i in range(len(words)) :
    pos.append( (formula.find(' '+words[i]+' '), words[i]) )
  pos.sort()

  ordered_words = [x[1] for x in pos]

  for i in range(len(ordered_words)) :
    formula = formula.replace(' '+ordered_words[i]+' ',' $ ')
  
  if Y['DEBUG'] == True :
    print formula

  if formula not in Y['reason_formulas'] :
    return False
  (type, order, not_values, pos_values) = Y['reason_formulas'][formula]

  stemmed_words = [Z['stemmer'].stem(w) for w in ordered_words]
  w1 = stemmed_words[int(order[0])]
  w2 = stemmed_words[int(order[1])]
  w3 = stemmed_words[int(order[2])] if len(order)==3 else ''

  if Y['DEBUG'] == True :
    print (type, order, not_values, pos_values)
    print w1,w2,w3

  if type == 'sub' :
    Z['subclass_parents'][w1] = w2 # this doesnt account for multiple parents
    #for child in subclass_children[w1] :
    #    subclass_neighbors[child].append(w1)
    #    subclass_neighbors[w1].append(child)
  elif type == 'req' :
    if (w1,pos_values[0]) not in Z['requirements'] :
      Z['requirements'][(w1,pos_values[0])] = []
    Z['requirements'][(w1,pos_values[0])].append( (w2,pos_values[1],not_values[1]) )
  elif type == 'neg' :
    Z['negations'][w1] = w2
  elif type == 'obs' :
    Z['observations'][(w1,w2,w3)] = (not_values[1])
      
  return True

















######################################################################
############################ DIALOG SYNTAX ###########################
######################################################################

def parse_question(msg,Z) :
  syntax_words = set(['can','could','might','will','may','do','does','did','is','are'])
  tok = Z['tokenize'](unicode(msg))
  words = filter(lambda w: w.pos_ in ('VERB','NOUN','ADJ','ADV') and w.text not in syntax_words, tok)
  return (words[0].text,words[1].text,words[2].text)


def phrase_question(cur_question,Z) :
  type = cur_question[0]
  w = cur_question[1]
  v = cur_question[2]
  if type == 'reason' :
    return "I don't know. Explain the answer to me."
  quality = Z['reasons'][v]
  options = ','.join(Z['scale_options'][quality])
  if type == 'scale' :
    return "What is closest to the %s of %s? (%s)" % (quality,w,options)
  if type == 'limit' :
    return "What is the max %s a %s can %s? (%s)?" % (quality,w,v,options)
  return ''

def process_reasons(s,Z,Y) :
  sentences = nltk.sent_tokenize(s)
  for s in sentences :
    ans = process_reason(s,Z,Y)
    if ans == False :
      return False
  return True

# msg is the option chosen, w is the word asked about
def process_answer_scale(msg,w,v,Z) :
  msg = Z['stemmer'].stem(msg)
  quality = Z['reasons'][v]
  options = Z['scale_options'][quality]
  if msg in options :
    if (quality+':'+w,-1) not in Z['requirements'] :
      Z['requirements'][(quality+':'+w,-1)] = []
    Z['requirements'][(quality+':'+w,-1)].append( (quality+':'+msg,-1,True) )
    return True
  return False

def process_answer_limit(msg,w,v,Z) :
  msg = Z['stemmer'].stem(msg)
  quality = Z['reasons'][v]
  options = Z['scale_options'][quality]
  if msg in options :
    Z['limits'][(w,v)] = quality+':'+msg
    return True
  return False

specials = {'person':'human', 'people':'human'}
def reduce_words(w0,w1,w2,Z) :
    w0 = Z['stemmer'].stem(w0)
    w1 = Z['stemmer'].stem(w1)
    w2 = Z['stemmer'].stem(w2)
    if w0 in specials :
        w0 = specials[w0]
    if w1 in specials :
        w1 = specials[w1]
    if w2 in specials :
        w2 = specials[w2]
    return w0,w1,w2




app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route('/')
def hello():
  global save
  save = {'questions': [], 'cur_question': ()}
  return render_template('main.html')


@app.route('/msg')
def message():
  msg = request.args.get('msg')
  return message(msg,Z,Y,save)

def message(msg,Z,Y,save) :
  msg = str(msg).lower()

  # Had asked for a new question
  if save['cur_question'] == () :
    msg = filter(lambda c: str.isalpha(c) or c == ' ', msg)
    (w1,v,w2) = parse_question(msg,Z)
    (w1,v,w2) = reduce_words(w1,v,w2,Z)

    answer = answer_question(w1,v,w2,Z)
    if answer == True :
      return 'Yes.| Ask me question.'
    if answer == False :
      return 'No.| Ask me question.'

    elif v not in Z['verb_types'] or Z['verb_types'][v] == 'regular' :
      save['questions'].append(('reason',w1,v,w2))

    elif Z['verb_types'][v] == 'passive' :
      if (Z['reasons'][v]+':'+w2,-1) not in Z['requirements'] and Z['reasons'][v]+':'+w2 not in Z['scales'] :
        save['questions'].append(('scale',w2,v))
      if (Z['reasons'][v]+':'+w1,-1) not in Z['requirements'] and Z['reasons'][v]+':'+w1 not in Z['scales'] :
        save['questions'].append(('scale',w1,v))
    
    elif Z['verb_types'][v] == 'active' :
      if (w1,v) not in Z['limits'] :
        save['questions'].append(('limit',w1,v,w2))
      if (Z['reasons'][v]+':'+w2,-1) not in Z['requirements'] and Z['reasons'][v]+':'+w2 not in Z['scales'] :
        save['questions'].append(('scale',w2,v))

	# Had asked for a reason explanation
  elif save['cur_question'][0] == 'reason' :
    successful = process_reasons(msg,Z,Y)
    if successful == False :
      return "I don't understand. Please try again."

	# Had asked a quality question
  elif save['cur_question'][0] == 'scale' :
    successful = process_answer_scale(msg, save['cur_question'][1], save['cur_question'][2], Z)
    if successful == False :
      return "I don't understand. Please select an option above."

	# Had asked an active limit question
  elif save['cur_question'][0] == 'limit' :
    successful = process_answer_limit(msg, save['cur_question'][1], save['cur_question'][2], Z)
    if successful == False :
      return "I don't understand. Please select an option above."

  if len(save['questions']) == 0 :
    save['cur_question'] = ()
    return 'Thanks.|Ask me a question.'

  save['cur_question'] = save['questions'].pop()
  return phrase_question(save['cur_question'], Z)

if __name__ == "__main__":
  global Z,Y,save
  Z = load_rules()
  Z['DEBUG'] = False
  Y = load_reasons()
  Y['DEBUG'] = True
  save = {'questions': [], 'cur_question': ()}
  #message('cat smash dog')
  app.run(port=5001,debug=True)



######################################################################
############################### TESTS ################################
######################################################################

def run_tests() :

  # ANSWER TESTS
  tests = [
    ('raccoon eat garbage',True),
    ('dog eat chocolate',False),
    
    ('bird fly outerspace',False),
    ('bird fly april',True),
    ('dog fly april',False),
    ('bird swim april',None),
    
    ('airplane fit brick',False),
    ('brick sink water',True),
    ('unicorn sink water',None),
    ('water sink unicorn',None),
    
    ('dog jump dog', True),
    ('dog jump house',False),
    ('dog jump unicorn',None),
    ('unicorn jump dog',None)
  ]

  Z = load_rules()
  Z['DEBUG'] = False
  for (q,t) in tests :
    words = q.split()
    w0,w1,w2 = reduce_words(words[0],words[1],words[2],Z)
    ans = answer_question(w0,w1,w2,Z)
    #print '\t',t,ans,w0,w1,w2
    #print ''
    assert(ans==t)


  # PASSIVE VERB TEST
  Z = load_rules()
  Z['DEBUG'] = False
  save = {'questions': [], 'cur_question': ()}

  message('dog smash cat',Z,save)
  message('peanut',Z,save)
  message('train',Z,save)
  ans = message('dog smash cat',Z,save); assert('No' in ans)
  ans = message('cat smash dog',Z,save); assert('Yes' in ans)
  message('cat smash giraffe',Z,save)
  message('peanut',Z,save)
  ans = message('cat smash giraffe',Z,save); assert('Yes' in ans)
  ans = message('dog smash giraffe',Z,save); assert('No' in ans)


  # ACTIVE VERB TEST
  Z = load_rules()
  Z['DEBUG'] = False
  save = {'questions': [], 'cur_question': ()}

  message('unicorn throw wave',Z,save)
  message('peanut',Z,save)
  message('brick',Z,save)
  ans = message('unicorn throw wave',Z,save); assert('Yes' in ans)
  message('unicorn throw gopher',Z,save)
  message('peanut',Z,save)
  ans = message('unicorn throw gopher',Z,save); assert('Yes' in ans)
  message('gopher throw unicorn',Z,save)
  message('train',Z,save)
  message('cellphone',Z,save)
  ans = message('gopher throw unicorn',Z,save); assert('No' in ans)


  # REASON PARSING TESTS
  Z = load_rules()
  Z['DEBUG'] = False
  Y = load_reasons()
  Y['DEBUG'] = True

  process_reason('a radishes is a dogs',Z,Y)
  assert(Z['subclass_parents']['radish']=='dog')

  process_reason('radish is the opposite of onion',Z,Y)
  assert(Z['negations']['radish']=='onion')
      
  process_reason('radish is not done',Z,Y)
  assert((u'done', -1, False) in Z['requirements'][(u'radish', -1)])

  process_reason('a cheetah runs fast',Z,Y)
  assert( Z['observations'][(u'cheetah', u'run', u'fast')]==True )


  # REASON VERB TESTS
  Z = load_rules()
  Z['DEBUG'] = False
  Y = load_reasons()
  Y['DEBUG'] = True
  save = {'questions': [], 'cur_question': ()}

  message('elephant swim april',Z,Y,save)
  message('a elephant cannot swim',Z,Y,save)
  ans = message('elephant swim january',Z,Y,save)
  assert('No' in ans)

  message('frog swim desert',Z,Y,save)
  message('you need water to swim',Z,Y,save)
  message('frog swim desert',Z,Y,save)
  message('a desert has no water',Z,Y,save)
  ans = message('frog swim desert',Z,Y,save)
  assert('No' in ans)








