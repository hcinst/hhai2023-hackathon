import json, re, math, csv
import regex
import random
import itertools
import numpy
import matplotlib.pyplot as plt
from itertools import cycle

# define colors for painting test polygons (expand this list as needed)
plotcolors=cycle(['blue','orange','green','red','yellow','brown','purple','turquoise','gold',
  'lime','yellowgreen','greenyellow','orangered'])

# display megatile annotations with a different color for each unique player or cohort
# annotation marker size reflects how many individual annotations were combined
def showmega(annotations):
  mypoints = []
  x = []
  y = []
  s = []
  c = []
  cindex = -1
  cplayer = []
  for myrecs in annotations:
    player = myrecs[2]
    if cplayer != player:
      cplayer = player
      color=next(plotcolors)
      cindex += 1
    annotators = myrecs[3]
    mypoints.append([myrecs[0],myrecs[1]])
    x.append(myrecs[0])
    y.append(myrecs[1])
    s.append(myrecs[3] * 50)
    c.append(color)
  plt.scatter(x,y,s=s,c=c)
  ax = plt.gca()
  ax.set_xlim([0, 550])
  ax.set_ylim([0, 550])
  plt.show()

# pixel distance thresholds (based on plaque type) for determining if two annotation
# points on a megatile originating from two different annotators refer to the same plaque.
# these thresholds are used in the functions that count true positives (tp),
# false positives (fp), and false negatives (fn).
dist_cored=15
dist_caa=50

# print list in readable indented hierarchical format
def nicelist(item,level):
    for each_item in item:
        if isinstance(each_item,list):
            nicelist(each_item,level+1)
        else:
            for tabspace in range(level):
                print("\t",end='')
            print(each_item)

# print cohort list nicely
def niceprint(cohorts):
        for each_pass in cohorts:
#            print("\t",'pass')
            for each_marker in each_pass:
#            	print("\t\t",end='')
            	print(each_marker)

# test if two lists have at least one member in common
def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
    if (a_set & b_set):
        return True
    else:
        return False

# return euclidean distance between two points
def fn_euclid(coord1,coord2):
  x1=coord1[0]
  y1=coord1[1]
  x2=coord2[0]
  y2=coord2[1]
  dist = math.sqrt(math.pow(abs(x1-x2),2)+math.pow(abs(y1-y2),2))
  return dist

# count true positives
# for a given plaque type, count number of points within treshold distance
# between test set and ref set
def fn_tp(test_set,ref_set,plaque):
  tally=0
  if plaque=='cored':
    dist=dist_cored
  if plaque=="caa":
    dist=dist_caa
  for coords1 in test_set:
    for coords2 in ref_set:
      if fn_euclid(coords1,coords2) <= dist:
        tally += 1
  return tally

# count false positives
def fn_fp(test_set,ref_set,plaque):
  tally=0
  if plaque=='cored':
    dist=dist_cored
  if plaque=="caa":
    dist=dist_caa
  for coords1 in test_set:
    tally2=0
    for coords2 in ref_set:
      if fn_euclid(coords1,coords2) <= dist:
        tally2 += 1
    if tally2 == 0:
      tally += 1
  return tally

# count false negatives
def fn_fn(test_set,ref_set,plaque):
  tally=0
  if plaque=='cored':
    dist=dist_cored
  if plaque=="caa":
    dist=dist_caa
  for coords1 in ref_set:
    tally2=0
    for coords2 in test_set:
      if coords2[0] != -1 and fn_euclid(coords1,coords2) <= dist:
        tally2 += 1
        #print('tally2',tally2)
    if tally2 == 0:
      tally += 1
  #print('fn tally',tally)
  #print(test_set, ref_set)
  return tally

# calculate F1 score for a given plaque type (determines threshold distance)
# from two sets of points
def fn_f1(test_set,ref_set,plaque):
  tp = fn_tp(test_set,ref_set,plaque)
  fp = fn_fp(test_set,ref_set,plaque)
  fn = fn_fn(test_set,ref_set,plaque)
  if (tp+tp+fp+fn == 0):
    f1 = -1
  else:
    f1 = fn_fscore(tp,fp,fn)
  return f1

# fscore calculation from confusion matrix parameters (tp, fp, and fn)
def fn_fscore(tp,fp,fn):
  if (tp+tp+fp+fn == 0):
    f1 = -1
  else:
    f1 = (tp+tp) / (tp+tp+fp+fn)
  return f1


# -------------------
# START EXECUTION HERE


inputfile='pilot_megatiles_experts.csv'
plaque_types=(['cored','caa'])
experts=[]
artifact_datasets=[]

filer = open(inputfile, 'r')
Lines = filer.readlines()
filer.close()

count = 0
for line in Lines:
    count += 1
    if count > 1:
        p=re.split(r',(?=")', line) # parses by comma but not within quotes
        adnumber = str(int(p[0]))
        artifact_datasets.append(adnumber)

# expert1 in this code represents the gold standard data used for player feedback.
# the index in this code for expert 1 is 0.
# creates "experts" list:
#   expert_index = 0,1 to indicate which expert
#   cindex = 0,1 to indicate which plaque type was annotated
#   artifact_datatype = unique id referring to a specific megatile that was analyzed
#   xc,yc = coordinates of point on megatile where expert clicked on a plaque
count = 0
for line in Lines:
    count += 1
    if count > 1:
        p=re.split(r',(?=")', line) # parses by comma but not within quotes

        artifact_datatype = str(int(p[0]))

        expert1 = p[1]
        expert2 = p[2]

        answer1 = expert1[1:-1]
        answer2 = expert2[1:-2]

        jexpert=[json.loads(answer1),json.loads(answer2)]

        expert_index = -1
        for expert in jexpert:
            expert_index += 1
            for i in expert:
                label=i['label']
                answer=i['answer']
                cindex = plaque_types.index(label)
                for k in answer:
                    xc = k['x'] 
                    yc = k['y']
                    experts.append([expert_index,cindex,artifact_datatype,xc,yc])

# calculate average agreement between two experts 
# on each plaque type and overall using f-score
tp_type=[0,0]
fp_type=[0,0]
fn_type=[0,0]
tp_total = 0
fp_total = 0
fn_total = 0
for ad in artifact_datasets:
    for label in plaque_types:
        annot1=[]
        annot2=[]
        for annotation in experts:
            if (annotation[2] == ad) and (annotation[1] == plaque_types.index(label)):
                if (annotation[0] == 0):
                    annot1.append([annotation[3],annotation[4]])
                if (annotation[0] == 1):
                    annot2.append([annotation[3],annotation[4]])

        tp=fn_tp(annot1,annot2,label)
        fp=fn_fp(annot1,annot2,label)
        fn=fn_fn(annot1,annot2,label)
        f1 = fn_f1(annot1,annot2,label)

        tp_total += tp
        fp_total += fp
        fn_total += fn

        tp_type[plaque_types.index(label)] += tp
        fp_type[plaque_types.index(label)] += fp
        fn_type[plaque_types.index(label)] += fn

f1_cored = fn_fscore(tp_type[0],fp_type[0],fn_type[0])
f1_caa = fn_fscore(tp_type[1],fp_type[1],fn_type[1])
f1_total = fn_fscore(tp_total,fp_total,fn_total)
print('cored:',tp_type[0],fp_type[0],fn_type[0],f1_cored)
print('caa:',tp_type[1],fp_type[1],fn_type[1],f1_caa)
print('total:',tp_total,fp_total,fn_total,f1_total)


# read in all player annotations from data file
# creates "players" list (structure matches "experts" list):
#   player_id = unique id to indicate which player
#   cindex = 0,1 to indicate which plaque type was annotated
#   artifact_datatype = unique id referring to a specific megatile that was analyzed
#   xc,yc = coordinates of point on megatile where expert clicked on a plaque

inputfile='pilot_megatiles_players.csv'
players=[]  
filer = open(inputfile, 'r')
Lines = filer.readlines()
filer.close()
csvFile = csv.reader(Lines, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)

count = 0
for line in csvFile:
    count += 1
    if count > 1:
        artifact_datatype = str(int(line[2]))        
        player_id = line[1]
        uncut_answer = line[4:]
        uncut_answer = str(uncut_answer).replace('\\', '')
        uncut_answer = uncut_answer.replace('\'','')
        uncut_answer = uncut_answer[1:-3]
        label_answer=json.loads(uncut_answer)
        for i in label_answer:
            label=i['label']
            answer=i['answer']
            cindex = plaque_types.index(label)
            for k in answer:
                xc = k['x']
                yc = k['y']
                players.append([player_id,cindex,artifact_datatype,xc,yc])

print('expert annotations:',len(experts))
print('player annotations:',len(players))

# generate a list of all player ids without repeats and excluding player 259
player_list=[]
for i in players:
    if i[0] not in player_list and int(i[0]) != 259: # exclude bad data
        player_list.append(i[0])
print('number of players:',len(player_list))

# compare individual player annotations to both experts (no crowd results here, just
# baseline individual performance)
# show average f-score of all players to each expert for each plaque type
print('***** assess average agreement of all players with each expert (cohort size = 1) *****')
result_rows=0
pilot_results=[]
pilot_results.append(['player_id','expert_id','artifact_datatype_id','plaque_type','tp','fp','fn','f1'])
for expert_index in range(0,2):
    tp_type=[0,0]
    fp_type=[0,0]
    fn_type=[0,0]
    tp_total = 0
    fp_total = 0
    fn_total = 0

    for i in player_list:      
      for ad in artifact_datasets:
        for label in plaque_types:
            expert=[]
            player=[]
            for annotation in experts:
                if (annotation[2] == ad) and (annotation[1] == plaque_types.index(label)):
                    if (annotation[0] == expert_index):
                        expert.append([annotation[3],annotation[4]])
            for annotation in players:
                if (annotation[2] == ad) and (annotation[1] == plaque_types.index(label)):
                    if (annotation[0] == i):
                        player.append([annotation[3],annotation[4]])

            tp=fn_tp(player,expert,label)
            fp=fn_fp(player,expert,label)
            fn=fn_fn(player,expert,label)
            f1 = fn_f1(player,expert,label)

            result_rows += 1
            pilot_results.append([result_rows,i,expert_index,ad,plaque_types.index(label),tp,fp,fn,f1])

            tp_total += tp
            fp_total += fp
            fn_total += fn

            tp_type[plaque_types.index(label)] += tp
            fp_type[plaque_types.index(label)] += fp
            fn_type[plaque_types.index(label)] += fn
    print('compared to expert', expert_index)
    f1_cored = fn_fscore(tp_type[0],fp_type[0],fn_type[0])
    f1_caa = fn_fscore(tp_type[1],fp_type[1],fn_type[1])
    f1_total = fn_fscore(tp_total,fp_total,fn_total)
    print('cored:',tp_type[0],fp_type[0],fn_type[0],f1_cored)
    print('caa:',tp_type[1],fp_type[1],fn_type[1],f1_caa)
    print('total:',tp_total,fp_total,fn_total,f1_total)

with open("megatile_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(pilot_results)

# -----------

# crowd-analysis functions

# ---------
# create crowd list that matches format of players list so the same code can be used
# to compare the crowd answers to the expert answers
# (this is where we were losing the false negatives - used 233 as test case)
# (IMPORTANT: all megatiles used in pilot had at least one expert annotation, so if we use
# this code for situations where experts annotated null, then we'll need to make sure
# those null annotations are counted, and not ignored)
def build_crowd(crowd):
  new_tally = 0
  for megatile in crowd_answers:
    for cohort in megatile[1]:
      for label_type in cohort[1]:
        for ratings in label_type[1]:
          if ratings == []:
            crowd.append([cohort[0],label_type[0],megatile[0],-1,-1]) # include null ratings
            new_tally += 1
          for rating in ratings:
            crowd.append([cohort[0],label_type[0],megatile[0],rating[0],rating[1]])
            new_tally += 1
  result = new_tally


# HACKATHON - code your wisdom of crowds approach here -LHV TEST
def hack_crowd_answer(cohort_ratings,cohort,players,megatile,plaque_type,deltas):
      # generate list of all markers from cohort
      for player_rating in players:
        if player_rating[0] in cohort and player_rating[1] == plaque_type and player_rating[2] == megatile:
          # create record: x, y, player_id (or later, list of contributing ids), contributor count, combined
          cohort_ratings[0].append([player_rating[3],player_rating[4],[player_rating[0]],1,False])
      # HACKATHON - comment/uncomment this or copy/paste as needed to visualize a megatile
      if len(cohort_ratings) > 0:
        showmega(cohort_ratings[-1])
          
      # ************* HACKATHON - INSERT YOUR WISDOM OF CROWDS CODE HERE **************
      # only the final list item in cohort_ratings will be used by the build_crowd_answers
      # function to represent the crowd answer.

      return cohort_ratings

def build_crowd_answer(cohort_ratings,cohort,players,megatile,plaque_type,deltas):
      # generate collapse pass zero (no collapsing yet - just a list of all markers from cohort)
      for player_rating in players:
        if player_rating[0] in cohort and player_rating[1] == plaque_type and player_rating[2] == megatile:
          # create record: x, y, player_id (or later, list of contributing ids), contributor count, combined
          cohort_ratings[0].append([player_rating[3],player_rating[4],[player_rating[0]],1,False])

      pass_count = 0 # keeps track of how many collapse passes have been made
      collapse_tally = -1 # enable while loop to start
      while collapse_tally != 0:
        collapse_tally = 0 # reset collapse count
        label_pairs = [] # all possible combinations of label pairs from a cohort
        # iterate through all pairwise combinations of player labels from most recent collapse pass
        for indexed_rating1, indexed_rating2 in itertools.combinations(enumerate(cohort_ratings[pass_count]), 2):
          rating1_index = indexed_rating1[0]
          cohort_rating1 = indexed_rating1[1]
          rating2_index = indexed_rating2[0]
          cohort_rating2 = indexed_rating2[1]
          
          # if cohort_rating1[2] != cohort_rating2[2]: # different players
          if not(common_member(cohort_rating1[2],cohort_rating2[2])):
            x1 = cohort_rating1[0]
            y1 = cohort_rating1[1]
            c1 = [x1,y1]
            x2 = cohort_rating2[0]
            y2 = cohort_rating2[1]
            c2 = [x2,y2]
            pair_dist = math.dist(c1,c2)
            label_pairs.append([rating1_index,rating2_index,pair_dist])
        label_pairs.sort(key=lambda x: float(x[2])) # sort all marker pairs in cohort by their distance
        
        # collapse based on label_pairs
        # first, identify which pairs to collapse based on proximity
        # a given marker cannot be collapsed with more than another so flag by setting to true
        for label_pair in label_pairs:
          if label_pair[2] <= deltas[plaque_type]: # close enough to each other to collapse
            if cohort_ratings[pass_count][label_pair[0]][4] is False and \
               cohort_ratings[pass_count][label_pair[1]][4] is False: # neither yet been combined
              collapse_tally += 1 
              cohort_ratings[pass_count][label_pair[0]][4] = True # mark for combining
              cohort_ratings[pass_count][label_pair[1]][4] = True # mark for combining
              x1 = cohort_ratings[pass_count][label_pair[0]][0]
              y1 = cohort_ratings[pass_count][label_pair[0]][1]
              x2 = cohort_ratings[pass_count][label_pair[1]][0]
              y2 = cohort_ratings[pass_count][label_pair[1]][1]
              c1 = cohort_ratings[pass_count][label_pair[0]][3] # markers already combined
              c2 = cohort_ratings[pass_count][label_pair[1]][3] # markers already combined
              ct = c1 + c2
              w1 = c1 / ct # marker weight
              w2 = c2 / ct # marker weight
              xn = (x1 * w1) + (x2 * w2) # new (collapsed) marker x
              yn = (y1 * w1) + (y2 * w2) # new (collapsed) marker y
              p1 = cohort_ratings[pass_count][label_pair[0]][2] # marker contributors list
              p2 = cohort_ratings[pass_count][label_pair[1]][2] # marker contributors list
              pn = p1 + p2 # combined contributor list
              if len(cohort_ratings) < (pass_count + 2):
                cohort_ratings.append([])
              cohort_ratings[pass_count+1].append([xn,yn,pn,ct,False]) # add collapsed markers to next pass
        if collapse_tally > 0:
          pass_count += 1
          # carry forward any uncollapsed markers to next pass
          for cohort_rating in cohort_ratings[pass_count-1]:
            if cohort_rating[4] is False:
              cohort_ratings[pass_count].append(cohort_rating)
      return cohort_ratings

# determine final crowd answers by applying threshold to collapsed answers
def apply_thresholds(crowd_answers,response_thresholds):
  keep_tally = 0
  remove_tally = 0
  for megatile in crowd_answers:
    for cohort in megatile[1]:
      for label_type in cohort[1]:
        for ratings in label_type[1]:
          # print('assess cohort ratings')
          for rating in ratings:
            confidence = rating[3] / cohort_size
            if confidence > response_thresholds[label_type[0]]:
              keep_tally += 1
            else:
              remove_tally +=1

        for ratings_index in range(len(label_type[1])):
          label_type[1][ratings_index] = [x for x in label_type[1][ratings_index] 
            if (x[3] / cohort_size) > response_thresholds[label_type[0]]]
  return[keep_tally,remove_tally]

def build_ad_player_list(ad_player_list):
  for ad in artifact_datasets:
      ad_player_list.append([ad,[]])
      for i in crowd:
          if i[0] not in ad_player_list[-1][1] and i[2] == ad:
              ad_player_list[-1][1].append(i[0])

# crowd_answers is a massive list of nested lists:
# [megatiles-empty (50)
#   [megatile_id
#     [cohorts-empty (sample_size) 
#       [[list of players in cohort (cohort size)]
#         [plaque types - empty (2)
#           [plaque type
#             [[annotations - empty
#               [x, y, [list of annotators], number of annotations, False]]]]]]]]]
def build_crowd_answers(artifact_datasets,crowd_answers,samples,cohort_size,deltas):
  for megatile in artifact_datasets: # for each megatile
    crowd_answers.append([megatile,[]])
    track_cohorts=[] #keep track of previous selections to sample without replacement
    for i in range(0,samples):
      cohort = random.sample(player_list,cohort_size)
      while cohort in track_cohorts: # this is where it will get stuck if cohort_size=1
        cohort = random.sample(player_list,cohort_size)
      track_cohorts.append(cohort)
      crowd_answers[-1][1].append([cohort,[]])
      for plaque_type in range(0,2): # for each plaque type (0=Cored,1=CAA)
        crowd_answers[-1][1][-1][1].append([plaque_type,[]])
        cohort_ratings=[[]] # individual answers and all collapse passes for each cohort
        
        # HACKATHON - comment out the next line
        build_crowd_answer(cohort_ratings,cohort,players,megatile,plaque_type,deltas)
        
        # HACKATHON - uncomment this line
        # hack_crowd_answer(cohort_ratings,cohort,players,megatile,plaque_type,deltas)
        
        # only append final collapse pass (maximally collapsed set of annotations)
        # do not include all collapse passes in crowd_answers list
        crowd_answers[-1][1][-1][1][-1][1].append(cohort_ratings[-1])

# run a full set of simulations based on fixed cohort size and number of samples
# returns an overall average f-score combining both Cored and CAA plaques
def run_sim(deltas,response_thresholds,cohort_size,samples):
  global crowd_answers 
  crowd_answers = [] # accumulate all fully collapsed cohort answers for all megatiles
  build_crowd_answers(artifact_datasets,crowd_answers,samples,cohort_size,deltas)

  keep_tally = 0
  remove_tally = 0
  keep_tally,remove_tally = apply_thresholds(crowd_answers,response_thresholds)

  global crowd
  crowd = []
  new_tally = build_crowd(crowd)

  total_tally = keep_tally + remove_tally

  # compare to expert answers
  gtp = 0
  gfp = 0
  gfn = 0
  gf1 = 0

  gtp_type=[0,0]
  gfp_type=[0,0]
  gfn_type=[0,0]

  global ad_player_list
  ad_player_list=[]
  build_ad_player_list(ad_player_list)

  result_rows=0
  pilot_results=[]
  pilot_results.append(['player_id','expert_id','artifact_datatype_id','plaque_type','tp','fp','fn','f1'])
  for expert_index in range(0,1):
    for ad in artifact_datasets:
      tp_type=[0,0]
      fp_type=[0,0]
      fn_type=[0,0]
      tp_total = 0
      fp_total = 0
      fn_total = 0
      for label_type in range(0,2):
        for i in ad_player_list:
          expert=[]

          if i[0] == ad:
            for annotation in experts:
              if (annotation[2] == ad) and (annotation[1] == label_type):
                if (annotation[0] == expert_index):
                  expert.append([annotation[3],annotation[4]])
            for player_index in i[1]:
              player=[]
              for annotation in crowd:
                if (annotation[2] == ad) and (annotation[1] == label_type):
                  if (annotation[0] == player_index):
                    player.append([annotation[3],annotation[4]])

              tp=fn_tp(player,expert,plaque_types[label_type])
              fp=fn_fp(player,expert,plaque_types[label_type])
              fn=fn_fn(player,expert,plaque_types[label_type])
              f1 = fn_f1(player,expert,plaque_types[label_type])

              result_rows += 1
              pilot_results.append([result_rows,player_index,expert_index,ad,label_type,tp,fp,fn,f1])

              tp_total += tp
              fp_total += fp
              fn_total += fn

              tp_type[label_type] += tp
              fp_type[label_type] += fp
              fn_type[label_type] += fn

              gtp += tp
              gfp += fp
              gfn += fn
              
              gtp_type[label_type] += tp
              gfp_type[label_type] += fp
              gfn_type[label_type] += fn

      f1_cored = fn_fscore(tp_type[0],fp_type[0],fn_type[0])
      f1_caa = fn_fscore(tp_type[1],fp_type[1],fn_type[1])
      f1_total = fn_fscore(tp_total,fp_total,fn_total)
      f1_grand_total = fn_fscore(gtp,gfp,gfn)
      f1_grand_cored = fn_fscore(gtp_type[0],gfp_type[0],gfn_type[0])
      f1_grand_caa = fn_fscore(gtp_type[1],gfp_type[1],gfn_type[1])

    answers.append([deltas[0],deltas[1],response_thresholds[0],response_thresholds[1],cohort_size,samples,
      gtp,gfp,gfn,f1_grand_total,
      gtp_type[0],gfp_type[0],gfn_type[0],f1_grand_cored,
      gtp_type[1],gfp_type[1],gfn_type[1],f1_grand_caa])
    return f1_grand_total


# --------
# main crowd analysis routine
# note: if cohort_size = 1, then routine will freeze if sample more than 30 because sampling
#   without replacement
# deltas = distance thresholds for Cored, CAA
# response_threshold = responder percentage threshold above which to retain a crowd answer
# cohort_size = number of players per sampled cohort
# samples = number of cohorts to sample for each megatile
# params sent to run_sim = deltas,response_threshold,cohort_size,samples,n

answers=[]
maxval=0

print('cohort_size, f-score')
# iterate through different cohort sizes
for cohort_size in numpy.arange(2,31,1):
  # adjust thresholds dynamically based on cohort size and rounding aspects
  response_threshold_cored = round((cohort_size * 0.12)+0.02)/cohort_size
  response_threshold_caa = round((cohort_size * 0.04)+0.02)/cohort_size
  if (response_threshold_cored * cohort_size) < 1:
    response_threshold_cored = 1/cohort_size
  
  # return an average f-score for each cohort size in iteration above
  fscore = run_sim([2,2],[response_threshold_cored,response_threshold_caa],cohort_size,50)
  print (cohort_size, fscore, end="")

  # if new fscore exceeds previous max fscore, then add ***  
  if fscore > maxval:
    maxval = fscore
    print(' ***')
  else:
    print('')

print(answers)
with open("megatile_answers.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(answers)

with open("megatile_crowd_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(pilot_results)
