"""
Complex Opinion in the Political Arena
    - Measuring the interactions between campaign strategy, voting methodology, and
      election result happiness.

Nodes are considered to be people in a voting society with a d-length vector of opinions O.

Each singular opinion oᵢ∈ O is a value in [-1, 1] that represents how strongly
the individual feels about issue i.
    i close to -1 implies the individual strongly opposes the issue,
    i close to  1 implies the individual strongly supports the issue,
    and i close to 0 implies a neutral opinion.

Each oᵢ is sampled from a uniform distribution in [-1, 1].
"""
import pickle
import sys, os
import argparse
import numpy as np
import pandas as pd
from functools import reduce
from collections import Counter
from itertools import combinations

TOTAL_DISAGREE = -1
TOTAL_AGREE = 1
N_TRIALS = 100

def intgt0(x):
    """verify input is an integer > 0"""
    x = int(x)
    if x > 0:
        return x
    raise ValueError

def parse_args():
    """sets up the arguments that can be passed into the script"""
    parser = argparse.ArgumentParser(description="Quantifying voter happiness")
    parser.add_argument("-N","--population-size",
                        dest="pop_size",
                        type=intgt0,
                        help="Number of individuals who will cast a vote",
                        default=1000)

    parser.add_argument("-d","--num-opinions",
                        dest="vector_size",
                        type=intgt0,
                        help="Number of opinions per individual",
                        default=10)

    parser.add_argument("-k","--num-candidates",
                        dest="num_candidates",
                        type=intgt0,
                        help="the number of candidates per election",
                        default=10)

    parser.add_argument("-o","--output-directory",
                        dest="output",
                        help="place to find population files or create them",
                        default="ElectoralProcesses")
    return parser

class Voter:
    ID = 0

    def __init__(self, n_opinions=10, opinion_vector=None, voter_id=None):
        """
        Initialize a voting individual in the population.
        Creates an opinion vector and assigns an voter ID number
        args:
            :n_opinions (int) - size of the indiv.'s opinion vector
            :opinion_vector (array-like) - the vector of opinions (generated if None)
            :voter_id (int) - the id for this voter
        """
        if opinion_vector is None:
            #self._opinions = np.random.normal(0, 1, n_opinions)
            self._opinions = np.random.uniform(TOTAL_DISAGREE, TOTAL_AGREE, n_opinions)
        else:
            self._opinions = np.array(opinion_vector)

        if voter_id is None:
            self._id = Voter.ID
            Voter.ID += 1
        else:
            self._id = voter_id

    @property
    def num_opinions(self):
        return len(self.opinions)

    @classmethod
    def reset_ids(cls):
        cls.ID = 0

    @property
    def opinions(self):
        return np.array(self._opinions)

    @property
    def id(self):
        return self._id

    def to_list(self):
        return [self.id] + [op for op in self.opinions]

    def agreeability(self, candidate, t_lev):
        """
        return the amount of agreeability a candidate's views have with this voter

        agreeability values close to 0 imply that this voter's opinions align with the candidate's
        larger agreeability values imply that this voter's opinions and the opinions of the candidate
        deviate

        args:
            :candidate (Candidate)
            :transparency value, t
        returns:
            :the mean absolute error between the candidate's (exposed) opinion vector
             and the voter's own opinion
        """
        relevant_opinions = self.opinions[candidate.exposure(t_lev)]
        return sum(abs(relevant_opinions - candidate.opinions[candidate.exposure(t_lev)]))/t_lev

    def happiness(self, candidate):
        """
        return how happy a voter is with a candidate by computing the
        mean absolute error between their opinion vectors
        """
        return sum(abs(self.opinions - candidate.opinions))/len(self.opinions)

class Candidate(Voter):
    def __init__(self, voter, transparencies):
        """
        Initialize a Candidate, one such individual that has all the qualities
        of a Voter with the added attribute of a set of exposed opinions

        args:
            :transparencies (list) - the list of number exposed by all masks
        """
        super(Candidate, self).__init__(n_opinions=voter.num_opinions,
                                        opinion_vector=voter.opinions,
                                        voter_id=voter.id)
        # generate a mask of each size `transparency`
        visible = {}
        self._opinion_masks = {}
        for t in transparencies:
            if t > 1:
                visible[t] = visible[t-1] + sorted(np.random.choice(list([i for i in range(7) if i not in visible[t-1]]),size=1,replace=False))
            else:
                visible[t] = sorted(np.random.choice(list(range(7)),size=t,replace=False))
            self._opinion_masks[t] = [False]*7
            for d in visible[t]:
                self._opinion_masks[t][d] = True
        #print(self._opinion_masks)


    def exposure(self,t):
        return self._opinion_masks[t]


    def __str__(self):
        return f"Candidate(masks={self._opinion_masks}, n_ops={self.num_opinions})"

    def __repr__(self):
        return str(self)

def _get_voters(output_dir, n_voters, n_opinions, population_num):
    """
    generates and caches voting individuals with some number of opinions
    - the file will be cached to:
        output_dir/VotingPopulation__N_{n_voters}__D_{n_opinions}.csv
    - if the output file specified already exists, this file will be opened
      and its contents returned as a set of individuals

    args:
        :output_dir (str)  - directory path to the cache the populatin file
        :n_voters (int)    - number of voters to create
        :n_opinions (str)  - number of opinions per voter
        :population_num    - id of the population
    returns:
        :(list) - the list of voters, realized as Individual objects
    """
    cache_file = os.path.join(output_dir,"populations",f"VotingPopulation{population_num:02d}__N_{n_voters}__D_{n_opinions}.csv")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(cache_file):
        pop_df = pd.read_csv(cache_file)
        voter_rows = pop_df.values.tolist()
        voters = []
        for voter_data in voter_rows:
            v_id        = voter_data[0]
            opinion_vec = voter_data[1:]
            voters.append(Voter(voter_id=v_id, opinion_vector=opinion_vec))
        return voters
    else:
        pop_df = pd.DataFrame(columns=["id"] + [f"opinion_{i}" for i in range(n_opinions)])
        voters = []
        for i in range(n_voters):
            voter = Voter(n_opinions=n_opinions)
            voters.append(voter)
            pop_df.loc[i] = voter.to_list()

        pop_df.to_csv(cache_file, index=False)
        return voters


def choose_candidates(population, transparency_levels, n_candidates):
    """
    choose candidates from the voting pool
    args:
        :population (list of Voter) - the voting population
        :transparency_level (int)   - the number of exposed opinions
    returns:
        :a list of candidates with the given transparency lvl
    """
    nominated = np.random.choice(population, n_candidates)

    return [Candidate(nom, transparency_levels) for nom in nominated]

def population_stream(output_dir, n_voters, n_opinions, n_populations=100):
    """
    generate different populations of voters (size n_voters) with some number of opinions
    args:
        :output_dir (str) - path to read/write populations from/to
        :n_voters   (int) - number of voting individuals in the system
        :n_opinions (int) - number of opinions per indiv.
        :n_populations (int) - the number of populations to stream out
    yields:
        :a generator of these voter populations
    """
    for pop in range(n_populations):
        yield _get_voters(output_dir, n_voters, n_opinions, pop)

def calculate_happiness(population, elect):
    """
    calculate the happiness distribution across this population
    for this elected official
    args:
        :population (list of Voter)
        :elect (Candidate)
    """
    return np.array(list(map(lambda v:v.happiness(elect), population)))

#########################################################################
################## ranked choice voting section #########################
#########################################################################

def ranked_choice_voting(population, nominees, t):
    """
    Ranked choice voting scheme;

    Each voter in the population lists their ideal candidates from 1 to num_candidates

    After ranking has taken place, the ballots are tallied in the following way:

        1. measure the number of voters for each candidate that ran
        2. If a candidate's vote count ≥ 50%, that candidate wins and the algorithm is completed.
        3. Else, kick the bottom candidate out of the race and redistribute their
           votes to the remaining candidates. Go to 1.

    In this context, 'redistribute' means look at the rank sheets for each voter that
    has a booted candidate and allocate that voter's vote to the next candidate on the rank sheet.

    args:
        :population (list of Voter) - the voting population
        :nominees (list of Candidate) - the number of candidates running in the election
    returns:
        :the candidate that won
    """

    # casting votes
    all_ballots = []
    for voter in population:
        all_ballots.append(rank_sheet(voter, nominees, t))

    return ranked_choice_tally_votes(all_ballots, nominees)

def ranked_choice_tally_votes(ballot_box, nominees):
    """
    recursively searches for the candidate that is chosen by the ranked choice
    voting algorithm.

    args:
        :ballot_box (list of list of Candidate) - rankings for the population
    returns:
        :the candidate that won
    """
    votes = [ballot[0].id for ballot in ballot_box if ballot]
    id2candidate = {c.id:c for c in nominees}
    candidate2ballot = Counter(votes)
    candidate2prop = calculate_shares(candidate2ballot)
    max_prop = max(candidate2prop.values())
    if max_prop >= 0.51:
        max_candidate_id = list(filter(lambda c: candidate2prop[c] == max_prop, candidate2prop))[0]
        max_candidate = id2candidate[max_candidate_id]
        return max_candidate
    else:
        min_prop = min(candidate2prop.values())
        min_candidate_id = list(filter(lambda c: candidate2prop[c] == min_prop, candidate2prop))[0]
        for ballot in ballot_box:
            if ballot and ballot[0].id == min_candidate_id:
                ballot.pop(0)
        return ranked_choice_tally_votes(ballot_box, nominees)

def calculate_shares(candidate2votes):
    total_votes     = sum(candidate2votes.values())
    candidate2share = {c:candidate2votes[c]/total_votes for c in candidate2votes}

    return candidate2share

def rank_sheet(voter, candidate_roster, t):
    """
    return the ballot sheet for a voter in ranked choice method
    args:
        :voter (Voter)
        :candidate_roster (list of Candidates)
    """
    return sorted(candidate_roster,
                  key=lambda candidate:voter.agreeability(candidate, t))

###################################################################
################# approval voting section #########################
###################################################################
def approval_voting(pop, noms, t, mx_disagree=1):
    """
    implements approval based voting
    each Voter in the population votes for all of the candidates with
    a disagreeability value <= max_disagreeability

    args:
        :pop (list of Voter)
        :noms (list of Candidate)
        :mx_disagree (int)
    returns:
        :the winner of the election
    """
    # :)
    ballot_box = []
    for voter in pop:
        ballot_box.extend([c.id for c in approved(voter, noms, t)])
    id2candidate = {c.id:c for c in noms}
    return id2candidate[Counter(ballot_box).most_common(1)[0][0]]



def approved(voter, nominees, t, max_disag=1):
    """
    return the approved candidates of the voter

    args:
        :voter (Voter)
        :nominess (list of Candidate)
        :max_disagg (int)
    returns:
        :list of Candidate
    """
    return [candidate for candidate in nominees if voter.agreeability(candidate, t) <= max_disag]


###################################################################
################## general voting section #########################
###################################################################

def top_contender(voter, candidates, t):
    return rank_sheet(voter, candidates, t)[0]

def general_election(population, nominees, t):
    """
    perform United States style voting by selecting the argmax of opinion

    args:
        :population (list of Voter)
    returns:
        :the winner
    """
    id2candidate = {c.id:c for c in nominees}
    votes = Counter([top_contender(voter, nominees, t).id for voter in population])
    winner_id = max(votes.items(), key=lambda tup:tup[1])[0]
    winner = id2candidate[winner_id]
    return winner


if __name__ == "__main__":
    args = parse_args().parse_args()

    os.makedirs(os.path.join(args.output, "populations"), exist_ok=True)
    for etype in ["general","ranked","approval"]:
        os.makedirs(os.path.join(args.output, etype), exist_ok=True)

    # if the voting scheme to test is ranked, then try election ranked voting for
    # 100 separate trials, for varying transparency levels, for varying #'s of ranks

    transparency_lvls = list(range(1, args.vector_size + 1))
    # save count of times that different voting schemes arrived at the same winner
    comb2ct = {t:{comb:0 for comb in combinations(["general","ranked","approval"],2)} for t in transparency_lvls}

    for i, population in enumerate(population_stream(args.output, args.pop_size, args.vector_size)):
        nominees = choose_candidates(population, transparency_lvls, args.num_candidates)
        for t in transparency_lvls:
            scheme2winner = {}
            for election_process, voting_scheme in zip((general_election, ranked_choice_voting, approval_voting),
                                                   ("general","ranked","approval")):

                format_ = f"Population{i:03d}__V_{voting_scheme}__T{t:02d}__K{args.num_candidates:03d}.bin"
                output_dir = os.path.join(args.output, voting_scheme)
                filename = os.path.join(output_dir, format_)
                #if not os.path.exists(filename):
                elected  = election_process(population, nominees, t)
                happiness_ratings = calculate_happiness(population, elected)
                scheme2winner[voting_scheme] = elected.id
                with open(filename, 'wb') as pk:
                    pickle.dump(happiness_ratings, pk)
            print('\r done with elections for p {} and t {}, outcomes: {}'.format(i,t,scheme2winner),end = '',flush=True)
            # compute the number of times each scheme arrived at the same conclusion
            for scheme_comb in comb2ct[t]:
                if scheme2winner[scheme_comb[0]] == scheme2winner[scheme_comb[1]]:
                    comb2ct[t][scheme_comb] += 1

        Voter.reset_ids()
        #print('\r done with pop ',i, end = '',flush=True)

    # save number of times each scheme / transparency lvl arrived at same winner
    match_df = pd.DataFrame(columns=list(combinations(["general","ranked","approval"], 2)),index = transparency_lvls)
    for t in transparency_lvls:
        match_df.loc[t] = [comb2ct[t][comb] for comb in match_df.columns]

    match_df.to_csv(os.path.join(args.output, "same_winners.csv"),index=False)
