import numpy as np
from numpy import ma


class Ballot:
    def __init__(self, preferences: np.ndarray) -> None:
        self.preferences = preferences
        self.weight = 1
        
    def get_preference(self, eliminated_candidates: np.ndarray) -> np.ndarray:
        candidate_preferences = ma.array(self.preferences, mask=eliminated_candidates)
        if np.all(candidate_preferences == -1): # voter does not care
            return candidate_preferences / candidate_preferences.sum() * self.weight
        
        candidate_preferences = ma.masked_values(candidate_preferences, -1)
        vote = ma.masked_all(self.preferences.shape)
        vote[np.argmin(candidate_preferences)] = self.weight
        return vote
    
    def adjust_weight(self, factor: float) -> None:
        self.weight *= factor
        

def get_votes(ballots: np.ndarray, eliminated_candidates: np.ndarray) -> ma.MaskedArray:
    return ma.array([ballot.get_preference(eliminated_candidates) for ballot in ballots])


def reduce_voters(votes: ma.MaskedArray, eliminated_candidate: int, multiplier: float) -> np.ndarray:
    multipliers = np.ones(votes.shape[0])
    for i, vote in enumerate(votes):
        if vote[eliminated_candidate] == sum(vote):
            multipliers[i] = multiplier
    return multipliers


def find_winners(ballots: np.ndarray, num_winners: int) -> np.ndarray:
    eliminated_candidates = np.zeros(ballots.shape[1])
    ballots = np.array([Ballot(ballot) for ballot in ballots])
    quota = len(ballots) / (num_winners + 1)

    winners = []
    while len(winners) < num_winners:
        votes = get_votes(ballots, eliminated_candidates)
        vote_counts = np.sum(votes, axis=0)
        most_popular = np.argmax(vote_counts)
        most_votes = vote_counts[most_popular]
        if most_votes > quota:
            winners.append(most_popular)
            eliminated_candidates[most_popular] = 1
            multiplier = (most_votes - quota) / most_votes
            multipliers = reduce_voters(votes, most_popular, multiplier)
            for i in range(len(ballots)):
                ballots[i].adjust_weight(multipliers[i])
            continue
        
        least_popular = np.argmin(vote_counts)
        eliminated_candidates[least_popular] = 1
    
    return np.array(winners)


def main() -> None:
    ballots = np.array([
        [1, 2, -1, -1, -1, -1, -1],
        [1, 2, -1, -1, -1, -1, -1],
        [1, 2, -1, -1, -1, -1, -1],
        [1, 2, -1, -1, -1, -1, -1],
        
        [-1, 1, 2, 3, -1, -1, -1],
        [-1, 1, 2, 3, -1, -1, -1],
        [-1, 1, 2, 3, -1, -1, -1],
        [-1, 1, 2, 3, -1, -1, -1],
        [-1, 1, 2, 3, -1, -1, -1],
        [-1, 1, 2, 3, -1, -1, -1],
        [-1, 1, 2, 3, -1, -1, -1],
        
        [-1, 3, 1, 2, -1, -1, -1],
        
        [-1, -1, 3, 1, 2, -1, -1],
        [-1, -1, 3, 1, 2, -1, -1],
        [-1, -1, 3, 1, 2, -1, -1],
        
        [-1, -1, -1, 2, 1, 3, -1],
        
        [-1, -1, -1, -1, -1, 1, -1],
        [-1, -1, -1, -1, -1, 1, -1],
        [-1, -1, -1, -1, -1, 1, -1],
        [-1, -1, -1, -1, -1, 1, -1],
        
        [-1, -1, -1, -1, -1, 2, 1],
        [-1, -1, -1, -1, -1, 2, 1],
        [-1, -1, -1, -1, -1, 2, 1],
	])
    
    print(find_winners(ballots, 3))

if __name__ == '__main__':
    main()
