# generate n policy
from collections import Counter
import random
import numpy as np


class Agent:
    def __init__(self):
        self.so_far_score = 0
        self.random_rate = 0.15
        self.vote_num = 3
        self.opp_move = ''
        self.bot_move = ''
        self.history_DNA = ''
        self.history_DNA_2 = ''
        # depend on history_DNA
        self.nuclease = {'01': 'a', '12': 'b', '20': 'c',
                         '10': 'd', '21': 'e', '02': 'f',
                         '00': 'g', '11': 'h', '22': 'i'}
        # depend on history_DNA_2
        self.wlt = {'01': 'b', '12': 'b', '20': 'b',
                    '10': 'c', '21': 'c', '02': 'c',
                    '00': 'a', '11': 'a', '22': 'a'}

        # vs opp
        self.stat_bot = Stat()
        self.stat_move = None
        self.opp_bot = OPP()
        self.opp_bot_move = None
        # vs self
        self.stat_bot_2 = Stat()
        self.stat_move_2 = None
        self.opp_bot_2 = OPP()
        self.opp_bot_move_2 = None

        # fellow up
        self.base_predictor_len = 6

        self.ensemble_predictors_len = 1
        self.number_of_predictors = (self.base_predictor_len * 4 + self.ensemble_predictors_len) * 3

        # init predictors
        self.predictors = [random.choice(['0', '1', '2']) for _ in range(self.number_of_predictors)]

        self.beaten = {'0': '2', '1': '0', '2': '1'}
        self.beat = {'0': '1', '1': '2', '2': '0'}

        self.predictor_score = list(np.random.rand(self.number_of_predictors))

        self.move = random.choice(['0', '1', '2'])

    def update_history(self, opp_action, step):
        opp_action = str(opp_action)
        for i in range(self.number_of_predictors):
            self.predictor_score[i] *= 0.9

            if opp_action == self.predictors[i]:
                # win, get prediction!
                self.so_far_score += 1
                self.predictor_score[i] += 1
            elif opp_action == self.beaten[self.predictors[i]]:
                # lose
                self.so_far_score -= 1
                self.predictor_score[i] -= 1
            else:
                # tie
                if abs(self.so_far_score) < 25:
                    tie_score = 0
                elif abs(self.so_far_score) < 50:
                    tie_score = self.so_far_score / 50
                else:
                    tie_score = self.so_far_score/abs(self.so_far_score)
                self.predictor_score[i] += tie_score

        # History matching
        self.opp_move += opp_action
        self.bot_move += self.move
        self.history_DNA += self.nuclease[opp_action + self.move]
        self.history_DNA_2 += self.wlt[opp_action + self.move]

        self.stat_move = str(self.stat_bot.statistical_prediction_agent(step, int(opp_action), int(self.move)))
        self.opp_bot_move = str(self.opp_bot.transition_agent(step, int(opp_action)))
        self.stat_move_2 = str(self.stat_bot_2.statistical_prediction_agent(step, int(opp_action),int(self.move)))
        self.opp_bot_move_2 = str(self.opp_bot_2.transition_agent(step, int(self.move)))

    def update_predictor(self, index, dna_len, step):
        if index == 0:
            target = self.history_DNA
        elif index == 1:
            target = self.history_DNA_2
        elif index == 2:
            target = self.bot_move
        else:
            target = self.opp_move

        # no same DNA in the length, next
        while dna_len >= 1 and not target[step - dna_len:step] in target[0:step - 1]:
            dna_len -= 1

        find_i = target.find(target[step - dna_len:step], 0, step - 1)
        self.predictors[self.base_predictor_len * index + 0] = self.opp_move[dna_len + find_i]
        self.predictors[self.base_predictor_len * index + 1] = self.beat[self.bot_move[dna_len + find_i]]
        rfind_i = target.rfind(target[step - dna_len:step], 0, step - 1)
        self.predictors[self.base_predictor_len * index + 2] = self.opp_move[dna_len + rfind_i]
        self.predictors[self.base_predictor_len * index + 3] = self.beat[self.bot_move[dna_len + rfind_i]]

    def all_update(self, step):
        limit = min([step, 10])
        for i in range(self.base_predictor_len-2):
            self.update_predictor(i, limit, step)

        self.predictors[self.base_predictor_len * 4 - 8] = self.stat_move
        self.predictors[self.base_predictor_len * 4 - 7] = self.beat[self.stat_move]
        self.predictors[self.base_predictor_len * 4 - 6] = self.stat_move_2
        self.predictors[self.base_predictor_len * 4 - 5] = self.beaten[self.stat_move_2]
        self.predictors[self.base_predictor_len * 4 - 4] = self.opp_bot_move
        self.predictors[self.base_predictor_len * 4 - 3] = self.beat[self.opp_bot_move]
        self.predictors[self.base_predictor_len * 4 - 2] = self.opp_bot_move_2
        self.predictors[self.base_predictor_len * 4 - 1] = self.beaten[self.opp_bot_move_2]

        # ensemble
        self.predictors[self.base_predictor_len * 4] = '0'

        # guess my action
        for i in range(self.base_predictor_len * 4 + self.ensemble_predictors_len, self.number_of_predictors):
            self.predictors[i] = self.beaten[self.predictors[i - self.base_predictor_len * 4]]

    def action(self, step):
        self.all_update(step)

        # select
        if self.random_rate < random.random():
            threshold = sorted(self.predictor_score)[-self.vote_num]
            vote = {'0': 0, '1': 0, '2': 0}
            for s, p in zip(self.predictor_score, self.predictors):
                if s >= threshold:
                    vote[p] += s

            self.move = self.beat[max(vote, key=vote.get)]
        else:
            self.move = random.choice(['0', '1', '2'])
        return self.move


# ensemble


class Stat:
    # Create a small amount of starting history
    history = {
        "guess": [0, 1, 2],
        "prediction": [0, 1, 2],
        "expected": [0, 1, 2],
        "action": [0, 1, 2],
        "opponent": [0, 1],
    }

    def statistical_prediction_agent(self, step, opp_action, my_real_action):
        actions = list(range(3))  # [0,1,2]
        last_action = self.history['action'][-1]
        opponent_action = opp_action if step > 0 else 2

        self.history['opponent'].append(opponent_action)

        move_frequency = Counter(self.history['opponent'])
        response_frequency = Counter(zip(self.history['action'], self.history['opponent']))
        move_weights = [move_frequency.get(n, 1) + response_frequency.get((last_action, n), 1) for n in
                        range(3)]
        guess = random.choices(population=actions, weights=move_weights, k=1)[0]

        # Compare our guess to how our opponent actually played
        guess_frequency = Counter(zip(self.history['guess'], self.history['opponent']))
        guess_weights = [guess_frequency.get((guess, n), 1) for n in range(3)]
        prediction = random.choices(population=actions, weights=guess_weights, k=1)[0]

        # Repeat, but based on how many times our prediction was correct
        prediction_frequency = Counter(zip(self.history['prediction'], self.history['opponent']))
        prediction_weights = [prediction_frequency.get((prediction, n), 1) for n in range(3)]
        expected = random.choices(population=actions, weights=prediction_weights, k=1)[0]

        # Play the +1 counter move
        action = (expected + 1) % 3

        # Persist state
        self.history['guess'].append(guess)
        self.history['prediction'].append(prediction)
        self.history['expected'].append(expected)
        self.history['action'].append(my_real_action)

        return action


class OPP:
    T = np.zeros((3, 3))
    P = np.zeros((3, 3))

    # a1 is the action of the opponent 1 step ago
    # a2 is the action of the opponent 2 steps ago
    a1, a2 = None, None

    def transition_agent(self, step, opp_action):
        if step > 1:
            self.a1 = opp_action
            self.T[self.a2, self.a1] += 1
            self.P = np.divide(self.T, np.maximum(1, self.T.sum(axis=1)).reshape(-1, 1))
            self.a2 = self.a1
            if np.sum(self.P[self.a1, :]) == 1:
                return (np.argmax(self.P[self.a1, :]) + 1) % 3
            else:
                return int(np.random.randint(3))
        else:
            if step == 1:
                self.a2 = opp_action
            return int(np.random.randint(3))


bot = Agent()


def work(observation, configuration):
    if observation.step == 0:
        move = bot.move
    else:
        bot.update_history(str(observation.lastOpponentAction), observation.step)
        move = bot.action(observation.step)

    return int(move)


# class Game:
#     def __init__(self):
#         self.step = 0
#         self.last_move = None
#
#     def compare(self, p1_action, p2_action):
#         self.step += 1
#         self.last_move = p2_action
#         if p1_action == p2_action:
#             return 0
#         elif (p1_action + 1) % 3 == p2_action:
#             return -1
#         else:
#             return 1
#
# def RandomBot():
#     return random.choice([0, 1])
#
# if True:
#     win_or_lose = [0, 0, 0]
#     env = Game()
#     bot = Agent()
#     for i in range(1000):
#         print(f'===round {i}===')
#         if i == 0:
#             p1_move = bot.move
#         else:
#             bot.update_history(str(env.last_move), env.step)
#             p1_move = bot.action(env.step)
#         p2_move = RandomBot()
#
#         result = env.compare(int(p1_move), p2_move)
#         win_or_lose[result+1] += 1
#     print(win_or_lose)