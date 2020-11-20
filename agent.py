# generate n policy

import random
import numpy as np

class Agent:
    def __init__(self):
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
        # fellow up
        self.predictor_len = 4
        self.number_of_predictors = self.predictor_len * 4 * 3

        # init predictors
        self.predictors = [random.choice(['0', '1', '2']) for _ in range(self.number_of_predictors)]

        self.beaten = {'0': '2', '1': '0', '2': '1'}
        self.beat = {'0': '1', '1': '2', '2': '0'}

        self.predictor_score = list(np.random.rand(self.number_of_predictors))

        self.move = random.choice(['0', '1', '2'])

    def update_history(self, opp_action):
        opp_action = str(opp_action)
        for i in range(self.number_of_predictors):
            self.predictor_score[i] *= 0.9

            if opp_action == self.predictors[i]:
                # win, get prediction!
                self.predictor_score[i] += 1
            elif opp_action == self.beaten[self.predictors[i]]:
                # lose
                self.predictor_score[i] -= 1
            else:
                # tie
                self.predictor_score[i] -= 0.5

        # History matching
        self.opp_move += opp_action
        self.bot_move += self.move
        self.history_DNA += self.nuclease[opp_action + self.move]
        self.history_DNA_2 += self.wlt[opp_action + self.move]

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
        self.predictors[self.predictor_len*index+0] = self.opp_move[dna_len + find_i]
        self.predictors[self.predictor_len*index+1] = self.beat[self.bot_move[dna_len + find_i]]
        rfind_i = target.rfind(target[step - dna_len:step], 0, step - 1)
        self.predictors[self.predictor_len*index+2] = self.opp_move[dna_len + rfind_i]
        self.predictors[self.predictor_len*index+3] = self.beat[self.bot_move[dna_len + rfind_i]]

    def all_update(self, step):
        limit = min([step, 15])
        for i in range(self.predictor_len):
            self.update_predictor(i, limit, step)
        # other
        for i in range(self.predictor_len*4, self.number_of_predictors):
            self.predictors[i] = self.beaten[self.predictors[i - self.predictor_len*4]]

    def action(self, step):
        self.all_update(step)

        # select
        threshold = sorted(self.predictor_score)[-5]
        vote = {'0': 0, '1': 0, '2': 0}
        for s, p in zip(self.predictor_score, self.predictors):
            if s >= threshold:
                vote[p] += s
        # print(vote)
        # print(max(vote, key=vote.get))
        # self.move = self.beat[self.predictors[self.predictor_score.index(max(self.predictor_score))]]
        self.move = self.beat[max(vote, key=vote.get)]
        return self.move


bot = Agent()


def work(observation, configuration):
    if observation.step == 0:
        move = bot.move
    else:
        bot.update_history(str(observation.lastOpponentAction))
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
#             bot.update_history(str(env.last_move))
#             p1_move = bot.action(env.step)
#         p2_move = RandomBot()
#
#         result = env.compare(int(p1_move), p2_move)
#         win_or_lose[result+1] += 1
#     print(win_or_lose)