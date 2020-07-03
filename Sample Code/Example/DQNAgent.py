import math
import os
import os.path
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PER import Memory

##############################################################################

gamma = 0.99
learning_rate = 0.0002

memory_size = 2 ** 20
update_target_freq = 10000

num_of_inputs  = 41
hidden_1       = 512
hidden_2       = 512
hidden_3       = 64
num_of_outputs = 4
batch_size     = 32

num_of_atoms  =  41
Vmin          = -4
Vmax          =  4

save_model_frequency = 1000

##############################################################################

use_cuda = torch.cuda.is_available()
# torch.cuda.set_device(0)
# device = torch.device("cuda:0" if use_cuda else "cpu")
device = "cpu"
Tensor = torch.Tensor
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor

##############################################################################

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(path):
    return torch.load(path)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def projection_distribution(next_state, reward, done, target_model):

    delta_z = float(Vmax - Vmin) / (num_of_atoms - 1)
    support = torch.linspace(Vmin, Vmax, num_of_atoms).to(device)

    next_dist = target_model(next_state).data * support
    next_action = next_dist.sum(2).max(1)[1]
    next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(2))
    next_dist = next_dist.gather(1, next_action).squeeze(1)

    reward = reward.unsqueeze(1).expand_as(next_dist)
    done = done.unsqueeze(1).expand_as(next_dist)
    support = support.unsqueeze(0).expand_as(next_dist)

    Tz = reward + (1 - done) * gamma * support
    Tz = Tz.clamp(min=Vmin, max=Vmax)
    b = (Tz - Vmin) / delta_z
    l = b.floor().long()
    u = b.ceil().long()

    offset = torch.linspace(0, (batch_size - 1) * num_of_atoms, batch_size).long() \
        .unsqueeze(1).expand(batch_size, num_of_atoms).to(device)

    proj_dist = torch.zeros(next_dist.size()).to(device)
    proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
    proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

    return proj_dist

##############################################################################

class ExperienceReplay(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, experience):
        if self.position >= len(self.memory):
            self.memory.append(experience)
        else:
            self.memory[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return zip(*random.sample(self.memory, batch_size))

    def __len__(self):
        return len(self.memory)

##############################################################################

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()

        self.in_features  = in_features
        self.out_features = out_features
        self.std_init     = std_init

        self.weight_mu    = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu    = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        weight = self.weight_mu
        bias   = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        epsilon_in  = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x

##############################################################################

class FinalDQN(nn.Module):

    def __init__(self, options):
        super(FinalDQN, self).__init__()

        self.double_dqn, self.dueling_dqn, self.per, self.noisy_dqn, self.dist_dqn = options

        self.NUM_ATOM_KEY = 1 + (num_of_atoms - 1) * self.dist_dqn

        if self.dueling_dqn:
            self.linear1 = nn.Linear(num_of_inputs, hidden_1)
            self.linear2 = nn.Linear(hidden_1, hidden_2)

            if self.noisy_dqn:
                self.noisy_advantage1 = NoisyLinear(hidden_2, hidden_3)
                self.noisy_advantage2 = NoisyLinear(hidden_3, self.NUM_ATOM_KEY * num_of_outputs)

                self.noisy_value1 = NoisyLinear(hidden_2, hidden_3)
                self.noisy_value2 = NoisyLinear(hidden_3, self.NUM_ATOM_KEY )
            else:
                self.advantage1 = nn.Linear(hidden_2, hidden_3)
                self.advantage2 = nn.Linear(hidden_3, ( self.NUM_ATOM_KEY * num_of_outputs ))

                self.value1 = nn.Linear(hidden_2, hidden_3)
                self.value2 = nn.Linear(hidden_3, self.NUM_ATOM_KEY)
        else:
            if self.noisy_dqn:
                self.linear = nn.Linear(num_of_inputs, hidden_1)
                self.noisy1 = NoisyLinear(hidden_1, hidden_2)
                self.noisy2 = NoisyLinear(hidden_2, self.NUM_ATOM_KEY * num_of_outputs)
            else:
                self.linear = nn.Linear(num_of_inputs, hidden_1)
                self.linear1 = nn.Linear(hidden_1, hidden_2)
                self.linear2 = nn.Linear(hidden_2, self.NUM_ATOM_KEY * num_of_outputs)

        self.activation = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)

        if self.dueling_dqn:
            output1 = self.linear1(x)
            output1 = self.activation(output1)
            output2 = self.linear2(output1)
            output2 = self.activation(output2)

            if self.noisy_dqn:
                output_advantage = self.noisy_advantage1(output2)
                output_advantage = self.activation(output_advantage)
                output_advantage = self.noisy_advantage2(output_advantage)

                output_value = self.noisy_value1(output2)
                output_value = self.activation(output_value)
                output_value = self.noisy_value2(output_value)
            else:
                output_advantage = self.advantage1(output2)
                output_advantage = self.activation(output_advantage)
                output_advantage = self.advantage2(output_advantage)

                output_value = self.value1(output2)
                output_value = self.activation(output_value)
                output_value = self.value2(output_value)

            if self.dist_dqn:
                output_value     = output_value.view(batch_size, 1, self.NUM_ATOM_KEY)
                output_advantage = output_advantage.view(batch_size, num_of_outputs, self.NUM_ATOM_KEY)

            output_final = output_value + output_advantage - output_advantage.mean()
        else:
            if self.noisy_dqn:
                output1 = self.linear(x)
                output1 = self.activation(output1)

                output2 = self.noisy1(output1)
                output2 = self.activation(output2)

                output_final = self.noisy2(output2)
            else:
                output1 = self.linear(x)
                output1 = self.activation(output1)

                output2 = self.linear1(output1)
                output2 = self.activation(output2)

                output_final = self.linear2(output2)

        if self.dist_dqn:
            output_final = F.softmax(output_final.view(-1, self.NUM_ATOM_KEY)).view(-1, num_of_outputs, self.NUM_ATOM_KEY)

        return output_final

    def reset_noise(self):
        if self.dueling_dqn:
            if self.noisy_dqn:
                self.noisy_advantage1.reset_noise()
                self.noisy_advantage2.reset_noise()

                self.noisy_value1.reset_noise()
                self.noisy_value2.reset_noise()
            else:
                pass
        else:
            if self.noisy_dqn:
                self.noisy1.reset_noise()
                self.noisy2.reset_noise()
            else:
                pass

##############################################################################

class DQN(object):

    def __init__(self, options, resume_previous_train):
        self.model  = FinalDQN(options).to(device)
        self.target = FinalDQN(options).to(device)

        self.double_dqn, self.dueling_dqn, self.per, self.noisy_dqn, self.dist_dqn = options
        options_list = ["double", "dueling", "per", "noisy", "dist"]

        if self.per:
            self.memory = Memory(memory_size)
        else:
            self.memory = ExperienceReplay(memory_size)

        self.loss_func = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=learning_rate)
        self.num_updates = 0

        if all(options):
            self.PATH = "Plots/RL/all/network_all.pth"
        elif not any(options):
            self.PATH = "Plots/RL/vanilla_dqn/network_vanilla_dqn.pth"
        else:
            zero_idx = list(options).index(0)
            self.PATH = "Plots/RL/no_" + options_list[zero_idx] + "_dqn/network_no_" + options_list[zero_idx] + "_dqn.pth"

        if resume_previous_train and os.path.exists(self.PATH):
            print("Loading previously saved model ... ")
            self.model.load_state_dict(load_model(self.PATH))

    ################################## Select Action ##################################

    def act(self, state, epsilon):

        if self.noisy_dqn and self.dist_dqn:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            dist = self.model.forward(state).data.cpu()
            dist = dist * torch.linspace(Vmin, Vmax, num_of_atoms)
            action = dist.sum(2).max(1)[1].numpy()[0]

        elif self.noisy_dqn:
            state = FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.model.forward(state)
            action = q_values.max(1)[1].data[0]

        elif self.dist_dqn:
            random_for_egreedy = torch.rand(1)[0]
            if random_for_egreedy > epsilon:
                with torch.no_grad():
                    state = torch.FloatTensor(state).unsqueeze(0).to(device)
                    dist = self.model.forward(state).data.cpu()
                    dist = dist * torch.linspace(Vmin, Vmax, num_of_atoms)
                    action = dist.sum(2).max(1)[1].numpy()[0]
            else:
                action = random.randint(0, 3)

        else:
            random_for_egreedy = torch.rand(1)[0]
            if random_for_egreedy > epsilon:
                with torch.no_grad():
                    state = Tensor(state).to(device)
                    action_from_nn = self.model(state)
                    action = torch.max(action_from_nn, 0)[1]
                    action = action.item()
            else:
                action = random.randint(0, 3)

        return action

    ################################# Optimize Network #################################

    def optimize(self, experience):

        ################################# Add/Take Experience #################################

        if self.per:
            self.memory.add(experience)

            if self.memory.tree.n_entries < batch_size:
                return

            batch, idxs, is_weights = self.memory.sample(batch_size)

            state = Tensor([info[0] for info in batch]).to(device)
            action = LongTensor([info[1] for info in batch]).to(device)
            new_state = Tensor([info[2] for info in batch]).to(device)
            reward = Tensor([info[3] for info in batch]).to(device)
            done = Tensor([info[4] for info in batch]).to(device)

        else:
            self.memory.push(experience)

            if len(self.memory) < batch_size:
                return

            state, action, new_state, reward, done = self.memory.sample(batch_size)

            state = Tensor(state).to(device)
            new_state = Tensor(new_state).to(device)
            reward = Tensor(reward).to(device)
            action = LongTensor(action).to(device)
            done = Tensor(done).to(device)

        ################################# Find Loss #################################

        if self.dist_dqn:
            proj_dist = projection_distribution(new_state, reward, done, self.target)
            if self.double_dqn:
                dist = self.model(state)
            else:
                dist = self.target(state)
            action = action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, num_of_atoms)
            dist = dist.gather(1, action).squeeze(1)
            dist.data.clamp_(0.01, 0.99)
            loss = -(proj_dist * dist.log()).sum(1)
            if self.per:
                weights = torch.FloatTensor(is_weights)
                loss = loss * weights
            loss = loss.mean()
        else:
            if self.double_dqn:
                new_state_indices = self.model(new_state).detach()
                max_new_state_indices = torch.max(new_state_indices, 1)[1]

                new_state_values = self.target(new_state).detach()
                max_new_state_values = new_state_values.gather(1, max_new_state_indices.unsqueeze(1)).squeeze(1)
            else:
                new_state_values = self.target(new_state).detach()
                max_new_state_values = torch.max(new_state_values, 1)[0]
            target_value = reward + (1 - done) * gamma * max_new_state_values
            predicted_value = self.model(state).gather(1, action.unsqueeze(1)).squeeze(1)
            if self.per:
                abs_error = torch.abs(target_value - predicted_value).detach()
                self.memory.update(idxs, abs_error.numpy())
                loss = (torch.FloatTensor(is_weights) * self.loss_func(predicted_value, target_value)).mean()
            else:
                loss = self.loss_func(predicted_value, target_value).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.num_updates % update_target_freq == 0:
            self.target.load_state_dict(self.model.state_dict())

        if self.num_updates % save_model_frequency == 0:
            save_model(self.model, self.PATH)

        if self.noisy_dqn:
            self.model.reset_noise()
            self.target.reset_noise()

        self.num_updates += 1

##############################################################################