from torch import argmax, log, exp, clamp, min, nn
from torch.distributions import Categorical
from torch.optim import Adam

class Agent:
    """
    Class to invluce the networks that the agent consists off.
    And function for using them.
    """
    def __init__(self, actor, critic, train_parameters, manager=None):
        self.actor = actor
        self.critic = critic
        self.manager = manager
        self.train_parameters = train_parameters

        # Add optimizers
        self.actor_optim = Adam(self.actor.parameters(), lr=self.train_parameters.actor_lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.train_parameters.critic_lr)
        if manager is not None:
            self.manager_optim = Adam(self.manager.parameters(), lr=self.train_parameters.manager_lr)

    def train(self, batch_obs, batch_acts, batch_log_probs, batch_rtgs):
        """
        Trains the agents networks
        """
        # Critics evalutations
        value_estimate, _ = self.evaluate(batch_obs, batch_acts)

        advantage = _compute_advantage(batch_rtgs, value_estimate)

        for _ in range(self.train_parameters.n_updates_per_iteration):
            # Here we update the networks a few times with the current rollout
            value_estimate, curr_log_probs = self.evaluate(batch_obs, batch_acts)

            actor_loss = _compute_actor_loss(curr_log_probs, batch_log_probs,\
                                                advantage, self.train_parameters.clip)

            self.actor_optim.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optim.step()

            critic_loss = nn.MSELoss()(value_estimate, batch_rtgs)

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

    def get_action(self, obs, greedy=False):
        """
        Getting an action from the actor given an observation.
        Either using the stochastic policy or the greedy one.
        """
        probs = self.actor(obs)

        if greedy:
            # Select action with highest probability
            action = argmax(probs, dim=1)
        else:
            # Sample action
            distr = Categorical(probs)
            action = distr.sample()

        log_prob = log(probs[0, action])

        return action, log_prob.detach()

    def evaluate(self, batch_obs, batch_acts):
        """
        Evaluates a batch of observations.
        Returns the critics vlaue estimate,
        and the log probs for the actions with the current actor.
        """
        value_estimate = self.critic(batch_obs).squeeze()

        all_probs = self.actor(batch_obs)
        log_probs = log(all_probs[range(len(batch_acts)), batch_acts])

        return value_estimate, log_probs


def _compute_actor_loss(curr_log_probs, batch_log_probs, advantage, clip):
    # Since, exp(log(a) - log(b)) = (a / b), we can perform this computation
    ratios = exp(curr_log_probs - batch_log_probs)
    surr1 = ratios * advantage
    surr2 = clamp(ratios, 1 - clip, 1 + clip) * advantage

    actor_loss = (-min(surr1, surr2)).mean()

    return actor_loss

def _compute_advantage(batch_rtgs, value_estimate):
    # Compute advantage
    advantage = batch_rtgs - value_estimate.detach()

    # Normalize advantage to make the learning more stable
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)
    # add small number to avoid zero division

    return advantage
