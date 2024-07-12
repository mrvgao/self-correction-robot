"""
Train a robot task with self correction from value function.

v = V(s_t-1) + R(s_t);

R(s_t) = 1 if P(s_t) == 1 else -1;

loss = v - V(s_t)

\detla s_t = \partial(loss) / \partial(s_t)

s_t' = s_t' + \alpha \delta s_t

\pi(s_t') = a';

loss2 = a' - \pi(s_t)

\theta' += \gamma * \partial(loss2) / \partial(\theta; \pi)

# reset the to V(s_t-1)

new_action = \pi(\theta')

obs.step(new_action)
"""