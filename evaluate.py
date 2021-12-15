import random
import numpy as np
from flatland.envs.rail_env_utils import load_flatland_environment_from_file
from flatland.utils.rendertools import RenderTool
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.malfunction_generators import ParamMalfunctionGen, MalfunctionParameters
from flatland.envs.rail_env import RailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.observations import TreeObsForRailEnv
from observation_utils import normalize_observation

env_params = {
    "small": {
        "n_agents": 5,
        "x_dim": 30,
        "y_dim": 30,
        "n_cities": 2,
        "max_rails_between_cities": 2,
        "max_rail_pairs_in_city": 1,
        "malfunction_rate": 0.0,
        "seed": 0
    },
    "medium": {
        "n_agents": 10,
        "x_dim": 30,
        "y_dim": 30,
        "n_cities": 2,
        "max_rails_between_cities": 2,
        "max_rail_pairs_in_city": 2,
        "malfunction_rate": 0.0,
        "seed": 0
    },
    "large":
    {
        # Test_2
        "n_agents": 20,
        "x_dim": 30,
        "y_dim": 30,
        "n_cities": 3,
        "max_rails_between_cities": 2,
        "max_rail_pairs_in_city": 2,
        "malfunction_rate": 0.0,
        "seed": 0
    }
}


def get_env(name):
    env = load_flatland_environment_from_file(name+".pkl")
    params = env_params[name]
    env._seed(params["seed"])
    return env, params


def get_state_action_size(env):
    n_features_per_node = env.obs_builder.observation_dim
    n_nodes = sum([np.power(4, i) for i in range(3)])
    state_size = n_features_per_node * n_nodes
    return state_size, 5


def create_rail_env(env_params, observation_tree_depth=2,
                    observation_max_path_depth=30):
    predictor = ShortestPathPredictorForRailEnv(observation_max_path_depth)
    tree_observation = TreeObsForRailEnv(max_depth=observation_tree_depth,
                                         predictor=predictor)

    n_agents = env_params.n_agents
    x_dim = env_params.x_dim
    y_dim = env_params.y_dim
    n_cities = env_params.n_cities
    max_rails_between_cities = env_params.max_rails_between_cities
    max_rail_pairs_in_city = env_params.max_rail_pairs_in_city
    seed = env_params.seed

    # Break agents from time to time
    malfunction_parameters = MalfunctionParameters(
        malfunction_rate=env_params.malfunction_rate,
        min_duration=20,
        max_duration=50
    )

    return RailEnv(
        width=x_dim, height=y_dim,
        rail_generator=sparse_rail_generator(
            max_num_cities=n_cities,
            grid_mode=False,
            max_rails_between_cities=max_rails_between_cities,
            max_rail_pairs_in_city=max_rail_pairs_in_city
        ),
        line_generator=sparse_line_generator(),
        number_of_agents=n_agents,
        malfunction_generator=ParamMalfunctionGen(malfunction_parameters),
        obs_builder_object=tree_observation,
        random_seed=seed
    )


def evaluate(env, params, policy, render=False, max_steps=int(1e8)):
    n_agents = params["n_agents"]
    seed = params["seed"]
    env._seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    score = 0

    obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True, random_seed=seed)

    agent_obs = [None] * n_agents
    action_dict = dict()

    if render:
        env_renderer = RenderTool(env, gl="PGL")

    # Build initial agent-specific observations
    for agent in env.get_agent_handles():
        if obs[agent]:
            agent_obs[agent] = normalize_observation(obs[agent], 2, observation_radius=10)

    # Run episode
    for step in range(max_steps):
        for agent in env.get_agent_handles():
            if info['action_required'][agent]:
                action = policy.act(agent_obs[agent])
            else:
                action = 0
            action_dict.update({agent: action})

        # Environment step
        next_obs, all_rewards, done, info = env.step(action_dict)

        # Process next observations
        for agent in env.get_agent_handles():
            if next_obs[agent]:
                agent_obs[agent] = normalize_observation(next_obs[agent], 2, observation_radius=10)

            score += all_rewards[agent]

        if render:
            env_renderer.render_env(
                    show=True,
                    frames=False,
                    show_observations=False,
                    show_predictions=False
                )

        if done['__all__']:
            break

    return score
