from gym.envs.registration import register

register(
    id='necTraffic-v11',
    entry_point='nec_traffic_v11.envs:NECTraffic',
)