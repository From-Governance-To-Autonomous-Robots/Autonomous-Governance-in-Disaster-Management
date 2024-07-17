from gym.envs.registration import register
from .taskEnvTrain import TaskSequenceTrainEnv
from .taskEnvVal import TaskSequenceValEnv
from .oracleEnvTrain import OracleSequenceTrainEnv
from .oracleEnvVal import OracleSequenceValEnv

register(
    id='TaskSequenceTrainEnv-v0',
    entry_point='envs.taskEnvTrain:TaskSequenceTrainEnv',
)

register(
    id='TaskSequenceValEnv-v0',
    entry_point='envs.taskEnvVal:TaskSequenceValEnv',
)

register(
    id='OracleSequenceValEnv-v0',
    entry_point='envs.oracleEnvVal:OracleSequenceValEnv',
)

register(
    id='OracleSequenceTrainEnv-v0',
    entry_point='envs.oracleEnvTrain:OracleSequenceTrainEnv',
)