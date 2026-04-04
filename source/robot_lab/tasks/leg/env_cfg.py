### --- IMPORTS --- ###
import math
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as UniformNoise
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise
from .leg import LEG_CFG
from .. import mdp

### --- SCENE DEFINITION --- ###
@configclass
class LegSceneCfg(InteractiveSceneCfg):

    # Robot.
    robot: ArticulationCfg = LEG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Lights.
    light = AssetBaseCfg(
        prim_path="/World/SkyLight",
        spawn=sim_utils.DomeLightCfg(
            color=(0.9, 0.9, 0.9),
            intensity=750.0,
        ),
    )

### --- MDP ACTIONS --- ###
@configclass
class ActionsCfg:
    joint_effort = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["Hip", "Thigh_01", "Knee"],
        scale=0.25,
    )

### --- MDP OBSERVATIONS --- ###
@configclass
class ObservationsCfg:

    # Define the observation terms available to the agent.
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos                   = ObsTerm(func=mdp.joint_pos_rel, history_length=10, noise=GaussianNoise(mean=0.0, std=0.01))
        joint_vel                   = ObsTerm(func=mdp.joint_vel, history_length=10, noise=GaussianNoise(mean=0.0, std=0.01))

        phase_signal = ObsTerm(func=mdp.phase_sin_cos, params={"T": 5.0})

        # Post initialization.
        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        joint_pos                   = ObsTerm(func=mdp.joint_pos_rel, history_length=10)
        joint_vel                   = ObsTerm(func=mdp.joint_vel, history_length=10)
        
        phase_signal = ObsTerm(func=mdp.phase_sin_cos, params={"T": 5.0})
        target_foot_pos = ObsTerm(
            func=mdp.target_foot_pos_local,
            params={
                "stride_x": 0.0, 
                "stride_y": 0.10, 
                "clearance_z": 0.06, 
                "cycle_period": 5.0, 
                "stance_ratio": 0.5, 
                "foot_centre_pos": (-0.0025, 0.112, 0.643)
            } # Was (-0.0025, 0.112, 0.743)
        )
        actual_foot_pos = ObsTerm(
            func=mdp.actual_foot_pos_local,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["Calf"]), "foot_offset": (0.0, 0.0, -0.25)}
        )

        # Post initialization.
        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

### --- MDP EVENTS --- ###
@configclass
class EventCfg:
    # ON STARTUP
    randomize_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["Hip_01","Thigh","RS03","Calf"]),
            "mass_distribution_params": (0.8,1.2),
            "operation": "scale",
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )
    radomize_rigid_body_mass_inertia = EventTerm(
        func=mdp.randomize_rigid_body_mass_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_inertia_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    randomize_coms = EventTerm(
        func=mdp.randomize_rigid_body_coms,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "com_distribution_params": ((-0.025, 0.025), (-0.025, 0.025), (-0.025, 0.025)),
            "operation": "add",
            "distribution": "uniform",
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["Hip", "Thigh_01", "Knee"]),
            "stiffness_distribution_params": (50.0, 75.0),
            "damping_distribution_params": (3.3, 5.0),
            "operation": "abs",
            "distribution": "uniform",
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )

    # ON RESET:
    
    # Hip Limits: -90.0 to 30.0 degrees.
    # Thigh_01 Limits: 0.0 to 105.0 degrees.
    # Knee Limits: 0.0 to 90.0 degrees.
    reset_hip_joint = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["Hip"]),
            "position_range": (-1.57, 0.52),
            "velocity_range": (-0.5, 0.5),
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )
    reset_thigh_joint = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["Thigh_01"]),
            "position_range": (0.0, 1.83),
            "velocity_range": (-0.5, 0.5),
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )
    reset_knee_joint = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["Thigh_01"]),
            "position_range": (0.0, 1.57),
            "velocity_range": (-0.5, 0.5),
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )

    # ON INTERVAL:
    push_robot = EventTerm(
        func=mdp.apply_external_force_torque_stochastic,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["Thigh","RS03","Calf"]),
            "force_range": {
                "x": (-200.0, 200.0),
                "y": (-200.0, 200.0),
                "z": (-200.0, 200.0),
            },
            "torque_range": {"x": (-15.0, 15.0), "y": (-15.0, 15.0), "z": (-15.0, 15.0)},
            "probability": 0.002,  # Expect step = 1 / probability
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )
    

### --- MDP REWARDS --- ###
@configclass
class RewardsCfg:
    foot_tracking = RewTerm(
        func=mdp.track_step_trajectory,
        weight=-4.0, # Multiplies the error to penalize straying from target
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["Calf"]),
            "stride_x": 0.0, 
            "stride_y": 0.10, 
            "clearance_z": 0.06, 
            "cycle_period": 5.0, 
            "stance_ratio": 0.5, 
            "foot_offset": (0.0, 0.0, -0.25),
            "foot_centre_pos": (-0.0025, 0.112, 0.643)
        } # Was (-0.0025, 0.112, 0.743)
    )
    #pen_joint_torque                = RewTerm(func=mdp.joint_torques_l2, weight=-0.00008)
    #pen_joint_accel                 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-07)
    #pen_action_rate                 = RewTerm(func=mdp.action_rate_l2, weight=-16.0)
    #pen_action_smoothness           = RewTerm(func=mdp.ActionSmoothnessPenalty, weight=-0.04)


### --- MDP TERMINATIONS --- ###
@configclass
class TerminationsCfg:
    time_out        = DoneTerm(func=mdp.time_out, time_out=True)
    

### --- ENVIRONMENT CONFIGURATION --- ###
@configclass
class LegEnvCfg(ManagerBasedRLEnvCfg):
    scene: LegSceneCfg = LegSceneCfg(num_envs=4096, env_spacing=2.5) # Default num_envs and env_spacing.
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 20.0
        self.viewer.eye = (8.0, 8.0, 4.8)
        self.sim.dt = 0.005
        self.sim.render_interval = 4
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15