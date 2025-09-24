### --- IMPORTS --- ###
import math
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from .harold import HAROLD_CFG
from . import harold_cfg
from . import mdp

### --- SCENE DEFINITION --- ###
@configclass
class HaroldSceneCfg(InteractiveSceneCfg):

    # Ground Plane.
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0,100.0)),
    )

    # Robot.
    robot: ArticulationCfg = HAROLD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Sensors.
    contact_forces_L = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/HAROLD_R1_AS/HAROLD_R1_AS/LeftCalf", # Path to the left calf.
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        track_air_time=True,
    )
    contact_forces_R = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/HAROLD_R1_AS/HAROLD_R1_AS/RightCalf", # Path to the right calf.
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        track_air_time=True,
    )
    contact_forces_ALL = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/HAROLD_R1_AS/HAROLD_R1_AS/.*", # A path which includes all of the robot's prims.
        update_period=0.0,
        history_length=4,
        debug_vis=False,
        track_air_time=True,
    )
    calf_frame_sensor = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/HAROLD_R1_AS/HAROLD_R1_AS/Body", # Path to the body (the source frame).
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/HAROLD_R1_AS/HAROLD_R1_AS/LeftCalf", # Path to the left calf (a target frame).
                name="left_calf",
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/HAROLD_R1_AS/HAROLD_R1_AS/RightCalf", # Path to the right calf (a target frame).
                name="right_calf"
            ),
        ],
    )

    # Lights.
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

### --- MDP COMMANDS --- ###
@configclass
class CommandsCfg:
    # The commanded base linear and angular velocity setpoints.
    base_velocity   = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(harold_cfg.vel_resamp_per_min,harold_cfg.vel_resamp_per_max),
        rel_standing_envs=harold_cfg.fraction_still,
        heading_command=False, # Whether to use the heading command or angular velocity command.
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(harold_cfg.lin_vel_x_min,harold_cfg.lin_vel_x_max),
            lin_vel_y=(harold_cfg.lin_vel_y_min,harold_cfg.lin_vel_y_max),
            ang_vel_z=(harold_cfg.ang_vel_z_min,harold_cfg.ang_vel_z_max),
        ),
    )

    gait_command = mdp.UniformGaitCommandCfg(
        resampling_time_range=(harold_cfg.gait_resampling_period,harold_cfg.gait_resampling_period),
        debug_vis=False,
        ranges=mdp.UniformGaitCommandCfg.Ranges(
            frequencies=(harold_cfg.gait_freq_min,harold_cfg.gait_freq_max),
            offsets=(harold_cfg.gait_phase_offs_min,harold_cfg.gait_phase_offs_max),
            durations=(harold_cfg.gait_durations_min, harold_cfg.gait_durations_max),
            swing_height=(harold_cfg.gait_swing_height_min, harold_cfg.gait_swing_height_max)
        )
    )

### --- MDP ACTIONS --- ###
@configclass
class ActionsCfg:
    joint_effort = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=harold_cfg.joint_action_scale,
    )

### --- MDP OBSERVATIONS --- ###
@configclass
class ObservationsCfg:

    # Define the observation terms available to the agent.
    @configclass
    class PolicyCfg(ObsGroup):
        body_angular_velocity       = ObsTerm(func=mdp.base_ang_vel, history_length=harold_cfg.obs_history_length)
        proj_gravity                = ObsTerm(func=mdp.projected_gravity, history_length=harold_cfg.obs_history_length)
        joint_pos                   = ObsTerm(func=mdp.joint_pos, history_length=harold_cfg.obs_history_length)
        joint_vel                   = ObsTerm(func=mdp.joint_vel_rel, history_length=harold_cfg.obs_history_length)
        actions                     = ObsTerm(func=mdp.last_action, history_length=harold_cfg.obs_history_length)
        velocity_commands           = ObsTerm(
            func=mdp.generated_commands,
            history_length=harold_cfg.obs_history_length,
            params={
                "command_name": "base_velocity",
            }
        )
        gait_phase = ObsTerm(func=mdp.get_gait_phase, history_length=harold_cfg.obs_history_length)
        gait_command                = ObsTerm(
            func=mdp.get_gait_command,
            history_length=harold_cfg.obs_history_length,
            params={
                "command_name": "gait_command"
            }
        )

        # Post initialization.
        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()

### --- MDP EVENTS --- ###
@configclass
class EventCfg:
    # ON STARTUP:

    # ON RESET:
    reset_robot_position = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={},
    )

### --- MDP REWARDS --- ###
@configclass
class RewardsCfg:

    keep_balance = RewTerm(
        func=mdp.stay_alive,
        weight=harold_cfg.stay_alive_rew_weight
    )

    # r_v
    track_lin_vel_xy_exp =      RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=harold_cfg.xy_lin_vel_rew_weight,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.2)
        },
    )
    # r_w
    track_ang_vel_z_exp =       RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=harold_cfg.z_ang_vel_rew_weight,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.2)
        },
    )

    pen_base_height = RewTerm(
        func=mdp.base_height_l2,
        params={
            "target_height": harold_cfg.target_height,
        },
        weight=harold_cfg.base_height_rew_weight,
    )

    # r_vz
    lin_vel_z_l2 =              RewTerm(func=mdp.lin_vel_z_l2, weight=harold_cfg.z_lin_vel_rew_weight)
    # r_wxy
    ang_vel_xy_l2 =             RewTerm(func=mdp.ang_vel_xy_l2, weight=harold_cfg.xy_ang_vel_rew_weight)
    # (Not included in the BRAVER paper)
    flat_orientation_rew =      RewTerm(func=mdp.flat_orientation_l2, weight=harold_cfg.flat_body_weight)

    # Gait reward
    test_gait_reward = RewTerm(
        func=mdp.GaitReward,
        weight=1.0,
        params={
            "tracking_contacts_shaped_force": -2.0,
            "tracking_contacts_shaped_vel": -2.0,
            "gait_force_sigma": 25.0,
            "gait_vel_sigma": 0.25,
            "kappa_gait_probs": 0.05,
            "command_name": "gait_command",
            "sensor_cfg": SceneEntityCfg("contact_forces_ALL", body_names=["LeftFoot", "RightFoot"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["LeftFoot", "RightFoot"]),
        },
    )


### --- MDP TERMINATIONS --- ###
@configclass
class TerminationsCfg:
    time_out        = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact    = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces_ALL",
                body_names=["LeftThigh", "RightThigh", "LeftHip", "RightHip", "Body"],
            ),
            "threshold": 1.0,
        },
    )

### --- ENVIRONMENT CONFIGURATION --- ###
@configclass
class HaroldEnvCfg(ManagerBasedRLEnvCfg):
    scene: HaroldSceneCfg = HaroldSceneCfg(num_envs=4096, env_spacing=2.5) # Default num_envs and env_spacing.
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        self.decimation = harold_cfg.decimation_factor
        self.episode_length_s = harold_cfg.episode_length
        self.viewer.eye = harold_cfg.camera_pos
        self.sim.dt = harold_cfg.physics_time_step
        self.sim.render_interval = harold_cfg.render_interval_factor
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        if self.scene.contact_forces_L is not None:
            self.scene.contact_forces_L.update_period = self.sim.dt
        if self.scene.contact_forces_R is not None:
            self.scene.contact_forces_R.update_period = self.sim.dt
        if self.scene.contact_forces_ALL is not None:
            self.scene.contact_forces_ALL.update_period = self.sim.dt
        if self.scene.calf_frame_sensor is not None:
            self.scene.calf_frame_sensor.update_period = self.sim.dt
