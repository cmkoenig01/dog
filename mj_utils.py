import mujoco
import numpy as np

def get_qpos_indices(model, joints=['joint_x', 'joint_y']):
    qpos_inds = np.array([model.joint(j).qposadr[0] for j in joints])
    return qpos_inds


def get_qvel_indices(model, joints=['joint_x', 'joint_y']):
    qvel_inds = np.array([model.joint(j).dofadr[0] for j in joints])
    return qvel_inds


def set_qpos_values(mdata, joint_inds, joint_vals):
    mdata.qpos[joint_inds] = joint_vals


def get_qpos_values(mdata, joint_inds):
    return mdata.qpos[joint_inds]


def set_qvel_values(mdata, joint_inds, joint_vels):
    mdata.qvel[joint_inds] = joint_vels


def get_qvel_values(mdata, joint_inds):
    return mdata.qvel[joint_inds]


def get_ctrl_indices(model, motors=['actuator_x', 'actuator_y']):
    ctrl_inds = np.array([model.actuator(motor).id for motor in motors])
    return ctrl_inds


def set_ctrl_values(mdata, ctrl_inds, ctrl_vals):
    mdata.ctrl[ctrl_inds] = ctrl_vals


def colliding_body_pairs(contact, model):
    pairs = [
        (
            model.body(model.geom(c.geom1).bodyid[0]).name,
            model.body(model.geom(c.geom2).bodyid[0]).name
        ) for c in contact
    ]
    return pairs

def is_in_collision(
    model,
    mdata,
    joint_inds,
    joint_vals,
):
    # set the robot configuration to a certain state
    set_qpos_values(mdata, joint_inds, joint_vals)
    # check collision
    mujoco.mj_step1(model, mdata)
    # cols = colliding_body_pairs(mdata.contact, model)
    return len(mdata.contact) > 1