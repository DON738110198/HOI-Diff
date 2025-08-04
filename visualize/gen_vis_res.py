import pytorch3d.transforms as transforms 
import argparse
import os
import numpy as np
import yaml
import random 
import json 
import copy 

import trimesh 

from matplotlib import pyplot as plt
from pathlib import Path

import torch

import torch.nn.functional as F

import pytorch3d.transforms as transforms 

from ema_pytorch import EMA
import sys
sys.path.append("../../")
sys.path.append("../")
from manip.data.cano_traj_dataset import  quat_ik_torch, quat_fk_torch

# from manip.model.transformer_object_motion_cond_diffusion import ObjectCondGaussianDiffusion 

from manip.vis.blender_vis_mesh_motion import run_blender_rendering_and_save2video, save_verts_faces_to_mesh_file_w_object

# from manip.lafan1.utils import quat_inv, quat_mul, quat_between, normalize, quat_normalize 

# from t2m_eval.evaluation_metrics import compute_metrics, determine_floor_height_and_contacts, compute_metrics_long_seq   
import argparse
import os
from visualize import vis_utils
import shutil
from tqdm import tqdm

from scipy.spatial.transform import Rotation
import random
torch.manual_seed(1)
random.seed(1)

from visualize.vis_utils import simplified_mesh
from manip.data.cano_traj_dataset import CanoObjectTrajDataset, quat_ik_torch, quat_fk_torch
def gen_vis_res_generic(self, all_res_list, data_dict, step, cond_mask, vis_gt=False, vis_tag=None, \
                text_anno=None, cano_quat=None, gen_long_seq=False, curr_object_name=None, dest_out_vid_path=None, dest_mesh_vis_folder=None, \
                save_obj_only=False):
        # all_res_list维度为216，没有contact
        # Prepare list used for evaluation. 
        human_jnts_list = []
        human_verts_list = [] 
        obj_verts_list = [] 
        trans_list = []
        human_mesh_faces_list = []
        obj_mesh_faces_list = [] 

        # all_res_list: N X T X (3+9) 
        num_seq = all_res_list.shape[0]

        pred_normalized_obj_trans = all_res_list[:, :, :3] # N X T X 3 
        pred_seq_com_pos = self.ds.de_normalize_obj_pos_min_max(pred_normalized_obj_trans)

        if self.use_random_frame_bps:
            # reference_obj_rot_mat = data_dict['reference_obj_rot_mat'] # N X 1 X 3 X 3 
            # 不需要这个了
            pred_obj_rel_rot_mat = all_res_list[:, :, 3:3+9].reshape(num_seq, -1, 3, 3) # N X T X 3 X 3
            # pred_obj_rot_mat = self.ds.rel_rot_to_seq(pred_obj_rel_rot_mat, reference_obj_rot_mat)
            pred_obj_rot_mat = pred_obj_rel_rot_mat

        num_joints = 24
        # denormalize 人物的joint position
        normalized_global_jpos = all_res_list[:, :, 3+9:3+9+num_joints*3].reshape(num_seq, -1, num_joints, 3)
        global_jpos = self.ds.de_normalize_jpos_min_max(normalized_global_jpos.reshape(-1, num_joints, 3))
        global_jpos = global_jpos.reshape(num_seq, -1, num_joints, 3) # N X T X 22 X 3 

        global_root_jpos = global_jpos[:, :, 0, :].clone() # N X T X 3 
        # 
        global_rot_6d = all_res_list[:, :, 3+9+24*3:3+9+24*3+22*6].reshape(num_seq, -1, 22, 6)
        global_rot_mat = transforms.rotation_6d_to_matrix(global_rot_6d) # N X T X 22 X 3 X 3 

        # trans2joint = data_dict['trans2joint'].to(all_res_list.device).squeeze(1) # BS X  3 
        seq_len = data_dict['seq_len'] # BS, should only be used during for single window generation. 
        # if all_res_list.shape[0] != trans2joint.shape[0]:
        #     trans2joint = trans2joint.repeat(num_seq, 1, 1) # N X 24 X 3 
        #     seq_len = seq_len.repeat(num_seq) # N 
        seq_len = seq_len.detach().cpu().numpy() # N 

        for idx in range(num_seq):
            curr_global_rot_mat = global_rot_mat[idx] # T X 22 X 3 X 3 
            curr_local_rot_mat = quat_ik_torch(curr_global_rot_mat) # T X 22 X 3 X 3 
            curr_local_rot_aa_rep = transforms.matrix_to_axis_angle(curr_local_rot_mat) # T X 22 X 3 
            
            curr_global_root_jpos = global_root_jpos[idx] # T X 3
            # 能否不要这个
            # curr_trans2joint = trans2joint[idx:idx+1].clone() # 1 X 3 
            
            # root_trans = curr_global_root_jpos + curr_trans2joint.to(curr_global_root_jpos.device) # T X 3 
            root_trans = curr_global_root_jpos

            # Generate global joint position 
            bs = 1
            betas = data_dict['betas'][0]
            gender = data_dict['gender'][0]
            
            
            curr_obj_rot_mat = pred_obj_rot_mat[idx] # T X 3 X 3 
            curr_obj_quat = transforms.matrix_to_quaternion(curr_obj_rot_mat)
            curr_obj_rot_mat = transforms.quaternion_to_matrix(curr_obj_quat) # Potentially avoid some prediction not satisfying rotation matrix requirements.

            if curr_object_name is not None: 
                object_name = curr_object_name 
            else:
                curr_seq_name = data_dict['seq_name'][0]
                object_name = data_dict['obj_name'][0]
          
            # Get human verts 
            mesh_jnts, mesh_verts, mesh_faces = \
                run_smplx_model(root_trans[None].cuda(), curr_local_rot_aa_rep[None].cuda(), \
                betas.cuda(), [gender], self.ds.bm_dict, return_joints24=True)

            # Get object verts 
            obj_rest_verts, obj_mesh_faces = self.ds.load_rest_pose_object_geometry(object_name)
            obj_rest_verts = torch.from_numpy(obj_rest_verts)

            obj_mesh_verts = self.ds.load_object_geometry_w_rest_geo(curr_obj_rot_mat, \
                        pred_seq_com_pos[idx], obj_rest_verts.float().to(pred_seq_com_pos.device))

            actual_len = seq_len[idx]

            human_jnts_list.append(mesh_jnts[0])
            human_verts_list.append(mesh_verts[0]) 
            obj_verts_list.append(obj_mesh_verts)
            trans_list.append(root_trans) 

            human_mesh_faces_list.append(mesh_faces)
            obj_mesh_faces_list.append(obj_mesh_faces) 
            

            if dest_mesh_vis_folder is None:
                if vis_tag is None:
                    dest_mesh_vis_folder = os.path.join(self.vis_folder, "blender_mesh_vis", str(step))
                else:
                    dest_mesh_vis_folder = os.path.join(self.vis_folder, vis_tag, str(step))
            
            if not os.path.exists(dest_mesh_vis_folder):
                os.makedirs(dest_mesh_vis_folder)

            if vis_gt:
                ball_mesh_save_folder = os.path.join(dest_mesh_vis_folder, \
                                "ball_objs_step_"+str(step)+"_bs_idx_"+str(idx)+"_gt")
                mesh_save_folder = os.path.join(dest_mesh_vis_folder, \
                                "objs_step_"+str(step)+"_bs_idx_"+str(idx)+"_gt")
                out_rendered_img_folder = os.path.join(dest_mesh_vis_folder, \
                                "imgs_step_"+str(step)+"_bs_idx_"+str(idx)+"_gt")
                out_vid_file_path = os.path.join(dest_mesh_vis_folder, \
                                "vid_step_"+str(step)+"_bs_idx_"+str(idx)+"_gt.mp4")


            if text_anno is not None:
                out_vid_file_path.replace(".mp4", "_"+text_anno.replace(" ", "_")+".mp4")
                
            # For faster debug visualization!!
            # mesh_verts = mesh_verts[:, ::30, :, :] # 1 X T X Nv X 3
            # obj_mesh_verts = obj_mesh_verts[::30, :, :] # T X Nv X 3 

            if cano_quat is not None:
                # mesh_verts: 1 X T X Nv X 3 
                # obj_mesh_verts: T X Nv' X 3 
                # cano_quat: K X 4 
                cano_quat_for_human = transforms.quaternion_invert(cano_quat[0:1][None].repeat(mesh_verts.shape[1], \
                                                            mesh_verts.shape[2], 1)) # T X Nv X 4 
                cano_quat_for_obj = transforms.quaternion_invert(cano_quat[0:1][None].repeat(obj_mesh_verts.shape[0], \
                                                            obj_mesh_verts.shape[1], 1)) # T X Nv X 4
                mesh_verts = transforms.quaternion_apply(cano_quat_for_human.to(mesh_verts.device), mesh_verts[0])
                obj_mesh_verts = transforms.quaternion_apply(cano_quat_for_obj.to(obj_mesh_verts.device), obj_mesh_verts) 

                save_verts_faces_to_mesh_file_w_object(mesh_verts.detach().cpu().numpy(), mesh_faces.detach().cpu().numpy(), \
                        obj_mesh_verts.detach().cpu().numpy(), obj_mesh_faces, mesh_save_folder)
            else: # here
                if gen_long_seq:
                    save_verts_faces_to_mesh_file_w_object(mesh_verts.detach().cpu().numpy()[0], \
                            mesh_faces.detach().cpu().numpy(), \
                            obj_mesh_verts.detach().cpu().numpy(), obj_mesh_faces, mesh_save_folder)
                else: # For single window here
                    save_verts_faces_to_mesh_file_w_object(mesh_verts.detach().cpu().numpy()[0][:seq_len[idx]], \
                            mesh_faces.detach().cpu().numpy(), \
                            obj_mesh_verts.detach().cpu().numpy()[:seq_len[idx]], obj_mesh_faces, mesh_save_folder)

                
            floor_blend_path = os.path.join(self.data_root_folder, "blender_files/floor_colorful_mat.blend")


            if dest_out_vid_path is None:
                dest_out_vid_path = out_vid_file_path.replace(".mp4", "_wo_scene.mp4")
            if not os.path.exists(dest_out_vid_path):
                if not vis_gt: # Skip GT visualiation 
                    if not save_obj_only:
                        run_blender_rendering_and_save2video(mesh_save_folder, out_rendered_img_folder, dest_out_vid_path, \
                                condition_folder=ball_mesh_save_folder, vis_object=True, vis_condition=True, \
                                scene_blend_path=floor_blend_path)

            if vis_gt: # here
                if not save_obj_only:
                    run_blender_rendering_and_save2video(mesh_save_folder, out_rendered_img_folder, dest_out_vid_path, \
                            condition_folder=ball_mesh_save_folder, vis_object=True, vis_condition=True, \
                            scene_blend_path=floor_blend_path)
                

            if idx > 1:
                break 

        return human_verts_list, human_jnts_list, trans_list, global_rot_mat, pred_seq_com_pos, pred_obj_rot_mat, \
        obj_verts_list, human_mesh_faces_list, obj_mesh_faces_list, dest_out_vid_path  

def run_smplx_model(root_trans, aa_rot_rep, betas, gender, bm_dict, return_joints24=True):
    # root_trans: BS X T X 3
    # aa_rot_rep: BS X T X 22 X 3 
    # betas: BS X 16
    # gender: BS 
    bs, num_steps, num_joints, _ = aa_rot_rep.shape
    if num_joints != 52:
        padding_zeros_hand = torch.zeros(bs, num_steps, 30, 3).to(aa_rot_rep.device) # BS X T X 30 X 3 
        aa_rot_rep = torch.cat((aa_rot_rep, padding_zeros_hand), dim=2) # BS X T X 52 X 3 

    aa_rot_rep = aa_rot_rep.reshape(bs*num_steps, -1, 3) # (BS*T) X n_joints X 3 
    betas = betas[:, None, :].repeat(1, num_steps, 1).reshape(bs*num_steps, -1) # (BS*T) X 16 
    gender = np.asarray(gender)[:, np.newaxis].repeat(num_steps, axis=1)
    gender = gender.reshape(-1).tolist() # (BS*T)

    smpl_trans = root_trans.reshape(-1, 3) # (BS*T) X 3  
    smpl_betas = betas # (BS*T) X 16
    smpl_root_orient = aa_rot_rep[:, 0, :] # (BS*T) X 3 
    smpl_pose_body = aa_rot_rep[:, 1:22, :].reshape(-1, 63) # (BS*T) X 63
    smpl_pose_hand = aa_rot_rep[:, 22:, :].reshape(-1, 90) # (BS*T) X 90 zero

    B = smpl_trans.shape[0] # (BS*T) 

    smpl_vals = [smpl_trans, smpl_root_orient, smpl_betas, smpl_pose_body, smpl_pose_hand]
    # batch may be a mix of genders, so need to carefully use the corresponding SMPL body model
    gender_names = ['male', 'female', "neutral"]
    pred_joints = []
    pred_verts = []
    prev_nbidx = 0
    cat_idx_map = np.ones((B), dtype=np.int64)*-1
    for gender_name in gender_names:
        gender_idx = np.array(gender) == gender_name
        nbidx = np.sum(gender_idx)

        cat_idx_map[gender_idx] = np.arange(prev_nbidx, prev_nbidx + nbidx, dtype=np.int64)
        prev_nbidx += nbidx

        gender_smpl_vals = [val[gender_idx] for val in smpl_vals]

        if nbidx == 0:
            # skip if no frames for this gender
            continue
        
        # reconstruct SMPL
        cur_pred_trans, cur_pred_orient, cur_betas, cur_pred_pose, cur_pred_pose_hand = gender_smpl_vals
        bm = bm_dict[gender_name]

        pred_body = bm(pose_body=cur_pred_pose, pose_hand=cur_pred_pose_hand, \
                betas=cur_betas, root_orient=cur_pred_orient, trans=cur_pred_trans)
        
        pred_joints.append(pred_body.Jtr)
        pred_verts.append(pred_body.v)

    # cat all genders and reorder to original batch ordering
    if return_joints24:
        x_pred_smpl_joints_all = torch.cat(pred_joints, axis=0) # () X 52 X 3 
        lmiddle_index= 28 
        rmiddle_index = 43 
        x_pred_smpl_joints = torch.cat((x_pred_smpl_joints_all[:, :22, :], \
            x_pred_smpl_joints_all[:, lmiddle_index:lmiddle_index+1, :], \
            x_pred_smpl_joints_all[:, rmiddle_index:rmiddle_index+1, :]), dim=1) 
    else:
        x_pred_smpl_joints = torch.cat(pred_joints, axis=0)[:, :num_joints, :]
        
    x_pred_smpl_joints = x_pred_smpl_joints[cat_idx_map] # (BS*T) X 22 X 3 

    x_pred_smpl_verts = torch.cat(pred_verts, axis=0)
    x_pred_smpl_verts = x_pred_smpl_verts[cat_idx_map] # (BS*T) X 6890 X 3 

    
    x_pred_smpl_joints = x_pred_smpl_joints.reshape(bs, num_steps, -1, 3) # BS X T X 22 X 3/BS X T X 24 X 3  
    x_pred_smpl_verts = x_pred_smpl_verts.reshape(bs, num_steps, -1, 3) # BS X T X 6890 X 3 

    mesh_faces = pred_body.f 
    
    return x_pred_smpl_joints, x_pred_smpl_verts, mesh_faces 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default = "/data3/wh/hoi_diffusion_model/HOI_Diff/save/behave_enc_512/samples_behave_enc_512_vertices/results.npy", required=True, help='stick figure mp4 file to be rendered.')
    parser.add_argument("--cuda", type=bool, default=True, help='')
    parser.add_argument("--device", type=int, default=0, help='')
    parser.add_argument("--obj_mesh_path", type=str, default='/data3/wh/hoi_diffusion_model/behave_t2m/object_mesh')
    params = parser.parse_args()

    input_path = params.input_path
    data_dict = np.load(input_path, allow_pickle=True)
    motion = data_dict.item()["motion"] # (10, 22, 3, 196)
    motion_obj = data_dict.item()["motion_obj"] # (10, 1, 6, 196)
    obj_name = data_dict.item()["obj_name"]

    # 处理obj的rot和trans
    vertices_list = []
    faces_list = []
    for b in range(motion.shape[0]):
        tmp_obj_name = obj_name[b].split("_")[2]
        mesh_path = os.path.join(params.obj_mesh_path, simplified_mesh[tmp_obj_name])
        tmp_mesh = trimesh.load(mesh_path)
        vertices = tmp_mesh.vertices
        faces = tmp_mesh.faces
        center = np.mean(vertices, 0)
        vertices -= center
        angle, trans = motion_obj[b, 0, :3], motion_obj[b, 0, 3:]
        rot = Rotation.from_rotvec(angle.transpose(1, 0)).as_matrix()
        vertices = np.matmul(vertices[np.newaxis], rot.transpose(0, 2, 1)[:, np.newaxis])[:, 0] + trans.transpose(1, 0)[:, np.newaxis]
        vertices = vertices.transpose(1, 2, 0)
        vertices_list.append(vertices)
        faces_list.append(faces)

    npy2obj_object = vis_utils.npy2obj_object(input_path, params.obj_mesh_path, 0, 0,
                                       device=params.device, cuda=params.cuda, if_color=True)

    # # human
    npy2obj = vis_utils.npy2obj(input_path, 0, 0,
                                device=params.device, cuda=params.cuda, if_color=True)
    out_npy_path = input_path.replace(".npy", "smpl_param.npy")
    out_obj_npy_path = input_path.replace(".npy", "obj_param")
    print('Saving SMPL params to [{}]'.format(os.path.abspath(out_npy_path)))   
    npy2obj.save_npy(out_npy_path)
    npy2obj_object.save_npy(out_obj_npy_path)
    