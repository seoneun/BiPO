import os
import time

from tqdm import tqdm
import clip
import numpy
import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg

import visualize.plot_3d_global as plot_3d
from utils.motion_process import recover_from_ric


def tensorborad_add_video_xyz(writer, xyz, nb_iter, tag, nb_vis=4, title_batch=None, outname=None):
    xyz = xyz[:1]
    bs, seq = xyz.shape[:2]
    xyz = xyz.reshape(bs, seq, -1, 3)
    plot_xyz = plot_3d.draw_to_batch(xyz.cpu().numpy(),title_batch, outname)
    plot_xyz =np.transpose(plot_xyz, (0, 1, 4, 2, 3)) 
    writer.add_video(tag, plot_xyz, nb_iter, fps = 20)


@torch.no_grad()        
def evaluation_vqvae(out_dir, val_loader, net, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, eval_wrapper, draw = True, save = True, savegif=False, savenpy=False) : 
    """
    Evaluate the VQVAE, used in train and test.
    Compute the FID, DIV, and R-Precision.
    """
    net.eval()
    nb_sample = 0
    
    draw_org = []
    draw_pred = []
    draw_text = []


    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    for batch in val_loader:

        # Get motion and parts. We use parts to represent parts' motion.
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, name, parts = batch

        motion = motion.cuda()

        if torch.isnan(motion).sum() > 0 or torch.isinf(motion).sum() > 0:
            print('Detected NaN or Inf in raw motion data')
            print('NaN elem numbers:', torch.isnan(motion).sum())
            print('Inf elem numbers:', torch.isinf(motion).sum())
            print('motion:', motion)

        # (text, motion) ==> (text_emb, motion_emb)
        #   motion is normalized.
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)

        if torch.isnan(em).sum() > 0:
            print('Detected NaN in em (embedding of motion), replace NaN with 0.0')
            print('NaN elem numbers:', torch.isnan(em).sum())
            print('em:', em)
            em = torch.nan_to_num(em)  # use default param to replace nan with 0.0. Require pytorch >= 1.8.0


        bs, seq = motion.shape[0], motion.shape[1]

        num_joints = 21 if motion.shape[-1] == 251 else 22
        
        pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        for i in range(bs):

            # [gt_motion] (augmented representation) ==[de-norm, convert]==> [gt_xyz] (xyz representation)
            pose = val_loader.dataset.inv_transform(motion[i:i+1, :m_length[i], :].detach().cpu().numpy())
            pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)


            # Preprocess parts: get single sample from the batch
            single_parts = []
            for p in parts:
                single_parts.append(p[i:i+1, :m_length[i]].cuda())

            # (parts, GT) ==> (reconstruct_parts)
            #   parts is normalized.
            pred_parts, loss_commit_list, perplexity_list = net(single_parts)

            # pred_pose, loss_commit, perplexity = net(motion[i:i+1, :m_length[i]])

            # pred_parts ==> whole_motion
            #   todo: support different shared_joint_rec_mode in the parts2whole function
            pred_pose = val_loader.dataset.parts2whole(pred_parts, mode=val_loader.dataset.dataset_name)

            # de-normalize reconstructed motion
            pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())

            # Convert to xyz representation
            #   todo: maybe we should support the recover_from_rot, not only ric.
            pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)
            
            if savenpy:
                np.save(os.path.join(out_dir, name[i]+'_gt.npy'), pose_xyz[:, :m_length[i]].cpu().numpy())
                np.save(os.path.join(out_dir, name[i]+'_pred.npy'), pred_xyz.detach().cpu().numpy())

            pred_pose_eval[i:i+1,:m_length[i],:] = pred_pose

            if i < min(4, bs):
                draw_org.append(pose_xyz)
                draw_pred.append(pred_xyz)
                draw_text.append(caption[i])

        # pred_pose_eval is normalized
        et_pred, em_pred = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length)

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)
            
        temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()

    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    logger.info(msg)
    
    if draw:
        writer.add_scalar('./Test/FID', fid, nb_iter)
        writer.add_scalar('./Test/Diversity', diversity, nb_iter)
        writer.add_scalar('./Test/top1', R_precision[0], nb_iter)
        writer.add_scalar('./Test/top2', R_precision[1], nb_iter)
        writer.add_scalar('./Test/top3', R_precision[2], nb_iter)
        writer.add_scalar('./Test/matching_score', matching_score_pred, nb_iter)

    
        if nb_iter % 5000 == 0 : 
            for ii in range(4):
                tensorborad_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/org_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'gt'+str(ii)+'.gif')] if savegif else None)
            
        if nb_iter % 5000 == 0 : 
            for ii in range(4):
                tensorborad_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/pred_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'pred'+str(ii)+'.gif')] if savegif else None)   

    
    if fid < best_fid : 
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        logger.info(msg)
        best_fid, best_iter = fid, nb_iter
        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_fid.pth'))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div) : 
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        logger.info(msg)
        best_div = diversity
        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_div.pth'))

    if R_precision[0] > best_top1 : 
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        logger.info(msg)
        best_top1 = R_precision[0]
        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_top1.pth'))

    if R_precision[1] > best_top2 : 
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        logger.info(msg)
        best_top2 = R_precision[1]
    
    if R_precision[2] > best_top3 : 
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        logger.info(msg)
        best_top3 = R_precision[2]
    
    if matching_score_pred < best_matching : 
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        logger.info(msg)
        best_matching = matching_score_pred
        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_matching.pth'))

    if save:
        torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    net.train()
    return best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger



@torch.no_grad()
def evaluation_transformer_batch(out_dir, val_loader, net, trans, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, clip_model, eval_wrapper, draw = True, save = True, savegif=False, embedding_motions=None, tmr=None, is_tmr=False, file_list=None):
    """
    This is used for evaluate GPT at training stage.
    It excludes the multi-modality evaluation by simply set a circle only at 1 time.
    """
    trans.eval()
    nb_sample = 0

    draw_org = []
    draw_pred = []
    draw_text = []
    draw_text_pred = []

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0
    for i in range(1):
        for batch in tqdm(val_loader):

            word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name, parts = batch

            bs, seq = pose.shape[:2]
            num_joints = 21 if pose.shape[-1] == 251 else 22

            if not is_tmr:

                text = clip.tokenize(clip_text, truncate=True).cuda()

                feat_clip_text = clip_model.encode_text(text).float()  # (B, 512)

            # TMR
            
            if embedding_motions is not None and tmr is not None:
                feat_tmr_text = tmr.encode_text(clip_text)

                cos_sim = F.cosine_similarity(feat_tmr_text.unsqueeze(1), embedding_motions.unsqueeze(0), dim=-1)
                indices = torch.argmax(cos_sim, dim=-1)
                retreivaled_motions = embedding_motions[indices]

                feat_clip_text = torch.concat([feat_clip_text, retreivaled_motions], dim=-1)
            

            '''
            if embedding_motions is not None and tmr is not None:
                feat_tmr_text = tmr.encode_text(clip_text)
                cos_sim = F.cosine_similarity(feat_tmr_text.unsqueeze(1), embedding_motions.unsqueeze(0), dim=-1)

                batch_size = feat_tmr_text.shape[0]

                quanzied_motion_path = 'output/base/VQVAE-ParCo-t2m-default/prob_0.35/quantized_dataset_t2m'

                motion_results = [[] for _ in range(6)]

                for i in range(batch_size):
                    sorted_indices = torch.argsort(cos_sim[i], dim=-1, descending=True).squeeze()


                    for r_index in sorted_indices:
                        r_motion_file_name = file_list[r_index.item()]
                        r_motion_file_name = r_motion_file_name[:r_motion_file_name.index('.npy')]
                        r_motion_file_name_list = [f'{r_motion_file_name}_Backbone.npy',
                                f'{r_motion_file_name}_L_Arm.npy',
                                f'{r_motion_file_name}_L_Leg.npy',
                                f'{r_motion_file_name}_R_Arm.npy',
                                f'{r_motion_file_name}_R_Leg.npy',
                                f'{r_motion_file_name}_Root.npy']

                        file_exists = all(os.path.exists(os.path.join(quanzied_motion_path, r)) for r in r_motion_file_name_list)

                        if file_exists:
                            r_motion_file_list = [torch.from_numpy(np.load(os.path.join(quanzied_motion_path, r))) for r in r_motion_file_name_list]

                            for idx, motion in enumerate(r_motion_file_list):
                                motion_results[idx].append(motion)
                            break
            '''


            if is_tmr:
                feat_clip_text = clip_model.encode_text(clip_text)
            
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
            pred_len = torch.ones(bs).long()


            # [Text-to-motion Generation] get generated parts' token sequence
            # get parts_index_motion given the feat_clip_text
            batch_parts_index_motion = trans.sample_batch(feat_clip_text, False, 0.0)  # List: [(B, seq_len), ..., (B, seq_len)]

            max_motion_seq_len = batch_parts_index_motion[0].shape[1]

            for k in range(bs):

                min_motion_seq_len = max_motion_seq_len
                parts_index_motion = []
                for part_index, name in zip(batch_parts_index_motion, ['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm']):

                    # get one sample
                    part_index = part_index[k:k+1]  # (1, seq_len)

                    # find the earliest end token position
                    idx = torch.nonzero(part_index == trans.parts_code_nb[name])

                    # # Debug
                    # print('part_index:', part_index)
                    # print('nonzero_idx', idx)

                    if idx.numel() == 0:
                        motion_seq_len = max_motion_seq_len
                    else:
                        min_end_idx = idx[:,1].min()
                        motion_seq_len = min_end_idx

                    if motion_seq_len < min_motion_seq_len:
                        min_motion_seq_len = motion_seq_len

                    parts_index_motion.append(part_index)

                # Truncate
                for j in range(len(parts_index_motion)):
                    if min_motion_seq_len == 0:
                        # assign a nonsense motion index to handle length is 0 issue.
                        parts_index_motion[j] = torch.ones(1,1).cuda().long()  # (B, seq_len) B==1, seq_len==1
                    else:
                        parts_index_motion[j] = parts_index_motion[j][:,:min_motion_seq_len]


                '''
                index_motion: (B, nframes). Here: B == 1, nframes == predicted_length
                '''

                # [Token-to-RawMotion with VQ-VAE decoder] get each parts' raw motion
                parts_pred_pose = net.forward_decoder(parts_index_motion)  # (B, pred_nframes, parts_sk_dim)
                #   todo: support different shared_joint_rec_mode in the parts2whole function
                pred_pose = val_loader.dataset.parts2whole(parts_pred_pose, mode=val_loader.dataset.dataset_name)  # (B, pred_nframes, raw_motion_dim)

                cur_len = pred_pose.shape[1]

                pred_len[k] = min(cur_len, seq)  # save the min len

                # It's actually should use pred_len[k] to replace cur_len and seq for understanding convenience
                #   Below code seems equal to use pred_len[k].
                #   But should not change it to keep the same test code with T2M-GPT.
                pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]

                if draw:
                    if i == 0 and k < 4:
                        pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
                        pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)
                        draw_pred.append(pred_xyz)
                        draw_text_pred.append(clip_text[k])


            et_pred, em_pred = eval_wrapper.get_co_embeddings(
                word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)

            if i == 0:
                pose = pose.cuda().float()

                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                if draw:
                    pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
                    pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)


                    for j in range(min(4, bs)):
                        draw_org.append(pose_xyz[j][:m_length[j]].unsqueeze(0))
                        draw_text.append(clip_text[j])

                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                R_precision += temp_R
                matching_score_pred += temp_match

                nb_sample += bs


    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample


    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    logger.info(msg)


    if draw:
        writer.add_scalar('./Test/FID', fid, nb_iter)
        writer.add_scalar('./Test/Diversity', diversity, nb_iter)
        writer.add_scalar('./Test/top1', R_precision[0], nb_iter)
        writer.add_scalar('./Test/top2', R_precision[1], nb_iter)
        writer.add_scalar('./Test/top3', R_precision[2], nb_iter)
        writer.add_scalar('./Test/matching_score', matching_score_pred, nb_iter)


        if nb_iter % 10000 == 0 :
            for ii in range(4):
                tensorborad_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/org_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'gt'+str(ii)+'.gif')] if savegif else None)

        if nb_iter % 10000 == 0 :
            for ii in range(4):
                tensorborad_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/pred_eval'+str(ii), nb_vis=1, title_batch=[draw_text_pred[ii]], outname=[os.path.join(out_dir, 'pred'+str(ii)+'.gif')] if savegif else None)


    if fid < best_fid :
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        logger.info(msg)
        best_fid, best_iter = fid, nb_iter
        if save:
            torch.save({'trans' : trans.state_dict()}, os.path.join(out_dir, 'net_best_fid.pth'))

    if matching_score_pred < best_matching :
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        logger.info(msg)
        best_matching = matching_score_pred
        if save:
            torch.save({'trans' : trans.state_dict()}, os.path.join(out_dir, 'net_best_matching.pth'))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div) :
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        logger.info(msg)
        best_div = diversity
        if save:
            torch.save({'trans' : trans.state_dict()}, os.path.join(out_dir, 'net_best_div.pth'))

    if R_precision[0] > best_top1 :
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        logger.info(msg)
        best_top1 = R_precision[0]
        if save:
            torch.save({'trans' : trans.state_dict()}, os.path.join(out_dir, 'net_best_top1.pth'))

    if R_precision[1] > best_top2 :
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        logger.info(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3 :
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        logger.info(msg)
        best_top3 = R_precision[2]

    if save:
        torch.save({'trans' : trans.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    trans.train()
    return best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger



@torch.no_grad()
def evaluation_transformer_test_batch(out_dir, val_loader, net, trans, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_multi, clip_model, eval_wrapper, draw = True, save = True, savegif=False, savenpy=False, mmod_gen_times=30, skip_mmod=False, embedding_motions=None, tmr=None, is_tmr=False):

    trans.eval()

    if skip_mmod:
        mmod_gen_times = 1

    nb_sample = 0

    draw_org = []
    draw_pred = []
    draw_text = []
    draw_text_pred = []
    draw_name = []

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0

    for batch in tqdm(val_loader):

        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, sample_name, parts = batch
        bs, seq = pose.shape[:2]
        num_joints = 21 if pose.shape[-1] == 251 else 22

        if not is_tmr:
            text = clip.tokenize(clip_text, truncate=True).cuda()
            feat_clip_text = clip_model.encode_text(text).float()

        if is_tmr:
            feat_clip_text = clip_model.encode_text(clip_text)

        # TMR
        
        if embedding_motions is not None and tmr is not None:
            feat_tmr_text = tmr.encode_text(clip_text)

            cos_sim = F.cosine_similarity(feat_tmr_text.unsqueeze(1), embedding_motions.unsqueeze(0), dim=-1)
            indices = torch.argmax(cos_sim, dim=-1)
            retreivaled_motions = embedding_motions[indices]

            feat_clip_text = torch.concat([feat_clip_text, retreivaled_motions], dim=-1)
        

        motion_multimodality_batch = []
        for i in range(mmod_gen_times):  # mmod_gen_times default: 30
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
            pred_len = torch.ones(bs).long()


            # [Text-to-motion Generation] get generated parts' token sequence
            # get parts_index_motion given the feat_clip_text
            batch_parts_index_motion = trans.sample_batch(feat_clip_text, True, 0.0)  # List: [(B, seq_len), ..., (B, seq_len)]

            max_motion_seq_len = batch_parts_index_motion[0].shape[1]


            for k in range(bs):

                min_motion_seq_len = max_motion_seq_len
                parts_index_motion = []
                for part_index, name in zip(batch_parts_index_motion, ['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm']):

                    # get one sample
                    part_index = part_index[k:k+1]  # (1, seq_len)

                    # find the earliest end token position
                    idx = torch.nonzero(part_index == trans.parts_code_nb[name])

                    # # Debug
                    # print('part_index:', part_index)
                    # print('nonzero_idx', idx)

                    if idx.numel() == 0:
                        motion_seq_len = max_motion_seq_len
                    else:
                        min_end_idx = idx[:,1].min()
                        motion_seq_len = min_end_idx

                    if motion_seq_len < min_motion_seq_len:
                        min_motion_seq_len = motion_seq_len

                    parts_index_motion.append(part_index)

                # Truncate
                for j in range(len(parts_index_motion)):
                    if min_motion_seq_len == 0:
                        # assign a nonsense motion index to handle length is 0 issue.
                        parts_index_motion[j] = torch.ones(1,1).cuda().long()  # (B, seq_len) B==1, seq_len==1
                    else:
                        parts_index_motion[j] = parts_index_motion[j][:,:min_motion_seq_len]



                # [Token-to-RawMotion with VQ-VAE decoder] get each parts' raw motion
                parts_pred_pose = net.forward_decoder(parts_index_motion)  # (B, pred_nframes, parts_sk_dim)
                #   todo: support different shared_joint_rec_mode in the parts2whole function
                pred_pose = val_loader.dataset.parts2whole(parts_pred_pose, mode=val_loader.dataset.dataset_name)  # (B, pred_nframes, raw_motion_dim)

                cur_len = pred_pose.shape[1]

                pred_len[k] = min(cur_len, seq)
                pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]

                if i == 0 and (draw or savenpy):
                    pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
                    pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)

                    if savenpy:
                        if not os.path.exists(os.path.join(out_dir, 'pred')):
                             os.makedirs(os.path.join(out_dir, 'pred'))
                        np.save(os.path.join(out_dir, 'pred', sample_name[k]+'_pred.npy'), pred_xyz.detach().cpu().numpy())
                        pose_vis = plot_3d.draw_to_batch(pred_xyz.detach().cpu().numpy(), [clip_text[k]], [os.path.join(out_dir, 'pred', sample_name[k]+'_pred.gif')])

                    if draw:
                        if i == 0:
                            draw_pred.append(pred_xyz)
                            draw_text_pred.append(clip_text[k])
                            draw_name.append(sample_name[k])

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)

            motion_multimodality_batch.append(em_pred.reshape(bs, 1, -1))

            if i == 0:
                pose = pose.cuda().float()

                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                if draw or savenpy:
                    pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
                    pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)

                    if savenpy:
                        for j in range(bs):
                            if not os.path.exists(os.path.join(out_dir, 'gt')):
                                os.makedirs(os.path.join(out_dir, 'gt'))
                            np.save(os.path.join(out_dir, 'gt', sample_name[j]+'_gt.npy'), pose_xyz[j][:m_length[j]].unsqueeze(0).cpu().numpy())
                            pose_vis = plot_3d.draw_to_batch(pose_xyz[j][:m_length[j]].unsqueeze(0).cpu().numpy(), [clip_text[j]], [os.path.join(out_dir, 'gt', sample_name[j]+'_gt.gif')])

                    if draw:
                        for j in range(bs):
                            draw_org.append(pose_xyz[j][:m_length[j]].unsqueeze(0))
                            draw_text.append(clip_text[j])

                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                R_precision += temp_R
                matching_score_pred += temp_match

                nb_sample += bs

        motion_multimodality.append(torch.cat(motion_multimodality_batch, dim=1))

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    multimodality = 0
    if not skip_mmod:
        print('Calculate multimodality...')
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}, multimodality. {multimodality:.4f}"
    logger.info(msg)


    if draw:
        for ii in range(len(draw_org)):
            tensorborad_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/'+draw_name[ii]+'_org', nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, draw_name[ii]+'_skel_gt.gif')] if savegif else None)

            tensorborad_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/'+draw_name[ii]+'_pred', nb_vis=1, title_batch=[draw_text_pred[ii]], outname=[os.path.join(out_dir, draw_name[ii]+'_skel_pred.gif')] if savegif else None)

    trans.train()
    return fid, best_iter, diversity, R_precision[0], R_precision[1], R_precision[2], matching_score_pred, multimodality, writer, logger


# (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists



def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
#         print(correct_vec, bool_mat[:, i])
        correct_vec = (correct_vec | bool_mat[:, i])
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    matching_score = dist_mat.trace()
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0), matching_score
    else:
        return top_k_mat, matching_score

def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()



def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)



def calculate_activation_statistics(activations):

    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_frechet_feature_distance(feature_list1, feature_list2):
    feature_list1 = np.stack(feature_list1)
    feature_list2 = np.stack(feature_list2)

    # normalize the scale
    mean = np.mean(feature_list1, axis=0)
    std = np.std(feature_list1, axis=0) + 1e-10
    feature_list1 = (feature_list1 - mean) / std
    feature_list2 = (feature_list2 - mean) / std

    dist = calculate_frechet_distance(
        mu1=np.mean(feature_list1, axis=0), 
        sigma1=np.cov(feature_list1, rowvar=False),
        mu2=np.mean(feature_list2, axis=0), 
        sigma2=np.cov(feature_list2, rowvar=False),
    )
    return dist