import torch
from loader import Box
from models.unet_model import UNet
from evaluate import test, get_target_label
import cfg
import os
from models.mask import MaskGenerator
from copy import deepcopy
import random
from tqdm import tqdm

def reverse(model, bd_gen):
    inv_classifier = deepcopy(model)
    inv_classifier.eval()
    tmp_img = torch.ones([1, 3, opt.size, opt.size], device=device)
    tmp_feat = inv_classifier.from_input_to_features(tmp_img)
    feat_shape = tmp_feat.shape
    init_mask = torch.randn(feat_shape).to(device)
    m_gen = MaskGenerator(init_mask=init_mask, classifier=inv_classifier)
    opt_m = torch.optim.Adam([m_gen.mask_tanh], lr=0.01)
    for m in range(opt.mround):
        tloss = 0
        tloss_pos_pred = 0
        tloss_neg_pred = 0
        m_gen.train()
        inv_classifier.train()
        pbar = tqdm(cln_trainloader, desc="Decoupling Benign Features")
        for batch_idx, (cln_img, targets) in enumerate(pbar):
            opt_m.zero_grad()
            cln_img = cln_img.to(device)
            targets = targets.to(device)
            feat_mask = m_gen.get_raw_mask()
            cln_feat = inv_classifier.from_input_to_features(cln_img)
            mask_pos_pred = inv_classifier.from_features_to_output(feat_mask*cln_feat)
            remask_neg_pred = inv_classifier.from_features_to_output((1-feat_mask)*cln_feat)
            mask_norm = torch.norm(feat_mask, 1)

            loss_pos_pred = ce(mask_pos_pred, targets)
            loss_neg_pred = ce(remask_neg_pred, targets)            
            loss = loss_pos_pred - loss_neg_pred

            loss.backward()
            opt_m.step()

            tloss += loss.item()
            tloss_pos_pred += loss_pos_pred.item()
            tloss_neg_pred += loss_neg_pred.item()
            pbar.set_postfix({"round": "{:d}".format(n), 
                              "epoch": "{:d}".format(m),
                              "loss": "{:.4f}".format(tloss/(batch_idx+1)), 
                              "loss_pos_pred": "{:.4f}".format(tloss_pos_pred/(batch_idx+1)),
                              "loss_neg_pred": "{:.4f}".format(tloss_neg_pred/(batch_idx+1)),
                              "mask_norm": "{:.4f}".format(mask_norm)})
            
    feat_mask = m_gen.get_raw_mask().detach()

    for u in range(opt.uround):
        tloss = 0
        tloss_benign_feat = 0
        tloss_backdoor_feat = 0
        tloss_norm = 0
        m_gen.eval()
        bd_gen.train()
        inv_classifier.eval()
        pbar = tqdm(cln_trainloader, desc="Training Backdoor Generator")
        for batch_idx, (cln_img, targets) in enumerate(pbar):
            cln_img = cln_img.to(device)
            bd_gen_img = bd_gen(cln_img)
            cln_feat = inv_classifier.from_input_to_features(cln_img)
            bd_gen_feat = inv_classifier.from_input_to_features(bd_gen_img)
            loss_benign_feat = mse(feat_mask*cln_feat, feat_mask*bd_gen_feat)
            loss_backdoor_feat = mse((1-feat_mask)*cln_feat, (1-feat_mask)*bd_gen_feat)
            loss_norm = mse(cln_img, bd_gen_img)

            if loss_norm > opt.norm_bound or loss_benign_feat > opt.feat_bound:
                loss = loss_norm
            else:
                loss = -loss_backdoor_feat + 0.01*loss_benign_feat
                
            if n > 0:
                inv_tlabel = torch.ones_like(targets, device=device)*detected_tlabel
                bd_gen_pred = inv_classifier(bd_gen_img)
                loss += ce(bd_gen_pred, inv_tlabel)

            opt_bd.zero_grad()
            loss.backward()
            opt_bd.step()
            
            tloss += loss.item()
            tloss_benign_feat += loss_benign_feat.item()
            tloss_backdoor_feat += loss_backdoor_feat.item()
            tloss_norm += loss_norm.item()

            pbar.set_postfix({"round": "{:d}".format(n), 
                              "epoch": "{:d}".format(u),
                              "loss": "{:.4f}".format(tloss/(batch_idx+1)), 
                              "loss_bengin_feat": "{:.4f}".format(tloss_benign_feat/(batch_idx+1)),
                              "loss_backdoor_feat": "{:.4f}".format(tloss_backdoor_feat/(batch_idx+1)),
                              "loss_norm": "{:.4f}".format(tloss_norm/(batch_idx+1))})

def unlearn(model, bd_gen):
    classifier = model    
    for ul in range(opt.ul_round):
        tloss = 0
        tloss_pred = 0
        tloss_feat = 0
        bd_gen.eval()
        classifier.train()
        pbar = tqdm(cln_trainloader, desc="Unlearning")
        for batch_idx, (cln_img, targets) in enumerate(pbar):
            targets = targets.to(device)
            bd_gen_num = int(0.1*cln_img.shape[0] + 1)
            bd_gen_list = random.sample(range(cln_img.shape[0]), bd_gen_num)
            cln_img = cln_img.to(device)
            bd_gen_img = deepcopy(cln_img).to(device)
            bd_gen_img[bd_gen_list] = bd_gen(bd_gen_img[bd_gen_list])

            cln_feat = classifier.from_input_to_features(cln_img)
            bd_gen_feat = classifier.from_input_to_features(bd_gen_img)
            bd_gen_pred = classifier.from_features_to_outpu
            t(bd_gen_feat)
            
            loss_pred = ce(bd_gen_pred, targets)
            loss_feat = mse(cln_feat, bd_gen_feat)
            loss = loss_pred + loss_feat

            opt_cls.zero_grad()
            loss.backward()
            opt_cls.step()
           
            tloss += loss.item()
            tloss_pred += loss_pred.item()
            tloss_feat += loss_feat.item()
            pbar.set_postfix({"round": "{:d}".format(n), 
                              "epoch": "{:d}".format(ul),
                              "loss": "{:.4f}".format(tloss/(batch_idx+1)), 
                              "loss_pred": "{:.4f}".format(tloss_pred/(batch_idx+1)),
                              "loss_feat": "{:.4f}".format(tloss_feat/(batch_idx+1))})
                        
        if ((ul+1) % 10) == 0:
            test(testloader=cln_testloader, testmodel=classifier, box=box, poisoned=False, poitarget=False, name="BA")
            test(testloader=cln_testloader, testmodel=classifier, box=box, poisoned=True, poitarget=True, passlabel=box.tlabel, name="ASR")


if __name__ == "__main__":
    opt = cfg.get_arguments().parse_args()
    device = opt.device
    box = Box(opt)
    save_path = box.get_save_path()
    _, _, classifier = box.get_state_dict()
    # optcls = torch.optim.SGD(classifier.parameters(), lr = 1e-3)
    opt_cls = torch.optim.Adam(classifier.parameters(), lr = 1e-4)
    bd_gen = UNet(n_channels=3, num_classes=3, base_filter_num=32, num_blocks=4)
    bd_gen.load_state_dict(torch.load(os.path.join(save_path, "pretrain/init_generator.pt"), map_location=torch.device('cpu')))
    bd_gen = bd_gen.to(device)
    opt_bd = torch.optim.Adam(bd_gen.parameters(), lr=opt.gen_lr)

    cln_trainloader = box.get_dataloader(train="clean", batch_size=opt.batch_size, shuffle=True)
    cln_testloader = box.get_dataloader(train="test", batch_size=opt.batch_size, shuffle=False)

    mse = torch.nn.MSELoss()
    ce = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax()

    detected_tlabel = None
    for n in range(opt.nround):
        reverse(classifier, bd_gen)
        if n == 0:
            detected_tlabel = get_target_label(testloader=cln_trainloader, testmodel=classifier, box=box, midmodel=bd_gen)
        elif opt.earlystop:
            checked_tlabel = get_target_label(testloader=cln_trainloader, testmodel=classifier, box=box, midmodel=bd_gen)
            if checked_tlabel != detected_tlabel:
                break
        unlearn(classifier, bd_gen)