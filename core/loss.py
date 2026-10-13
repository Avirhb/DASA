"""Original ABD-I/R cross-teaching objective; backend passed explicitly."""

from types import SimpleNamespace
import torch
import torch.nn.functional as F


def loss_step(model1, model2, batch_l, batch_u, iteration, device, config, abd):
    """Same ABD-I/R and CE/Dice cross-teaching terms as the existing trainer."""
    n = config["labeled_bs"]
    args = SimpleNamespace(
        labeled_bs=n,
        patch_size=config["abd_patch_size"],
        h_size=config["abd_grid_size"],
        w_size=config["abd_grid_size"],
        top_num=config["abd_top_num"],
    )
    volume = torch.cat([batch_l["image"], batch_u["image"]]).to(device)
    labels = torch.cat([batch_l["label"], batch_u["label"]]).to(device)
    strong = torch.cat([batch_l["image_strong"], batch_u["image_strong"]]).to(device)
    strong_labels = torch.cat([batch_l["label_strong"], batch_u["label_strong"]]).to(
        device
    )
    out1, out2 = model1(volume), model2(strong)
    soft1, soft2 = torch.softmax(out1, 1), torch.softmax(out2, 1)
    max1, max2 = soft1.detach().max(1)[0], soft2.detach().max(1)[0]
    pseudo1 = soft1[n:].detach().argmax(1)
    pseudo2 = soft2[n:].detach().argmax(1)
    sup_image, sup_labels = abd.ABD_I(
        max1, max2, volume, strong, labels, strong_labels, args
    )
    sup1, sup2 = model1(sup_image.unsqueeze(1)), model2(sup_image.unsqueeze(1))
    unsup_image = abd.ABD_R(max1, max2, volume, strong, out1[n:], out2[n:], args)
    unsup1, unsup2 = model1(unsup_image.unsqueeze(1)), model2(unsup_image.unsqueeze(1))
    unsup_soft1, unsup_soft2 = torch.softmax(unsup1, 1), torch.softmax(unsup2, 1)
    dice = abd.losses.DiceLoss(config["num_classes"])

    def supervised(logits, soft, target):
        return config["supervised_mix"] * (
            F.cross_entropy(logits, target.long()) + dice(soft, target.unsqueeze(1))
        )

    loss1 = supervised(out1[:n], soft1[:n], labels[:n])
    loss2 = supervised(out2[:n], soft2[:n], strong_labels[:n])
    loss3 = supervised(sup1, torch.softmax(sup1, 1), sup_labels)
    loss4 = supervised(sup2, torch.softmax(sup2, 1), sup_labels)
    cross1 = dice(soft1[n:], pseudo2.unsqueeze(1))
    cross2 = dice(soft2[n:], pseudo1.unsqueeze(1))
    cross3 = dice(unsup_soft1, unsup_soft2.detach().argmax(1).unsqueeze(1))
    cross4 = dice(unsup_soft2, unsup_soft1.detach().argmax(1).unsqueeze(1))
    weight = config["consistency"] * abd.ramps.sigmoid_rampup(
        iteration // config["ramp_interval"], config["consistency_rampup"]
    )
    total = loss1 + config["abd_supervised_weight"] * loss3 + weight * (cross1 + cross3)
    total = (
        total
        + loss2
        + config["abd_supervised_weight"] * loss4
        + weight * (cross2 + cross4)
    )
    return total, float(weight)
