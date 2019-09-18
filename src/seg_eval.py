import numpy as np
import copy
import nibabel as nib

# calculate evaluation metrics for segmentation
def seg_eval_metric(pred_label, gt_label, output_chn):
    class_n = np.unique(gt_label)
    # dice
    dice_c = dice_n_class(move_img=pred_label, refer_img=gt_label, output_chn=output_chn)
    return dice_c

# dice value
def dice_n_class(move_img, refer_img, output_chn):
    # list of classes
    c_list_old = np.unique(refer_img)
    # for those class not in the Gt, set dice to zero
    c_list = np.arange(output_chn)
    dice_c = []
    for c in range(len(c_list)):
        # intersection
        ints = np.sum(((move_img == c_list[c]) * 1) * ((refer_img == c_list[c]) * 1))
        # sum
        sums = np.sum(((move_img == c_list[c]) * 1) + ((refer_img == c_list[c]) * 1)) + 0.0001
        dice_c.append((2.0 * ints) / sums)

    return dice_c


# conformity value
def conform_n_class(move_img, refer_img):
    # list of classes
    c_list = np.unique(refer_img)

    conform_c = []
    for c in range(len(c_list)):
        # intersection
        ints = np.sum(((move_img == c_list[c]) * 1) * ((refer_img == c_list[c]) * 1))
        # sum
        sums = np.sum(((move_img == c_list[c]) * 1) + ((refer_img == c_list[c]) * 1)) + 0.0001
        # dice
        dice_temp = (2.0 * ints) / sums
        # conformity
        conform_temp = (3*dice_temp - 2) / dice_temp

        conform_c.append(conform_temp)

    return conform_c


# Jaccard index
def jaccard_n_class(move_img, refer_img, output_chn):
    # list of classes
    c_list_old = np.unique(refer_img)
    # c_list = [0, 1, 2, 3]
    c_list = np.arange(output_chn)

    jaccard_c = []
    for c in range(len(c_list)):
        move_img_c = (move_img == c_list[c])
        refer_img_c = (refer_img == c_list[c])
        # intersection
        ints = np.sum(np.logical_and(move_img_c, refer_img_c)*1)
        # union
        uni = np.sum(np.logical_or(move_img_c, refer_img_c)*1) + 0.0001

        jaccard_c.append(ints / uni)

    return jaccard_c


# precision and recall
def precision_recall_n_class(move_img, refer_img):
    # list of classes
    c_list = np.unique(refer_img)

    precision_c = []
    recall_c = []
    for c in range(len(c_list)):
        move_img_c = (move_img == c_list[c])
        refer_img_c = (refer_img == c_list[c])
        # intersection
        ints = np.sum(np.logical_and(move_img_c, refer_img_c)*1)
        # precision
        prec = ints / (np.sum(move_img_c*1) + 0.001)
        # recall
        recall = ints / (np.sum(refer_img_c*1) + 0.001)

        precision_c.append(prec)
        recall_c.append(recall)

    return precision_c, recall_c

# Sensitivity(recall of the positive)
def sensitivity(pred, gt, output_chn):
    '''
    calculate the sensitivity and the specificity
    :param pred: predictions
    :param gt:  ground truth
    :param output_chn: categories (including background)
    :return: A list contains sensitivities and the specificity, the first item is specificity
    and the others are sensitivities  of other categories
    '''
    s_list = np.arange(output_chn)
    sensitivity_s = []
    for s in range(output_chn):
        # TP
        TP = np.sum((pred == s_list[s])*(gt == s_list[s]))
        # FN
        FN = np.sum((pred != s_list[s])*(gt == s_list[s]))
        # sensitivity &specificity(for category 0 means specificity, while others means sensitivity)
        sensitivity = TP/(TP+FN+0.0001)
        sensitivity_s.append(sensitivity)
    return sensitivity_s

if __name__ == '__main__':
    pred = np.array([[1,1,1,1],[1,1,1,1],[0,0,0,0]])
    gt = np.array([[0,0,0,0],[1,1,1,1],[0,1,1,0]])
    sensi = sensitivity(pred, gt, 2)
    a =1