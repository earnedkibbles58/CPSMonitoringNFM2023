import math
import random



def calibrated_perception(gt_label): # gt_label needs to be an index
    num_classes = 3 ## hardcoded for 3 classes, could easily generalize though

    # conf1 = random.uniform(0.4,0.5)
    # conf2 = random.uniform(0,1-conf1)
    # conf3 = 1-conf1-conf2

    conf1 = random.uniform(0.4,0.5)
    conf2 = (1-conf1)/2
    conf3 = conf2

    # conf1 = 0.4
    # conf2 = 0.3
    # conf3 = 0.3


    temp_labels = list(range(num_classes)) #[0,1,2] ## HARDCODED
    temp_labels.remove(gt_label)
    # print(GTs)
    # print(temp_labels)
    randn = random.random()


    
    if randn <= conf1:
        pred_label1 = gt_label
    elif randn <= conf1 + (1-conf1)/2:
        pred_label1 = temp_labels[0]
    else:
        pred_label1 = temp_labels[1]
    

    temp_labels = list(range(num_classes))
    temp_labels.remove(pred_label1)
    if pred_label1 == gt_label:
        pred_label2 = random.choice(temp_labels)
        temp_labels.remove(pred_label2)
        pred_label3 = temp_labels[0]
    else:
        if random.random() <= conf2/(conf2+conf3):
            pred_label2 = gt_label
            temp_labels.remove(pred_label2)
            pred_label3 = temp_labels[0]
        else:
            pred_label3 = gt_label
            temp_labels.remove(pred_label3)
            pred_label2 = temp_labels[0]

    preds = [pred_label1, pred_label2, pred_label3]
    confs = [conf1, conf2, conf3]

    ## return in order, so just need to return the confs

    sorted_confs = []
    for label in range(num_classes):
        sorted_confs.append(confs[preds.index(label)])
    return sorted_confs


def get_bin(conf,num_bins):

    # print(conf)
    for j in range(num_bins):
        # print((j+1)*(1/num_bins))
        if conf<(j+1)*(1/num_bins):
            return j
    return num_bins-1

    
def main():

    random.seed(3476546556)
    numTest = 100000

    GTs = [0,1,2]
    obj_strs = ["nothing","rock","car"]
    
    all_confs = []
    all_GT_labels = []
    all_pred_labels = []

    for i in range(numTest):
        conf = random.uniform(0.5,1)
        conf2 = random.uniform(0,1-conf)
        conf3 = 1-conf-conf2

        gt_label = random.choice(GTs)
        # print(GTs)
        temp_labels = [0,1,2]
        temp_labels.remove(gt_label)
        # print(GTs)
        # print(temp_labels)
        randn = random.random()
        
        preds = []
        if randn <= conf:
            pred_label = gt_label
        elif randn <= conf + (1-conf)/2:
            pred_label = temp_labels[0]
        else:
            pred_label = temp_labels[1]
        preds.append(pred_label)

        temp_labels = [0,1,2]
        temp_labels.remove(pred_label)
        if pred_label == gt_label:
            pred_label2 = random.choice(temp_labels)
            temp_labels.remove(pred_label2)
            pred_label3 = temp_labels[0]
        else:
            if random.random() <= conf2/(conf2+conf3):
                pred_label2 = gt_label
                temp_labels.remove(pred_label2)
                pred_label3 = temp_labels[0]
            else:
                pred_label3 = gt_label
                temp_labels.remove(pred_label3)
                pred_label2 = temp_labels[0]

        # if random.random() <= conf2/(conf2+conf3):
        #     pred_label2 = temp_labels

        all_confs.append([conf,conf2,conf3])
        all_pred_labels.append([pred_label, pred_label2, pred_label3])
        all_GT_labels.append(gt_label)



    ## check for calibration
    # bin at level of 0.1
    num_bins = 10
    binned_counts = {}
    for bin in range(num_bins):
        binned_counts[bin] = [0,0]

    for i in range(len(all_GT_labels)):
        confs = all_confs[i]
        pred_labels = all_pred_labels[i]
        GT_label = all_GT_labels[i]

        for j in range(len(confs)):
            bin = get_bin(confs[j],num_bins)
            # print("conf: " + str(confs[j]) + ", bin: " + str(bin))
            pred_label = pred_labels[j]
            if pred_label == GT_label:
                binned_counts[bin][0]+=1
                binned_counts[bin][1]+=1
            else:
                binned_counts[bin][0]+=0
                binned_counts[bin][1]+=1
    
    print("Calibration across all classes")
    for bin in binned_counts:
        bin_lower = bin/num_bins
        bin_upper = (bin+1)/num_bins
        if binned_counts[bin][1] != 0:
            # print(binned_counts[bin])
            print("[" + str(bin_lower) + "," + str(bin_upper) + "]: " + str(float(binned_counts[bin][0]/binned_counts[bin][1])))

    for c in GTs:
        binned_counts = {}
        for bin in range(num_bins):
            binned_counts[bin] = [0,0]

        for i in range(len(all_GT_labels)):
            confs = all_confs[i]
            pred_labels = all_pred_labels[i]
            GT_label = all_GT_labels[i]

            for j in range(len(confs)):
                if c != pred_labels[j]:
                    continue
                bin = get_bin(confs[j],num_bins)
                # print("conf: " + str(confs[j]) + ", bin: " + str(bin))
                pred_label = pred_labels[j]
                if pred_label == GT_label:
                    binned_counts[bin][0]+=1
                    binned_counts[bin][1]+=1
                else:
                    binned_counts[bin][0]+=0
                    binned_counts[bin][1]+=1
        
        print("Calibration for predicted class " + str(c))
        for bin in binned_counts:
            bin_lower = bin/num_bins
            bin_upper = (bin+1)/num_bins
            if binned_counts[bin][1] != 0:
                # print(binned_counts[bin])
                print("[" + str(bin_lower) + "," + str(bin_upper) + "]: " + str(float(binned_counts[bin][0]/binned_counts[bin][1])))






if __name__ == '__main__':
    main()
