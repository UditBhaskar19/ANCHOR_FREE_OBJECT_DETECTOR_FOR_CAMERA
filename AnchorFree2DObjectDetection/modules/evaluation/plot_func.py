import matplotlib.pyplot as plt

def plot_roc(score_threshs, fp_rate_per_image, detection_rate, figsize=(15, 7)):
    decimal = 3
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fp_rate_per_image, detection_rate)

    for i, (fp, r, score) in enumerate(zip(fp_rate_per_image, detection_rate, score_threshs)):
        legend_txt = f"{i}: ( FP Rate: {round(fp, decimal)}, DET Rate: {round(r, decimal)}, Threshold: {round(score, decimal)} )"
        ax.plot(fp, r, marker='o', markersize=6, markeredgewidth=1, label=legend_txt)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=2)
    ax.set_xlabel('False Positives per image')
    ax.set_ylabel('Detection Rate')
    ax.grid(True, which='both', linestyle='--', alpha=0.5, color='gray')
    plt.show()


def plot_precision_recall(score_threshs, f1_score, precision, recall, figsize=(15, 7)):
    decimal = 3
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(precision, recall)

    for i, (f1, p, r, score) in enumerate(zip(f1_score, precision, recall, score_threshs)):
        legend_txt = f"{i}: ( Precision: {round(p, decimal)}, Recall: {round(r, decimal)}, F1: {round(f1, decimal)}, Threshold: {round(score, decimal)} )"
        ax.plot(p, r, marker='o', markersize=6, markeredgewidth=1, label=legend_txt)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=2)
    ax.set_xlabel('Precision')
    ax.set_ylabel('Recall')
    ax.grid(True, which='both', linestyle='--', alpha=0.5, color='gray')
    plt.show()


def compare_roc(
    FP_RATE_PER_IMAGE_LIST_class0, DETECTION_RATE_LIST_class0, 
    FP_RATE_PER_IMAGE_LIST_class1, DETECTION_RATE_LIST_class1,
    _IDX_TO_OBJ_CLASS_):
    plt.plot(FP_RATE_PER_IMAGE_LIST_class0, DETECTION_RATE_LIST_class0, color='blue',  label=_IDX_TO_OBJ_CLASS_[0])
    plt.plot(FP_RATE_PER_IMAGE_LIST_class1, DETECTION_RATE_LIST_class1, color='green', label=_IDX_TO_OBJ_CLASS_[1])
    plt.xlabel('False Positives per image')
    plt.ylabel('Detection Rate')
    plt.title(f'ROC Curve')
    plt.grid(True)
    plt.legend()
    plt.show()


def compare_pr(
    PRECISION_class0, RECALL_class0, 
    PRECISION_class1, RECALL_class1,
    _IDX_TO_OBJ_CLASS_):
    plt.plot(PRECISION_class0, RECALL_class0, color='blue',  label=_IDX_TO_OBJ_CLASS_[0])
    plt.plot(PRECISION_class1, RECALL_class1, color='green', label=_IDX_TO_OBJ_CLASS_[1])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title(f'PR Curve')
    plt.grid(True)
    plt.legend()
    plt.show()