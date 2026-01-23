def fuse_probabilities(clinical_prob, image_prob, alpha=0.6):

    if image_prob < 0.15:
        final_prob = clinical_prob
        mode = "Clinical Dominant"
    else:
        final_prob = alpha * clinical_prob + (1-alpha) * image_prob
        mode = "Fusion"

    prediction = "PCOS" if final_prob >= 0.5 else "Non-PCOS"

    return prediction, round(final_prob,3), mode
