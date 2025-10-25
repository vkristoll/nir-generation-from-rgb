import numpy as np

def track_losses(loss_dict, loss_avgs):
    """
    Appends average losses to tracking lists.
    loss_dict: dictionary of lists to store losses
    loss_avgs: dictionary of average losses for current epoch
    """
    for key in loss_dict:
        loss_dict[key].append(loss_avgs[key])

def export_metrics_csv(loss_dict, lr_dict, time_list, output_path, filename):
    """
    Combines all tracked metrics and exports them to a CSV file.
    """
    combined = list(zip(
        loss_dict["id_A"], loss_dict["id_B"], loss_dict["identity"],
        loss_dict["GAN_AB"], loss_dict["GAN_BA"], loss_dict["GAN"],
        loss_dict["cycle_A"], loss_dict["cycle_B"], loss_dict["cycle"],
        loss_dict["G"], loss_dict["realA"], loss_dict["fakeA"], loss_dict["D_A"],
        loss_dict["realB"], loss_dict["fakeB"], loss_dict["D_B"], loss_dict["D"],
        lr_dict["G"], lr_dict["D_A"], lr_dict["D_B"], time_list
    ))

    header = "#id_A, #id_B, #identity, #GAN_AB, #GAN_BA, #GAN, #cycle_A, #cycle_B, #cycle, #G, #realA, #fakeA, #D_A, #realB, #fakeB, #D_B, #D, #lrG, #lrDA, #lrDB, #time"

    np.savetxt(f"{output_path}/{filename}", np.array(combined), delimiter=",", header=header, fmt="%10.6f")
